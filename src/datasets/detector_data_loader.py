# RoadDefectDetector/src/datasets/detector_data_loader.py
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
from functools import partial  # Импортируем partial

# --- Загрузка Конфигурации ---
# ... (эта часть остается без изменений) ...
_current_dir = os.path.dirname(os.path.abspath(__file__))
_base_config_path = os.path.join(_current_dir, '..', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
except FileNotFoundError as e:
    print(f"ОШИБКА: Не найден один из файлов конфигурации для детектора.")
    BASE_CONFIG = {'master_dataset_path': 'data/Master_Dataset_From_Friend_Fallback'}
    DETECTOR_CONFIG = {'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
                       'train_params': {'batch_size': 2}}
    print("ПРЕДУПРЕЖДЕНИЕ: Файлы конфигурации не найдены...")
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл конфигурации для детектора: {e}")
    BASE_CONFIG = {'master_dataset_path': 'data/Master_Dataset_From_Friend_Fallback'}
    DETECTOR_CONFIG = {'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
                       'train_params': {'batch_size': 2}}
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка чтения YAML...")

TARGET_IMG_HEIGHT = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])[0]
TARGET_IMG_WIDTH = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])[1]
# Эта переменная будет использоваться напрямую в parse_xml_annotation
CLASSES_LIST_GLOBAL_FOR_DETECTOR = DETECTOR_CONFIG.get('classes', ['pit', 'crack'])
BATCH_SIZE = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 2)

_current_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
MASTER_DATASET_PATH_FROM_CONFIG = BASE_CONFIG.get('master_dataset_path', '')
if not MASTER_DATASET_PATH_FROM_CONFIG:
    MASTER_DATASET_PATH_ABS = os.path.join(_current_project_root, "data/Master_Dataset_From_Friend_Fallback")
else:
    if not os.path.isabs(MASTER_DATASET_PATH_FROM_CONFIG):
        MASTER_DATASET_PATH_ABS = os.path.join(_current_project_root, MASTER_DATASET_PATH_FROM_CONFIG)
    else:
        MASTER_DATASET_PATH_ABS = MASTER_DATASET_PATH_FROM_CONFIG


# --- Конец Загрузки Конфигурации ---


def parse_xml_annotation(xml_file_path):  # Убрали current_classes_list из параметров
    # print(f"DEBUG_PARSE: Начало парсинга XML: {xml_file_path}")
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_filename_node = root.find('filename')
        image_filename = image_filename_node.text if image_filename_node is not None else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")
        size_node = root.find('size')
        img_width_xml, img_height_xml = None, None
        if size_node is not None:
            # ... (код извлечения размеров остается) ...
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None and height_node is not None and \
                    width_node.text is not None and height_node.text is not None:
                try:
                    img_width_xml = int(width_node.text)
                    img_height_xml = int(height_node.text)
                except ValueError:
                    print(f"WARNING_PARSE: Некорректные значения width/height в XML {os.path.basename(xml_file_path)}")
        else:
            print(f"WARNING_PARSE: Тег <size> не найден в {os.path.basename(xml_file_path)}")

        objects = []
        object_nodes = root.findall('object')
        for i, obj_node in enumerate(object_nodes):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None:
                print(
                    f"WARNING_PARSE: Объект {i} в {os.path.basename(xml_file_path)} не имеет тега <name> или он пуст. Пропускаем.")
                continue
            class_name = class_name_node.text

            # Используем глобальный список классов
            if class_name not in CLASSES_LIST_GLOBAL_FOR_DETECTOR:
                print(
                    f"WARNING_PARSE: Класс '{class_name}' (объект {i}) не в списке {CLASSES_LIST_GLOBAL_FOR_DETECTOR} для {os.path.basename(xml_file_path)}. Пропускаем.")
                continue
            class_id = CLASSES_LIST_GLOBAL_FOR_DETECTOR.index(class_name)

            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None:
                print(
                    f"WARNING_PARSE: Объект {i} ('{class_name}') в {os.path.basename(xml_file_path)} не имеет тега <bndbox>. Пропускаем.")
                continue
            try:
                xmin = float(bndbox_node.find('xmin').text)
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text)
                ymax = float(bndbox_node.find('ymax').text)
            except (ValueError, AttributeError, TypeError):
                print(
                    f"WARNING_PARSE: Некорректные или отсутствующие координаты bndbox у объекта {i} ('{class_name}') в {os.path.basename(xml_file_path)}. Пропускаем.")
                continue
            if xmin >= xmax or ymin >= ymax:
                print(
                    f"WARNING_PARSE: Invalid bbox (xmin>=xmax or ymin>=ymax) у объекта {i} ('{class_name}') в {os.path.basename(xml_file_path)}. Пропускаем.")
                continue
            objects.append({
                "class_id": class_id,
                "class_name": class_name,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
            })
        return objects, img_width_xml, img_height_xml, image_filename
    except ET.ParseError as e_parse:
        print(f"ERROR_PARSE: Ошибка парсинга XML файла {xml_file_path}: {e_parse}")
        return None, None, None, None
    except Exception as e:
        print(f"ERROR_PARSE: Непредвиденная ошибка при парсинге {xml_file_path}: {e}")
        return None, None, None, None


@tf.function
def preprocess_image_and_boxes(image, boxes, target_height_tf, target_width_tf):
    # ... (код этой функции остается БЕЗ ИЗМЕНЕНИЙ) ...
    original_height_f = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    original_width_f = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    image_resized = tf.image.resize(image, [target_height_tf, target_width_tf])
    image_processed = image_resized / 255.0
    num_boxes = tf.shape(boxes)[0]
    if num_boxes > 0:
        safe_original_width_f = tf.maximum(original_width_f, 1.0)
        safe_original_height_f = tf.maximum(original_height_f, 1.0)
        scaled_boxes_norm = tf.stack([
            boxes[:, 0] / safe_original_width_f,
            boxes[:, 1] / safe_original_height_f,
            boxes[:, 2] / safe_original_width_f,
            boxes[:, 3] / safe_original_height_f
        ], axis=-1)
        scaled_boxes_norm = tf.clip_by_value(scaled_boxes_norm, 0.0, 1.0)
    else:
        scaled_boxes_norm = tf.zeros((0, 4), dtype=tf.float32)
    return image_processed, scaled_boxes_norm


# НОВАЯ ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ИЗ КОНФИГА ДЕТЕКТОРА
MAX_BOXES_PER_IMAGE_FROM_CONFIG = DETECTOR_CONFIG.get('max_boxes_per_image', 10)


def load_and_prepare_example_py_func(image_path_tensor, xml_path_tensor,
                                     target_height_for_py_func,
                                     target_width_for_py_func):
    image_path = image_path_tensor.numpy().decode('utf-8')
    xml_path = xml_path_tensor.numpy().decode('utf-8')

    # --- Загрузка изображения (как раньше) ---
    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path).convert('RGB')
        image_np = np.array(pil_image, dtype=np.float32)
    except Exception as e:
        # print(f"PY_FUNC Error loading image {image_path}: {e}")
        # Возвращаем заглушки нужной формы для y_true
        num_features_per_box = 4 + 1 + len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
        return np.zeros((target_height_for_py_func, target_width_for_py_func, 3), dtype=np.float32), \
            np.zeros((MAX_BOXES_PER_IMAGE_FROM_CONFIG, num_features_per_box), dtype=np.float32)

    # --- Парсинг XML (как раньше) ---
    objects, _, _, _ = parse_xml_annotation(xml_path)  # Использует глобальный CLASSES_LIST_GLOBAL_FOR_DETECTOR

    if objects is None:
        # print(f"PY_FUNC Error parsing XML {xml_path}")
        num_features_per_box = 4 + 1 + len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
        return np.zeros((target_height_for_py_func, target_width_for_py_func, 3), dtype=np.float32), \
            np.zeros((MAX_BOXES_PER_IMAGE_FROM_CONFIG, num_features_per_box), dtype=np.float32)

    # --- Предобработка изображения и масштабирование реальных рамок (как раньше) ---
    boxes_list_pixels = []
    class_ids_list_for_gt = []  # Для one-hot encoding классов
    if objects:
        for obj in objects:
            boxes_list_pixels.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            class_ids_list_for_gt.append(obj['class_id'])

    if not boxes_list_pixels:
        boxes_np_pixels = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes_np_pixels = np.array(boxes_list_pixels, dtype=np.float32)

    image_tensor_in_py = tf.convert_to_tensor(image_np, dtype=tf.float32)
    boxes_tensor_pixels_in_py = tf.convert_to_tensor(boxes_np_pixels, dtype=np.float32)

    image_processed_tensor, scaled_boxes_norm_tensor = preprocess_image_and_boxes(
        image_tensor_in_py, boxes_tensor_pixels_in_py,
        tf.constant(target_height_for_py_func, dtype=tf.int32),
        tf.constant(target_width_for_py_func, dtype=tf.int32)
    )
    # scaled_boxes_norm_tensor имеет форму [num_actual_objects, 4] с нормализованными xmin,ymin,xmax,ymax

    # --- НОВАЯ ЧАСТЬ: Формирование y_true для модели SimpleObjectDetector ---
    num_actual_objects = tf.shape(scaled_boxes_norm_tensor)[0].numpy()  # Количество реальных объектов на картинке
    num_classes = len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
    num_features_per_box = 4 + 1 + num_classes  # xmin,ymin,xmax,ymax, obj_score, class_probs...

    # Создаем y_true массив, заполненный нулями
    y_true_target_np = np.zeros((MAX_BOXES_PER_IMAGE_FROM_CONFIG, num_features_per_box), dtype=np.float32)

    if num_actual_objects > 0:
        # Берем только первые MAX_BOXES_PER_IMAGE объектов, если их больше
        objects_to_process = min(num_actual_objects, MAX_BOXES_PER_IMAGE_FROM_CONFIG)

        for i in range(objects_to_process):
            # Координаты (пока просто xmin, ymin, xmax, ymax нормализованные)
            # В более сложных моделях здесь будет кодирование относительно якорей/сетки (tx,ty,tw,th)
            y_true_target_np[i, 0:4] = scaled_boxes_norm_tensor[i].numpy()

            # Objectness score (1 для реальных объектов, 0 для пустых слотов)
            y_true_target_np[i, 4] = 1.0

            # Классы (one-hot encoding)
            class_id = class_ids_list_for_gt[i]  # Получаем class_id для текущего объекта
            y_true_target_np[i, 5 + class_id] = 1.0

            # Остальные слоты (от num_actual_objects до MAX_BOXES_PER_IMAGE) уже заполнены нулями,
    # что означает objectness_score = 0.

    return image_processed_tensor.numpy(), y_true_target_np


# В функции load_and_prepare_example_tf_wrapper ИЗМЕНИ Tout для y_true:
def load_and_prepare_example_tf_wrapper(image_path_tensor, xml_path_tensor,
                                        target_height_param, target_width_param):
    # Определяем количество фичей на бокс для Tout
    num_output_features_per_box = 4 + 1 + len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)

    img_processed_np, y_true_np = tf.py_function(
        func=load_and_prepare_example_py_func,
        inp=[image_path_tensor, xml_path_tensor, target_height_param, target_width_param],
        Tout=[tf.float32, tf.float32]  # Типы остаются теми же
    )

    img_processed_np.set_shape([target_height_param, target_width_param, 3])
    # Новая форма для y_true!
    y_true_np.set_shape([MAX_BOXES_PER_IMAGE_FROM_CONFIG, num_output_features_per_box])

    return img_processed_np, y_true_np


def create_detector_tf_dataset(image_paths_list, xml_paths_list, batch_size,
                               target_height_ds=TARGET_IMG_HEIGHT,
                               target_width_ds=TARGET_IMG_WIDTH,
                               # classes_list_ds больше не нужен как параметр, используется глобальный
                               shuffle=True, augment=False):
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(xml_paths_list, (list, tuple)):
        raise ValueError("image_paths_list и xml_paths_list должны быть Python списками или кортежами.")
    if len(image_paths_list) != len(xml_paths_list):
        raise ValueError("Количество путей к изображениям и XML должно совпадать.")

    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(image_paths_list, dtype=tf.string),
        tf.constant(xml_paths_list, dtype=tf.string)
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths_list), reshuffle_each_iteration=True)

    # Используем functools.partial или лямбду, чтобы передать параметры в map
    map_func = partial(load_and_prepare_example_tf_wrapper,
                       # classes_list_param убран
                       target_height_param=target_height_ds,
                       target_width_param=target_width_ds)

    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Пример использования (для тестирования этого файла) ---
if __name__ == '__main__':
    # ... (секция if __name__ == "__main__": остается БЕЗ ИЗМЕНЕНИЙ,
    # так как create_detector_tf_dataset больше не принимает classes_list_ds,
    # а load_and_prepare_example_tf_wrapper не принимает classes_list_param)
    # ... код из предыдущего ответа для if __name__ == "__main__": ...
    print(f"--- Тестирование detector_data_loader.py ---")
    print(f"Используется TARGET_IMG_HEIGHT: {TARGET_IMG_HEIGHT}, TARGET_IMG_WIDTH: {TARGET_IMG_WIDTH}")
    print(
        f"Используются классы для детектора: {CLASSES_LIST_GLOBAL_FOR_DETECTOR}")  # Используем новую глобальную переменную

    TEST_IMAGES_PARENT_SUBDIR = "Defective_Road_Images"
    IMAGES_DIR_EXAMPLE = os.path.join(MASTER_DATASET_PATH_ABS, TEST_IMAGES_PARENT_SUBDIR, "JPEGImages")
    ANNOTATIONS_DIR_EXAMPLE = os.path.join(MASTER_DATASET_PATH_ABS, TEST_IMAGES_PARENT_SUBDIR, "Annotations")

    print(f"\nТестовые пути для детектора:")
    print(f"  Изображения из: {IMAGES_DIR_EXAMPLE}")
    print(f"  Аннотации из: {ANNOTATIONS_DIR_EXAMPLE}")

    current_batch_size = BATCH_SIZE

    example_image_paths = []
    example_xml_paths = []

    if not os.path.isdir(IMAGES_DIR_EXAMPLE) or not os.path.isdir(ANNOTATIONS_DIR_EXAMPLE):
        print(f"ОШИБКА: Тестовые директории для детектора не найдены.")
    else:
        test_files_bases = ['defective_road_01', 'normal_road_01']
        for base_name in test_files_bases:
            found_img_for_base = False
            img_path_abs_candidate = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path_try = os.path.join(IMAGES_DIR_EXAMPLE, base_name + ext)
                if os.path.exists(img_path_try):
                    img_path_abs_candidate = img_path_try
                    found_img_for_base = True
                    break
                img_path_try_upper = os.path.join(IMAGES_DIR_EXAMPLE, base_name + ext.upper())
                if os.path.exists(img_path_try_upper):
                    img_path_abs_candidate = img_path_try_upper
                    found_img_for_base = True
                    break
            if found_img_for_base:
                xml_file_abs = os.path.join(ANNOTATIONS_DIR_EXAMPLE, base_name + ".xml")
                if os.path.exists(xml_file_abs):
                    example_image_paths.append(img_path_abs_candidate)
                    example_xml_paths.append(xml_file_abs)
                    print(
                        f"  INFO_TEST: Добавлен для теста: {os.path.basename(img_path_abs_candidate)} и {os.path.basename(xml_file_abs)}")
                else:
                    print(
                        f"  WARNING_TEST: XML {xml_file_abs} не найден для изображения {img_path_abs_candidate if img_path_abs_candidate else base_name}")
            else:
                print(
                    f"  WARNING_TEST: Изображение {base_name} с расширениями .jpg/.jpeg/.png не найдено в {IMAGES_DIR_EXAMPLE}")

        if not example_image_paths:
            print("Не найдено совпадающих пар изображение/аннотация для теста детектора.")
        else:
            print(f"\nНайдено {len(example_image_paths)} пар изображение/аннотация для теста детектора.")
            effective_batch_size = min(current_batch_size, len(example_image_paths))
            if effective_batch_size == 0 and len(example_image_paths) > 0: effective_batch_size = 1
            print(f"Будет протестировано на {len(example_image_paths)} файлах, батч: {effective_batch_size}.")
            if effective_batch_size > 0:
                try:
                    dataset = create_detector_tf_dataset(
                        example_image_paths,
                        example_xml_paths,
                        effective_batch_size,
                        target_height_ds=TARGET_IMG_HEIGHT,
                        target_width_ds=TARGET_IMG_WIDTH,
                        # classes_list_ds больше не передается, используется глобальный
                        shuffle=False
                    )
                    print("\nПример батча из датасета детектора:")
                    for i, (images_batch, labels_batch) in enumerate(dataset.take(1)):
                        print(f"\n--- Батч {i + 1} (детектор) ---")
                        print("Форма батча изображений:", images_batch.shape)
                        if isinstance(labels_batch, tf.RaggedTensor):
                            print("Тип меток: RaggedTensor")
                            print("Форма батча меток (батч, кол-во объектов, признаки):", labels_batch.shape)
                            if labels_batch.nrows() > 0:
                                for k in range(labels_batch.nrows().numpy()):
                                    print(f"  Метки для изображения {k} в батче:")
                                    if labels_batch[k].shape[0] > 0:
                                        print(labels_batch[k].to_tensor(default_value=-1).numpy())
                                    else:
                                        print("    (объектов нет)")
                        else:
                            print("Тип меток: Tensor")
                            print("Форма батча меток (простых, детектор):", labels_batch.shape)
                            if labels_batch.shape[0] > 0:
                                for k in range(labels_batch.shape[0]):
                                    print(f"  Метки для изображения {k} в батче:")
                                    if labels_batch[k].shape.rank > 0 and labels_batch[k].shape[0] > 0:
                                        print(labels_batch[k].numpy())
                                    else:
                                        print("    (объектов нет или некорректная форма)")
                            else:
                                print("  Метки для первого изображения в батче (детектор): Батч меток пуст.")
                except Exception as e_dataset:
                    print(f"ОШИБКА при итерации по датасету детектора: {e_dataset}")
                    import traceback

                    traceback.print_exc()
            else:
                print("Недостаточно файлов для создания батча.")
            print("\n--- Тестирование detector_data_loader.py завершено ---")