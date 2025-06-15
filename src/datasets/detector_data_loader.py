# src/datasets/detector_data_loader.py
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
from functools import partial

# --- Загрузка Конфигурации ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))  # src/datasets/
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))  # Корень проекта

_base_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS = True

try:
    # print(f"INFO: Попытка загрузить base_config из: {_base_config_path}")
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict):
        # print(f"ПРЕДУПРЕЖДЕНИЕ: base_config.yaml ({_base_config_path}) пуст или имеет неверный формат. Используются частичные дефолты.")
        BASE_CONFIG = {}
        CONFIG_LOAD_SUCCESS = False
    # print(f"INFO: base_config загружен: {bool(BASE_CONFIG)}")
except FileNotFoundError:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: Файл base_config.yaml не найден по пути: {_base_config_path}. Используются частичные дефолты.")
    CONFIG_LOAD_SUCCESS = False
except yaml.YAMLError as e:
    print(f"ОШИБКА YAML при чтении base_config.yaml: {e}. Используются частичные дефолты.")
    CONFIG_LOAD_SUCCESS = False

try:
    # print(f"INFO: Попытка загрузить detector_config из: {_detector_config_path}")
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict):
        # print(f"ПРЕДУПРЕЖДЕНИЕ: detector_config.yaml ({_detector_config_path}) пуст или имеет неверный формат. Используются частичные дефолты.")
        DETECTOR_CONFIG = {}
        CONFIG_LOAD_SUCCESS = False
    # print(f"INFO: detector_config загружен: {bool(DETECTOR_CONFIG)}")
except FileNotFoundError:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: Файл detector_config.yaml не найден по пути: {_detector_config_path}. Используются частичные дефолты.")
    CONFIG_LOAD_SUCCESS = False
except yaml.YAMLError as e:
    print(f"ОШИБКА YAML при чтении detector_config.yaml: {e}. Используются частичные дефолты.")
    CONFIG_LOAD_SUCCESS = False

# --- Параметры из Конфигов с дефолтами ---
_input_shape_list = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT = _input_shape_list[0]
TARGET_IMG_WIDTH = _input_shape_list[1]

CLASSES_LIST_GLOBAL_FOR_DETECTOR = DETECTOR_CONFIG.get('classes', ['pit',
                                                                   'treshina'])  # Убедись, что это ['pit', 'crack'] если ты изменил имя
NUM_CLASSES_DETECTOR = DETECTOR_CONFIG.get('num_classes', len(CLASSES_LIST_GLOBAL_FOR_DETECTOR))
if NUM_CLASSES_DETECTOR != len(CLASSES_LIST_GLOBAL_FOR_DETECTOR):
    NUM_CLASSES_DETECTOR = len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)

BATCH_SIZE_FROM_CONFIG = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 1)

NETWORK_STRIDE = 16
GRID_HEIGHT = TARGET_IMG_HEIGHT // NETWORK_STRIDE
GRID_WIDTH = TARGET_IMG_WIDTH // NETWORK_STRIDE

ANCHORS_WH_NORMALIZED_LIST = DETECTOR_CONFIG.get('anchors_wh_normalized', [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]])
ANCHORS_WH_NORMALIZED = np.array(ANCHORS_WH_NORMALIZED_LIST, dtype=np.float32)
NUM_ANCHORS_PER_LOCATION = ANCHORS_WH_NORMALIZED.shape[0]
config_num_anchors = DETECTOR_CONFIG.get('num_anchors_per_location')
if config_num_anchors is not None and config_num_anchors != NUM_ANCHORS_PER_LOCATION:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: num_anchors_per_location ({config_num_anchors}) не совпадает с anchors_wh_normalized ({NUM_ANCHORS_PER_LOCATION}).")

_master_dataset_path_from_cfg = BASE_CONFIG.get('master_dataset_path', 'data/Master_Dataset_Fallback')
if not os.path.isabs(_master_dataset_path_from_cfg):
    MASTER_DATASET_PATH_ABS = os.path.join(_project_root_dir, _master_dataset_path_from_cfg)
else:
    MASTER_DATASET_PATH_ABS = _master_dataset_path_from_cfg

_images_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')


# --- Конец Загрузки Конфигурации ---


def parse_xml_annotation(xml_file_path, classes_list):  # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
    """Парсит PASCAL VOC XML файл и возвращает список объектов и размеры изображения."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        image_filename_node = root.find('filename')
        # Используем имя самого XML файла для формирования имени изображения, если тег filename отсутствует или некорректен
        image_filename = image_filename_node.text if image_filename_node is not None and image_filename_node.text else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")  # Запасной вариант

        size_node = root.find('size')
        img_width_xml, img_height_xml = None, None  # Инициализируем
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None and height_node is not None and width_node.text is not None and height_node.text is not None:
                try:  # Добавим try-except для преобразования в int
                    img_width_xml = int(width_node.text)
                    img_height_xml = int(height_node.text)
                    if img_width_xml <= 0 or img_height_xml <= 0:  # Проверка на валидность
                        # print(f"DEBUG_PARSE_XML: Невалидные размеры в XML {os.path.basename(xml_file_path)}: w={width_node.text}, h={height_node.text}. Размеры будут None.")
                        img_width_xml, img_height_xml = None, None
                except ValueError:
                    # print(f"DEBUG_PARSE_XML: Не удалось преобразовать размеры в int в XML {os.path.basename(xml_file_path)}: w='{width_node.text}', h='{height_node.text}'. Размеры будут None.")
                    img_width_xml, img_height_xml = None, None
        # else:
        # print(f"DEBUG_PARSE_XML: Тег <size> не найден в {os.path.basename(xml_file_path)}. Размеры будут None.")

        objects = []
        for obj_node in root.findall('object'):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None:  # Проверка на существование тега и текста
                # print(f"DEBUG_PARSE_XML: Пропущен объект без имени в {os.path.basename(xml_file_path)}")
                continue
            class_name = class_name_node.text

            if class_name not in classes_list:  # Используем переданный classes_list
                # print(f"DEBUG_PARSE_XML: Неизвестный класс '{class_name}' в {os.path.basename(xml_file_path)}. Пропускаем объект. Ожидаемые классы: {classes_list}")
                continue
            class_id = classes_list.index(class_name)

            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None:  # Проверка на существование bndbox
                # print(f"DEBUG_PARSE_XML: Пропущен объект '{class_name}' без bndbox в {os.path.basename(xml_file_path)}")
                continue

            # Более безопасное извлечение координат
            try:
                xmin = float(bndbox_node.find('xmin').text)
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text)
                ymax = float(bndbox_node.find('ymax').text)
            except (ValueError, AttributeError, TypeError) as e_coord:  # AttributeError/TypeError если find вернул None
                # print(f"DEBUG_PARSE_XML: Ошибка координат для объекта '{class_name}' в {os.path.basename(xml_file_path)}: {e_coord}. Пропускаем объект.")
                continue

            if xmin >= xmax or ymin >= ymax:
                # print(f"DEBUG_PARSE_XML: Некорректные координаты bbox (xmin>=xmax or ymin>=ymax) в {os.path.basename(xml_file_path)} для {class_name}. Пропускаем объект.")
                continue

            # Клиппинг по размерам изображения, если они известны из XML
            if img_width_xml and img_height_xml:
                xmin = max(0, min(xmin, img_width_xml))
                ymin = max(0, min(ymin, img_height_xml))
                xmax = max(0, min(xmax, img_width_xml))
                ymax = max(0, min(ymax, img_height_xml))
                # Повторная проверка после клиппинга, так как клиппинг мог сделать их некорректными
                if xmin >= xmax or ymin >= ymax:
                    # print(f"DEBUG_PARSE_XML: Координаты bbox стали некорректными после клиппинга в {os.path.basename(xml_file_path)} для {class_name}. Пропускаем объект.")
                    continue

            objects.append({
                "class_id": class_id,
                "class_name": class_name,
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
            })
        return objects, img_width_xml, img_height_xml, image_filename
    except ET.ParseError as e_parse:  # Ошибка парсинга самого XML файла
        print(f"ERROR_PARSE_XML: Ошибка парсинга XML файла {os.path.basename(xml_file_path)}: {e_parse}")
    except Exception as e_generic:  # Другие непредвиденные ошибки
        print(f"ERROR_PARSE_XML: Непредвиденная ошибка при парсинге {os.path.basename(xml_file_path)}: {e_generic}")
    return None, None, None, None  # Возвращаем None в случае любой ошибки


@tf.function
def preprocess_image_and_boxes(image, boxes, target_height_tf, target_width_tf):
    # КОД preprocess_image_and_boxes ИЗ ПРЕДЫДУЩЕГО ОТВЕТА
    original_height_f = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    original_width_f = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    image_resized = tf.image.resize(image, [target_height_tf, target_width_tf])
    image_processed = image_resized / 255.0
    num_boxes = tf.shape(boxes)[0]
    if num_boxes > 0:
        safe_original_width_f = tf.maximum(original_width_f, 1e-6)
        safe_original_height_f = tf.maximum(original_height_f, 1e-6)
        scaled_boxes_norm = tf.stack([
            boxes[:, 0] / safe_original_width_f, boxes[:, 1] / safe_original_height_f,
            boxes[:, 2] / safe_original_width_f, boxes[:, 3] / safe_original_height_f
        ], axis=-1)
        scaled_boxes_norm = tf.clip_by_value(scaled_boxes_norm, 0.0, 1.0)
    else:
        scaled_boxes_norm = tf.zeros((0, 4), dtype=tf.float32)
    return image_processed, scaled_boxes_norm


def calculate_iou_numpy(box_wh, anchors_wh):
    # КОД calculate_iou_numpy ИЗ ПРЕДЫДУЩЕГО ОТВЕТА
    box_wh = np.array(box_wh)
    anchors_wh = np.array(anchors_wh)
    inter_w = np.minimum(box_wh[0], anchors_wh[:, 0])
    inter_h = np.minimum(box_wh[1], anchors_wh[:, 1])
    intersection_area = inter_w * inter_h
    box_area = box_wh[0] * box_wh[1]
    anchors_area = anchors_wh[:, 0] * anchors_wh[:, 1]
    union_area = box_area + anchors_area - intersection_area
    iou = intersection_area / (union_area + 1e-6)
    return iou


def load_and_prepare_detector_y_true_py_func(image_path_tensor, xml_path_tensor,
                                             py_target_height, py_target_width,
                                             py_grid_h, py_grid_w,
                                             py_anchors_wh, py_num_classes):  # py_anchors_wh это ANCHORS_WH_NORMALIZED
    """
    Python функция, которую будем оборачивать в tf.py_function.
    Принимает пути в виде байтовых строк.
    Формирует y_true для обучения детектора.
    """
    image_path = image_path_tensor.numpy().decode('utf-8')
    xml_path = xml_path_tensor.numpy().decode('utf-8')

    y_true_output_shape = (py_grid_h, py_grid_w, py_anchors_wh.shape[0], 5 + py_num_classes)
    image_output_shape = (py_target_height, py_target_width, 3)

    # --- ОТЛАДОЧНЫЙ ВЫВОД: Начало обработки файла ---
    print(f"\nDEBUG_PY_FUNC: Обработка файла: {os.path.basename(image_path)}")
    print(f"  XML путь: {os.path.basename(xml_path)}")
    print(f"  Целевые размеры изображения (H,W): ({py_target_height}, {py_target_width})")
    print(f"  Размеры сетки (H,W): ({py_grid_h}, {py_grid_w})")
    print(f"  Количество якорей на ячейку: {py_anchors_wh.shape[0]}")
    print(f"  Размеры якорей (W_norm, H_norm):\n{py_anchors_wh}")
    print(f"  Количество классов: {py_num_classes}")
    # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path).convert('RGB')
        image_np_original = np.array(pil_image, dtype=np.float32)
    except Exception as e_img_load:
        print(f"DEBUG_PY_FUNC ERROR: Не удалось загрузить изображение {image_path}: {e_img_load}")
        return np.zeros(image_output_shape, dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

    objects, xml_img_width_parsed, xml_img_height_parsed, _ = parse_xml_annotation(xml_path,
                                                                                   CLASSES_LIST_GLOBAL_FOR_DETECTOR)

    if objects is None:
        print(
            f"DEBUG_PY_FUNC INFO: Ошибка парсинга XML {xml_path} или объекты не найдены (возвращено None из parse_xml_annotation).")
        return np.zeros(image_output_shape, dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

    # --- ОТЛАДОЧНЫЙ ВЫВОД: Объекты из XML ---
    if objects:
        print(f"  DEBUG_PY_FUNC: Найдено {len(objects)} объектов в XML:")
        for i_obj, obj_data in enumerate(objects):
            print(f"    Объект {i_obj}: Класс='{obj_data['class_name']}' (ID={obj_data['class_id']}), "
                  f"Box_pixels=[{obj_data['xmin']:.0f}, {obj_data['ymin']:.0f}, {obj_data['xmax']:.0f}, {obj_data['ymax']:.0f}]")
    else:
        print(f"  DEBUG_PY_FUNC: В XML {os.path.basename(xml_path)} не найдено объектов. Формируется пустой y_true.")
    # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

    boxes_list_pixels = []
    class_ids_list_for_gt = []
    if objects:
        for obj in objects:
            boxes_list_pixels.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            class_ids_list_for_gt.append(obj['class_id'])

    image_tensor_in_py = tf.convert_to_tensor(image_np_original, dtype=tf.float32)
    boxes_tensor_pixels_in_py = tf.constant(boxes_list_pixels, dtype=tf.float32) if objects else tf.zeros((0, 4),
                                                                                                          dtype=tf.float32)

    image_processed_tensor, scaled_gt_boxes_norm_tensor = preprocess_image_and_boxes(
        image_tensor_in_py, boxes_tensor_pixels_in_py,
        tf.constant(py_target_height, dtype=tf.int32),
        tf.constant(py_target_width, dtype=tf.int32)
    )

    # --- ОТЛАДОЧНЫЙ ВЫВОД: Нормализованные GT боксы ---
    if tf.shape(scaled_gt_boxes_norm_tensor)[0] > 0:
        print(
            f"  DEBUG_PY_FUNC: Нормализованные GT боксы (xmin,ymin,xmax,ymax) после preprocess_image_and_boxes:\n{scaled_gt_boxes_norm_tensor.numpy()}")
    # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

    y_true_target_np = np.zeros(y_true_output_shape, dtype=np.float32)

    if objects and tf.shape(scaled_gt_boxes_norm_tensor)[0] > 0:
        gt_boxes_xywh_norm_list = []
        for i in range(scaled_gt_boxes_norm_tensor.shape[0]):
            box_norm_np = scaled_gt_boxes_norm_tensor[i].numpy()
            width_n = float(box_norm_np[2]) - float(box_norm_np[0])
            height_n = float(box_norm_np[3]) - float(box_norm_np[1])
            x_center_n = float(box_norm_np[0]) + width_n / 2.0
            y_center_n = float(box_norm_np[1]) + height_n / 2.0
            gt_boxes_xywh_norm_list.append([x_center_n, y_center_n, width_n, height_n])

        gt_boxes_xywh_norm_np = np.array(gt_boxes_xywh_norm_list, dtype=np.float32)

        # --- ОТЛАДОЧНЫЙ ВЫВОД: GT боксы в формате XYWH_norm ---
        print(
            f"  DEBUG_PY_FUNC: GT боксы (xc_n, yc_n, w_n, h_n) для сопоставления с якорями:\n{np.round(gt_boxes_xywh_norm_np, 3)}")
        # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

        anchor_assigned_mask = np.zeros((py_grid_h, py_grid_w, py_anchors_wh.shape[0]), dtype=bool)

        for i in range(gt_boxes_xywh_norm_np.shape[0]):
            gt_xc_n, gt_yc_n, gt_w_n, gt_h_n = gt_boxes_xywh_norm_np[i]
            gt_class_id = int(class_ids_list_for_gt[i])

            # --- ОТЛАДОЧНЫЙ ВЫВОД: Обработка GT объекта ---
            print(
                f"    DEBUG_PY_FUNC: Обработка GT объекта {i}: Класс ID={gt_class_id} ({CLASSES_LIST_GLOBAL_FOR_DETECTOR[gt_class_id]}), XYWH_norm=[{gt_xc_n:.3f}, {gt_yc_n:.3f}, {gt_w_n:.3f}, {gt_h_n:.3f}]")
            # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

            grid_x_center_float = gt_xc_n * float(py_grid_w)
            grid_y_center_float = gt_yc_n * float(py_grid_h)
            grid_x_idx = int(grid_x_center_float)
            grid_y_idx = int(grid_y_center_float)

            grid_x_idx = min(grid_x_idx, py_grid_w - 1)
            grid_y_idx = min(grid_y_idx, py_grid_h - 1)

            # --- ОТЛАДОЧНЫЙ ВЫВОД: Ячейка сетки ---
            print(
                f"      Центр объекта в сетке (float): ({grid_x_center_float:.2f}, {grid_y_center_float:.2f}) -> Ячейка_idx ({grid_y_idx}, {grid_x_idx})")
            # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

            best_iou = -1.0  # Инициализируем float
            best_anchor_idx = -1
            gt_box_shape_wh_for_iou = [gt_w_n, gt_h_n]

            ious = calculate_iou_numpy(gt_box_shape_wh_for_iou, py_anchors_wh)
            best_anchor_idx = int(np.argmax(ious))
            best_iou = float(ious[best_anchor_idx])

            # --- ОТЛАДОЧНЫЙ ВЫВОД: Выбор якоря ---
            print(
                f"      IoUs с якорями (по формам W,H): {np.round(ious, 3)}, Лучший якорь_idx: {best_anchor_idx}, Best IoU: {best_iou:.3f}")
            # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

            if best_iou >= 0.0 and not anchor_assigned_mask[
                grid_y_idx, grid_x_idx, best_anchor_idx]:  # Порог IoU можно будет добавить позже (> 0 это заглушка)
                anchor_assigned_mask[grid_y_idx, grid_x_idx, best_anchor_idx] = True

                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 4] = 1.0

                tx = grid_x_center_float - float(grid_x_idx)
                ty = grid_y_center_float - float(grid_y_idx)

                anchor_w_norm_val = float(py_anchors_wh[best_anchor_idx, 0])
                anchor_h_norm_val = float(py_anchors_wh[best_anchor_idx, 1])

                safe_gt_w_n = max(gt_w_n, 1e-9)
                safe_gt_h_n = max(gt_h_n, 1e-9)
                safe_anchor_w = max(anchor_w_norm_val, 1e-9)
                safe_anchor_h = max(anchor_h_norm_val, 1e-9)

                tw = np.log(safe_gt_w_n / safe_anchor_w)
                th = np.log(safe_gt_h_n / safe_anchor_h)

                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 0:4] = [tx, ty, tw, th]
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 5 + gt_class_id] = 1.0

                # --- ОТЛАДОЧНЫЙ ВЫВОД: Назначение якоря ---
                print(
                    f"      --> НАЗНАЧЕН Якорь_idx {best_anchor_idx} в ячейке ({grid_y_idx},{grid_x_idx}) для GT объекта {i}")
                print(f"          tx={tx:.2f}, ty={ty:.2f}, tw={tw:.2f}, th={th:.2f}")
                print(f"          Objectness=1.0, ClassID={gt_class_id} (one-hot index {5 + gt_class_id} = 1.0)")
                # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---
            else:
                # --- ОТЛАДОЧНЫЙ ВЫВОД: Якорь не назначен ---
                print(
                    f"      --> Якорь_idx {best_anchor_idx} в ячейке ({grid_y_idx},{grid_x_idx}) НЕ назначен для GT объекта {i} (уже занят или IoU={best_iou:.3f} недостаточно).")
                # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

    return image_processed_tensor.numpy(), y_true_target_np


def load_and_prepare_detector_y_true_tf_wrapper(image_path_tensor, xml_path_tensor):
    # КОД load_and_prepare_detector_y_true_tf_wrapper ИЗ ПРЕДЫДУЩЕГО ОТВЕТА
    # (с передачей TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, GRID_HEIGHT, GRID_WIDTH, ANCHORS_WH_NORMALIZED, NUM_CLASSES_DETECTOR в inp)
    img_processed_np, y_true_np = tf.py_function(
        func=load_and_prepare_detector_y_true_py_func,
        inp=[image_path_tensor, xml_path_tensor,
             TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH,
             GRID_HEIGHT, GRID_WIDTH,
             ANCHORS_WH_NORMALIZED, NUM_CLASSES_DETECTOR],
        Tout=[tf.float32, tf.float32]
    )
    img_processed_np.set_shape([TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3])
    y_true_output_shape_tf = (GRID_HEIGHT, GRID_WIDTH, NUM_ANCHORS_PER_LOCATION, 5 + NUM_CLASSES_DETECTOR)
    y_true_np.set_shape(y_true_output_shape_tf)
    return img_processed_np, y_true_np


def create_detector_tf_dataset(image_paths_list, xml_paths_list, batch_size, shuffle=True):
    # КОД create_detector_tf_dataset ИЗ ПРЕДЫДУЩЕГО ОТВЕТА
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
    dataset = dataset.map(
        load_and_prepare_detector_y_true_tf_wrapper,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    # Импортируем функцию визуализации из нашего нового модуля
    import sys

    _utils_path = os.path.abspath(os.path.join(_current_script_dir, '..', 'utils'))
    if _utils_path not in sys.path:
        sys.path.insert(0, _utils_path)
    try:
        from plot_utils import visualize_data_sample

        VISUALIZATION_ENABLED = True
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: Модуль plot_utils не найден. Визуализация будет отключена.")
        print(f"  Ожидался путь к utils: {_utils_path}")
        VISUALIZATION_ENABLED = False
    except Exception as e_imp:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при импорте plot_utils: {e_imp}. Визуализация будет отключена.")
        VISUALIZATION_ENABLED = False

    if not CONFIG_LOAD_SUCCESS:
        print(
            "\n!!! ВНИМАНИЕ: Конфигурационные файлы не были загружены корректно. Тестирование может быть неточным или использовать дефолты.")

    print(f"--- Тестирование detector_data_loader.py (3 СЛУЧАЙНЫХ ФАЙЛА) ---")
    print(f"Параметры сетки: GRID_HEIGHT={GRID_HEIGHT}, GRID_WIDTH={GRID_WIDTH}")
    print(f"Якоря (W_norm, H_norm) используются внутри py_func:\n{ANCHORS_WH_NORMALIZED}")
    print(f"Количество якорей на ячейку: {NUM_ANCHORS_PER_LOCATION}")
    print(f"Количество классов: {NUM_CLASSES_DETECTOR} ({CLASSES_LIST_GLOBAL_FOR_DETECTOR})")

    _master_dataset_abs_for_test = MASTER_DATASET_PATH_ABS

    # Имена подпапок категорий в Master_Dataset (из base_config.yaml)
    source_subfolder_keys_for_test = [
        BASE_CONFIG.get('source_defective_road_img_parent_subdir', 'Defective_Road_Images'),
        BASE_CONFIG.get('source_normal_road_img_parent_subdir', 'Normal_Road_Images'),
        BASE_CONFIG.get('source_not_road_img_parent_subdir', 'Not_Road_Images')
    ]
    images_subdir_name_in_master = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
    annotations_subdir_name_in_master = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')

    print(f"\nТестовые пути для detector_data_loader.py (сканируем Master_Dataset):")
    print(f"  Корень Master_Dataset: {_master_dataset_abs_for_test}")

    example_image_paths = []
    example_xml_paths = []

    if not os.path.isdir(_master_dataset_abs_for_test):
        print(f"ОШИБКА: Директория мастер-датасета не найдена: {_master_dataset_abs_for_test}")
    else:
        for category_subdir_name in source_subfolder_keys_for_test:
            if not category_subdir_name: continue

            current_images_dir_for_test = os.path.join(_master_dataset_abs_for_test, category_subdir_name,
                                                       images_subdir_name_in_master)
            current_annotations_dir_for_test = os.path.join(_master_dataset_abs_for_test, category_subdir_name,
                                                            annotations_subdir_name_in_master)

            if not os.path.isdir(current_images_dir_for_test) or not os.path.isdir(current_annotations_dir_for_test):
                continue

            valid_extensions = ['.jpg', '.jpeg', '.png']
            image_files_in_category = []
            for ext in valid_extensions:
                image_files_in_category.extend(glob.glob(os.path.join(current_images_dir_for_test, f"*{ext.lower()}")))
                image_files_in_category.extend(glob.glob(os.path.join(current_images_dir_for_test, f"*{ext.upper()}")))

            image_files_in_category = sorted(list(set(image_files_in_category)))

            for img_path_abs_str in image_files_in_category:
                base_name, _ = os.path.splitext(os.path.basename(img_path_abs_str))
                xml_file_abs_str = os.path.join(current_annotations_dir_for_test, base_name + ".xml")

                if os.path.exists(xml_file_abs_str):
                    example_image_paths.append(img_path_abs_str)
                    example_xml_paths.append(xml_file_abs_str)

        if not example_image_paths:
            print("\nНе найдено совпадающих пар изображение/аннотация в Master_Dataset.")
        else:
            print(f"\nВсего найдено {len(example_image_paths)} пар изображение/аннотация в Master_Dataset.")

            num_test_files_to_load = min(len(example_image_paths), 3)

            if num_test_files_to_load == 0:
                print("Нет файлов для создания тестового датасета.")
            else:
                import random

                # random.seed(42) # Закомментируй для реальной случайности, или установи для воспроизводимости

                paired_files = list(zip(example_image_paths, example_xml_paths))
                random.shuffle(paired_files)
                selected_pairs = paired_files[:num_test_files_to_load]

                test_image_paths = [pair[0] for pair in selected_pairs]
                test_xml_paths = [pair[1] for pair in selected_pairs]

                print(
                    f"Будет протестировано и визуализировано на {len(test_image_paths)} СЛУЧАЙНЫХ файлах из Master_Dataset:")
                for p_idx, p_path in enumerate(test_image_paths):
                    print(f"  {p_idx + 1}. {os.path.basename(p_path)}")

                current_test_batch_size = 1

                dataset = create_detector_tf_dataset(
                    test_image_paths,
                    test_xml_paths,
                    batch_size=current_test_batch_size,
                    shuffle=False
                )

                print("\nОбработка и визуализация примеров из датасета детектора:")
                try:
                    for i, (images_batch, y_true_batch) in enumerate(dataset.take(len(test_image_paths))):
                        # КОПИРУЙ ОСТАТОК БЛОКА if __name__ == '__main__': ИЗ ПРЕДЫДУЩЕГО ОТВЕТА СЮДА
                        # НАЧИНАЯ С print(f"\n--- Батч {i + 1} ...") И ДО КОНЦА БЛОКА
                        print(f"\n--- Батч {i + 1} (детектор с y_true для сетки/якорей) ---")
                        print("Форма батча изображений:", images_batch.shape)
                        print("Форма батча y_true:", y_true_batch.shape)

                        if images_batch.shape[0] > 0:
                            for k_img_in_batch in range(images_batch.shape[0]):
                                current_image_path_index_in_selection = i * current_test_batch_size + k_img_in_batch
                                current_img_filename_for_print = "Unknown"
                                if current_image_path_index_in_selection < len(test_image_paths):
                                    current_img_filename_for_print = os.path.basename(
                                        test_image_paths[current_image_path_index_in_selection])

                                print(
                                    f"  --- Изображение {k_img_in_batch + 1} в батче ({current_img_filename_for_print}) ---")
                                single_image_np = images_batch[k_img_in_batch].numpy()
                                single_y_true_np = y_true_batch[k_img_in_batch].numpy()

                                objectness_scores_map = single_y_true_np[..., 4]
                                responsible_anchors_indices = tf.where(objectness_scores_map > 0.5)

                                if tf.size(responsible_anchors_indices) > 0:
                                    print(
                                        f"    Найдено {tf.size(responsible_anchors_indices) // 3} 'ответственных' ячеек/якорей:")
                                    for cell_coords_tf in responsible_anchors_indices:
                                        gy, gx, ga = cell_coords_tf.numpy()
                                        anchor_data = single_y_true_np[gy, gx, ga]
                                        class_one_hot = anchor_data[5:]
                                        class_id_detected = np.argmax(class_one_hot)
                                        class_name_detected = CLASSES_LIST_GLOBAL_FOR_DETECTOR[
                                            class_id_detected] if class_id_detected < len(
                                            CLASSES_LIST_GLOBAL_FOR_DETECTOR) and class_one_hot[
                                                                      class_id_detected] > 0.5 else "None"
                                        print(
                                            f"      Ячейка({gy},{gx}), Якорь_idx {ga}: Obj={anchor_data[4]:.1f}, Box(tx,ty,tw,th)={np.round(anchor_data[0:4], 2)}, ClassID={class_id_detected} ({class_name_detected})")
                                else:
                                    print(
                                        f"    Не найдено объектов (все objectness в y_true <= 0.5). Нормально для негативных примеров.")

                                if VISUALIZATION_ENABLED:
                                    print(f"    Вызов visualize_data_sample для {current_img_filename_for_print}...")
                                    visualize_data_sample(single_image_np, single_y_true_np,
                                                          title=f"Ground Truth for {current_img_filename_for_print}")
                                else:
                                    print("    Визуализация отключена (plot_utils не импортирован).")
                        else:
                            print("  Батч изображений пуст.")
                except Exception as e_dataset:
                    print(f"ОШИБКА при итерации по датасету детектора: {e_dataset}")
                    import traceback

                    traceback.print_exc()

    print("\n--- Тестирование detector_data_loader.py (с новым y_true и визуализацией) завершено ---")