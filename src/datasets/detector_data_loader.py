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


def parse_xml_annotation(xml_file_path):
    # КОД parse_xml_annotation ИЗ ПРЕДЫДУЩЕГО ОТВЕТА (полная версия)
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_filename_node = root.find('filename')
        image_filename = image_filename_node.text if image_filename_node is not None else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")
        size_node = root.find('size')
        img_width_xml, img_height_xml = None, None
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None and height_node is not None and width_node.text is not None and height_node.text is not None:
                try:
                    img_width_xml = int(width_node.text)
                    img_height_xml = int(height_node.text)
                    if img_width_xml <= 0 or img_height_xml <= 0:
                        img_width_xml, img_height_xml = None, None
                except ValueError:
                    img_width_xml, img_height_xml = None, None
        objects = []
        for obj_node in root.findall('object'):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None: continue
            class_name = class_name_node.text
            if class_name not in CLASSES_LIST_GLOBAL_FOR_DETECTOR: continue
            class_id = CLASSES_LIST_GLOBAL_FOR_DETECTOR.index(class_name)
            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None: continue
            try:
                xmin = float(bndbox_node.find('xmin').text)
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text)
                ymax = float(bndbox_node.find('ymax').text)
            except (ValueError, AttributeError, TypeError):
                continue
            if xmin >= xmax or ymin >= ymax: continue
            if img_width_xml and img_height_xml:  # Клиппинг по размерам из XML
                xmin = max(0, min(xmin, img_width_xml))
                ymin = max(0, min(ymin, img_height_xml))
                xmax = max(0, min(xmax, img_width_xml))
                ymax = max(0, min(ymax, img_height_xml))
                if xmin >= xmax or ymin >= ymax: continue
            objects.append({"class_id": class_id, "class_name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax,
                            "ymax": ymax})
        return objects, img_width_xml, img_height_xml, image_filename
    except ET.ParseError as e_parse:
        print(f"ERROR_PARSE: Ошибка парсинга XML {os.path.basename(xml_file_path)}: {e_parse}")
    except Exception as e:
        print(f"ERROR_PARSE: Непредвиденная ошибка при парсинге {os.path.basename(xml_file_path)}: {e}")
    return None, None, None, None


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
                                             py_anchors_wh, py_num_classes):
    # КОД load_and_prepare_detector_y_true_py_func ИЗ ПРЕДЫДУЩЕГО ОТВЕТА
    image_path = image_path_tensor.numpy().decode('utf-8')
    xml_path = xml_path_tensor.numpy().decode('utf-8')
    y_true_output_shape = (py_grid_h, py_grid_w, py_anchors_wh.shape[0], 5 + py_num_classes)
    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path).convert('RGB')
        image_np_original = np.array(pil_image, dtype=np.float32)
    except Exception:
        return np.zeros((py_target_height, py_target_width, 3), dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

    objects, _, _, _ = parse_xml_annotation(xml_path)
    if objects is None:
        return np.zeros((py_target_height, py_target_width, 3), dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

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

    y_true_target_np = np.zeros(y_true_output_shape, dtype=np.float32)
    if objects:
        gt_boxes_xywh_norm_list = []
        for i in range(scaled_gt_boxes_norm_tensor.shape[0]):
            box_norm = scaled_gt_boxes_norm_tensor[i].numpy()
            width_n = box_norm[2] - box_norm[0]
            height_n = box_norm[3] - box_norm[1]
            x_center_n = box_norm[0] + width_n / 2
            y_center_n = box_norm[1] + height_n / 2
            gt_boxes_xywh_norm_list.append([x_center_n, y_center_n, width_n, height_n])
        gt_boxes_xywh_norm_np = np.array(gt_boxes_xywh_norm_list, dtype=np.float32)
        anchor_assigned_mask = np.zeros((py_grid_h, py_grid_w, py_anchors_wh.shape[0]), dtype=np.bool_)

        for i in range(gt_boxes_xywh_norm_np.shape[0]):
            gt_box_xywh = gt_boxes_xywh_norm_np[i]
            gt_class_id = class_ids_list_for_gt[i]
            grid_x_float = gt_box_xywh[0] * py_grid_w
            grid_y_float = gt_box_xywh[1] * py_grid_h
            grid_x_idx = int(grid_x_float)
            grid_y_idx = int(grid_y_float)
            grid_x_idx = min(grid_x_idx, py_grid_w - 1)
            grid_y_idx = min(grid_y_idx, py_grid_h - 1)
            best_iou = -1
            best_anchor_idx = -1
            gt_box_shape_wh = [gt_box_xywh[2], gt_box_xywh[3]]
            ious = calculate_iou_numpy(gt_box_shape_wh, py_anchors_wh)
            best_anchor_idx = np.argmax(ious)
            best_iou = ious[best_anchor_idx]
            if best_iou > 0 and not anchor_assigned_mask[grid_y_idx, grid_x_idx, best_anchor_idx]:
                anchor_assigned_mask[grid_y_idx, grid_x_idx, best_anchor_idx] = True
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 4] = 1.0
                tx = float(grid_x_float) - float(grid_x_idx) # Явное приведение обоих к float
                ty = float(grid_y_float) - float(grid_y_idx) # Явное приведение обоих к float
                anchor_w = py_anchors_wh[best_anchor_idx][0]
                anchor_h = py_anchors_wh[best_anchor_idx][1]
                tw = np.log(gt_box_xywh[2] / (anchor_w + 1e-9) + 1e-9)
                th = np.log(gt_box_xywh[3] / (anchor_h + 1e-9) + 1e-9)
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 0:4] = [tx, ty, tw, th]
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 5 + gt_class_id] = 1.0
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


# --- Пример использования (для тестирования этого файла) ---
if __name__ == '__main__':
    # --- КОПИЯ БЛОКА if __name__ == '__main__' ИЗ МОЕГО ПРЕДЫДУЩЕГО ПОЛНОГО ОТВЕТА ---
    # --- (с исправленной логикой загрузки конфигов и формирования путей) ---
    if not CONFIG_LOAD_SUCCESS:
        print("\n!!! ВНИМАНИЕ: Конфигурационные файлы не были загружены корректно...")
    print(f"--- Тестирование detector_data_loader.py (с y_true для сетки и якорей) ---")
    print(f"Параметры сетки: GRID_HEIGHT={GRID_HEIGHT}, GRID_WIDTH={GRID_WIDTH}")
    print(f"Якоря (W_norm, H_norm):\n{ANCHORS_WH_NORMALIZED}")
    print(f"Количество якорей на ячейку: {NUM_ANCHORS_PER_LOCATION}")
    print(f"Количество классов: {NUM_CLASSES_DETECTOR}")

    def_road_subdir = BASE_CONFIG.get('source_defective_road_img_parent_subdir', "Defective_Road_Images")
    norm_road_subdir = BASE_CONFIG.get('source_normal_road_img_parent_subdir', "Normal_Road_Images")
    not_road_subdir = BASE_CONFIG.get('source_not_road_img_parent_subdir', "Not_Road_Images")  # Для кота

    images_dir_defective = os.path.join(MASTER_DATASET_PATH_ABS, def_road_subdir, _images_subdir_name_cfg)
    annotations_dir_defective = os.path.join(MASTER_DATASET_PATH_ABS, def_road_subdir, _annotations_subdir_name_cfg)
    images_dir_normal = os.path.join(MASTER_DATASET_PATH_ABS, norm_road_subdir, _images_subdir_name_cfg)
    annotations_dir_normal = os.path.join(MASTER_DATASET_PATH_ABS, norm_road_subdir, _annotations_subdir_name_cfg)
    images_dir_notroad = os.path.join(MASTER_DATASET_PATH_ABS, not_road_subdir, _images_subdir_name_cfg)
    annotations_dir_notroad = os.path.join(MASTER_DATASET_PATH_ABS, not_road_subdir, _annotations_subdir_name_cfg)

    print(f"\nТестовые пути для детектора (проверь их!):")
    print(f"  Изображения с дефектами из: {images_dir_defective}")
    print(f"  Нормальные изображения из: {images_dir_normal}")
    print(f"  'Не дорога' изображения из: {images_dir_notroad}")

    example_image_paths = []
    example_xml_paths = []

    # Имена тестовых файлов (ЗАМЕНИ НА СВОИ РЕАЛЬНЫЕ ИМЕНА)
    test_files_map = {
        "defect_1": {"img_dir": images_dir_defective, "ann_dir": annotations_dir_defective,
                     "base_name": "defective_road_01"},  # Пример твоего файла с ямой и трещиной
        "normal_1": {"img_dir": images_dir_normal, "ann_dir": annotations_dir_normal, "base_name": "normal_road_01"},
        # Пример нормальной дороги
        "notroad_1": {"img_dir": images_dir_notroad, "ann_dir": annotations_dir_notroad, "base_name": "not_road_cat_01"}
        # Пример кота
    }

    for key, paths_info in test_files_map.items():
        found_img_for_base = False
        img_path_abs_candidate = None
        for ext in ['.jpg', '.jpeg', '.png']:  # Проверяем разные расширения
            img_path_try = os.path.join(paths_info["img_dir"], paths_info["base_name"] + ext)
            if os.path.exists(img_path_try):
                img_path_abs_candidate = img_path_try
                found_img_for_base = True
                break
            # Можно добавить проверку для .JPG, .PNG и т.д. если нужно

        if found_img_for_base:
            xml_file_abs = os.path.join(paths_info["ann_dir"], paths_info["base_name"] + ".xml")
            if os.path.exists(xml_file_abs):
                example_image_paths.append(img_path_abs_candidate)
                example_xml_paths.append(xml_file_abs)
                print(f"  Добавлен для теста ({key}): {os.path.basename(img_path_abs_candidate)}")
            else:
                print(
                    f"  ПРЕДУПРЕЖДЕНИЕ: XML {xml_file_abs} не найден для изображения {os.path.basename(img_path_abs_candidate) if img_path_abs_candidate else paths_info['base_name']}")
        else:
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Изображение {paths_info['base_name']} с расширениями .jpg/.jpeg/.png не найдено в {paths_info['img_dir']}")

    if not example_image_paths:
        print("\nОШИБКА: Не найдено ни одного ИЗ ЗАДАННЫХ тестовых файлов для `detector_data_loader.py`.")
    else:
        effective_batch_size = min(BATCH_SIZE_FROM_CONFIG, len(example_image_paths))  # Используем BATCH_SIZE из конфига
        if effective_batch_size == 0 and len(example_image_paths) > 0: effective_batch_size = 1

        if effective_batch_size > 0:
            print(f"\nСоздание датасета с batch_size = {effective_batch_size}...")
            dataset = create_detector_tf_dataset(
                example_image_paths, example_xml_paths, effective_batch_size, shuffle=False
            )
            print("\nПример батча из датасета детектора (новый y_true):")
            try:
                for i, (images_batch, y_true_batch) in enumerate(dataset.take(1)):
                    print(f"\n--- Батч {i + 1} (детектор с новым y_true) ---")
                    print("Форма батча изображений:", images_batch.shape)
                    print("Форма батча y_true:", y_true_batch.shape)
                    if y_true_batch.shape[0] > 0:
                        for k_img_in_batch in range(y_true_batch.shape[0]):  # Итерируемся по картинкам в батче
                            print(
                                f"  --- Изображение {k_img_in_batch + 1} в батче ({os.path.basename(example_image_paths[k_img_in_batch])}) ---")
                            y_true_single_image = y_true_batch[k_img_in_batch]
                            objectness_scores = y_true_single_image[..., 4]
                            responsible_cells = tf.where(objectness_scores > 0.5)
                            if tf.size(responsible_cells) > 0:
                                print(f"    Найдены 'ответственные' ячейки/якоря (grid_y, grid_x, anchor_idx):")
                                for cell_coords in responsible_cells.numpy():
                                    gy, gx, ga = cell_coords
                                    print(f"      Cell ({gy},{gx}), Anchor {ga}:")
                                    print(f"        Objectness: {y_true_single_image[gy, gx, ga, 4].numpy():.1f}")
                                    print(f"        Box (tx,ty,tw,th): {y_true_single_image[gy, gx, ga, 0:4].numpy()}")
                                    print(f"        Classes (one-hot): {y_true_single_image[gy, gx, ga, 5:].numpy()}")
                            else:
                                print("    Не найдено объектов (все objectness < 0.5).")
            except Exception as e_dataset:
                print(f"ОШИБКА при итерации по датасету детектора: {e_dataset}")
                import traceback

                traceback.print_exc()
        else:
            print("Недостаточно файлов для создания тестового батча (0 файлов найдено или batch_size=0).")

    print("\n--- Тестирование detector_data_loader.py (с новым y_true) завершено ---")