# RoadDefectDetector/src/datasets/detector_data_loader.py
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
import sys
from functools import partial  # Может понадобиться, если будем передавать больше аргументов в map

# --- Импорт Аугментаций ---
try:
    if __name__ == '__main__' and __package__ is None:  # Если запускаем файл напрямую
        # Добавляем родительскую директорию src/ в путь, чтобы можно было импортировать src.datasets.augmentations
        _parent_dir_for_direct_run = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if _parent_dir_for_direct_run not in sys.path:
            sys.path.insert(0, _parent_dir_for_direct_run)
        from datasets.augmentations import get_detector_train_augmentations
    else:  # Если импортируется как модуль
        from .augmentations import get_detector_train_augmentations

    AUGMENTATION_FUNC_AVAILABLE = True
    print("INFO (detector_data_loader.py): Модуль augmentations.py успешно импортирован.")
except ImportError as e_imp:
    print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Не удалось импортировать augmentations: {e_imp}")
    print("                                       Аугментация будет отключена.")
    AUGMENTATION_FUNC_AVAILABLE = False


    def get_detector_train_augmentations(h, w):
        return None  # Заглушка
except Exception as e_other_imp:
    print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Другая ошибка при импорте augmentations: {e_other_imp}")
    AUGMENTATION_FUNC_AVAILABLE = False


    def get_detector_train_augmentations(h, w):
        return None  # Заглушка

# --- Загрузка Конфигурации ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))

_base_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS = True

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS = False; print(f"ПРЕДУПРЕЖДЕНИЕ: Файл base_config.yaml не найден: {_base_config_path}.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS = False; print(f"ОШИБКА YAML в base_config.yaml.")

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS = False; print(f"ПРЕДУПРЕЖДЕНИЕ: Файл detector_config.yaml не найден: {_detector_config_path}.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS = False; print(f"ОШИБКА YAML в detector_config.yaml.")

if not CONFIG_LOAD_SUCCESS:
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки конфигов. Используются дефолты в detector_data_loader.")
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('classes', ['pit', 'crack'])  # Используй 'crack' если это твой класс
    DETECTOR_CONFIG.setdefault('num_anchors_per_location', 3)
    DETECTOR_CONFIG.setdefault('anchors_wh_normalized', [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]])
    DETECTOR_CONFIG.setdefault('use_augmentation', False)
    BASE_CONFIG.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    BASE_CONFIG.setdefault('master_dataset_path', 'data/Master_Dataset_Fallback')

_input_shape_list = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT = _input_shape_list[0]
TARGET_IMG_WIDTH = _input_shape_list[1]
CLASSES_LIST_GLOBAL_FOR_DETECTOR = DETECTOR_CONFIG.get('classes', ['pit', 'crack'])
NUM_CLASSES_DETECTOR = len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
# BATCH_SIZE_FROM_CONFIG = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 1) # Не используется в этом файле напрямую
NETWORK_STRIDE = 16
GRID_HEIGHT = TARGET_IMG_HEIGHT // NETWORK_STRIDE
GRID_WIDTH = TARGET_IMG_WIDTH // NETWORK_STRIDE
ANCHORS_WH_NORMALIZED_LIST = DETECTOR_CONFIG.get('anchors_wh_normalized', [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]])
ANCHORS_WH_NORMALIZED = np.array(ANCHORS_WH_NORMALIZED_LIST, dtype=np.float32)
NUM_ANCHORS_PER_LOCATION = ANCHORS_WH_NORMALIZED.shape[0]
USE_AUGMENTATION_CFG = DETECTOR_CONFIG.get('use_augmentation',
                                           False) and AUGMENTATION_FUNC_AVAILABLE  # Учитываем доступность функции
_master_dataset_path_from_cfg = BASE_CONFIG.get('master_dataset_path', 'data/Master_Dataset_Fallback')
if not os.path.isabs(_master_dataset_path_from_cfg):
    MASTER_DATASET_PATH_ABS = os.path.join(_project_root_dir, _master_dataset_path_from_cfg)
else:
    MASTER_DATASET_PATH_ABS = _master_dataset_path_from_cfg
_images_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')


# --- Конец Загрузки Конфигурации ---


# --- Функции parse_xml_annotation, preprocess_image_and_boxes, calculate_iou_numpy ---
# (Вставь сюда свои последние рабочие версии этих функций.
#  Я оставлю здесь те, что мы отладили)
def parse_xml_annotation(xml_file_path, classes_list):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        image_filename_node = root.find('filename')
        image_filename = image_filename_node.text if image_filename_node is not None and image_filename_node.text else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")
        size_node = root.find('size')
        img_width_xml, img_height_xml = None, None
        if size_node is not None:
            width_node, height_node = size_node.find('width'), size_node.find('height')
            if width_node is not None and height_node is not None and width_node.text and height_node.text:
                try:
                    img_width_xml, img_height_xml = int(width_node.text), int(height_node.text)
                    if img_width_xml <= 0 or img_height_xml <= 0: img_width_xml, img_height_xml = None, None
                except ValueError:
                    img_width_xml, img_height_xml = None, None
        objects = []
        for obj_node in root.findall('object'):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None: continue
            class_name = class_name_node.text
            if class_name not in classes_list: continue
            class_id = classes_list.index(class_name)
            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None: continue
            try:
                xmin, ymin, xmax, ymax = (float(bndbox_node.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax'])
            except (ValueError, AttributeError, TypeError):
                continue
            if xmin >= xmax or ymin >= ymax: continue
            if img_width_xml and img_height_xml:  # Клиппинг
                xmin, ymin, xmax, ymax = max(0, min(xmin, img_width_xml)), max(0, min(ymin, img_height_xml)), max(0,
                                                                                                                  min(xmax,
                                                                                                                      img_width_xml)), max(
                    0, min(ymax, img_height_xml))
                if xmin >= xmax or ymin >= ymax: continue  # Проверка после клиппинга
            objects.append({"class_id": class_id, "class_name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax,
                            "ymax": ymax})
        return objects, img_width_xml, img_height_xml, image_filename
    except ET.ParseError as e_parse:
        print(f"XML_PARSE_ERROR: {os.path.basename(xml_file_path)}: {e_parse}")
    except Exception as e_generic:
        print(f"XML_OTHER_ERROR: {os.path.basename(xml_file_path)}: {e_generic}")
    return None, None, None, None


@tf.function
def preprocess_image_and_boxes(image, boxes, target_height_tf, target_width_tf):
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
    box_wh = np.array(box_wh);
    anchors_wh = np.array(anchors_wh)
    inter_w = np.minimum(box_wh[0], anchors_wh[:, 0]);
    inter_h = np.minimum(box_wh[1], anchors_wh[:, 1])
    intersection = inter_w * inter_h
    box_area = box_wh[0] * box_wh[1];
    anchor_area = anchors_wh[:, 0] * anchors_wh[:, 1]
    union = box_area + anchor_area - intersection
    return intersection / (union + 1e-6)


# -------------------------------------------------------------------------------------

def load_and_prepare_detector_y_true_py_func(image_path_tensor, xml_path_tensor,
                                             py_target_height, py_target_width,
                                             py_grid_h, py_grid_w,
                                             py_anchors_wh, py_num_classes,
                                             py_apply_augmentation_arg):  # Изменено имя аргумента
    image_path = image_path_tensor.numpy().decode('utf-8')
    xml_path = xml_path_tensor.numpy().decode('utf-8')


    if hasattr(py_apply_augmentation_arg, 'numpy'):  # Если это EagerTensor
        py_apply_augmentation = bool(py_apply_augmentation_arg.numpy())
    else:  # Если это уже Python bool или numpy.bool_
        py_apply_augmentation = bool(py_apply_augmentation_arg)

    y_true_output_shape = (py_grid_h, py_grid_w, py_anchors_wh.shape[0], 5 + py_num_classes)
    image_output_shape = (py_target_height, py_target_width, 3)


    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path).convert('RGB')
        image_np_original_uint8 = np.array(pil_image, dtype=np.uint8)
    except Exception as e_img_load:
        return np.zeros(image_output_shape, dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

    objects, _, _, _ = parse_xml_annotation(xml_path, CLASSES_LIST_GLOBAL_FOR_DETECTOR)

    if objects is None:
        return np.zeros(image_output_shape, dtype=np.float32), \
            np.zeros(y_true_output_shape, dtype=np.float32)

    boxes_list_pixels_orig = []
    class_ids_list_for_gt_orig = []
    if objects:
        for obj in objects:
            boxes_list_pixels_orig.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            class_ids_list_for_gt_orig.append(obj['class_id'])

    current_image_np = image_np_original_uint8
    current_boxes_pixels = boxes_list_pixels_orig
    current_class_ids = class_ids_list_for_gt_orig

    if py_apply_augmentation and AUGMENTATION_FUNC_AVAILABLE and objects:
        try:
            augmenter = get_detector_train_augmentations(py_target_height, py_target_width)
            augmented = augmenter(image=current_image_np,
                                  bboxes=current_boxes_pixels,
                                  class_labels_for_albumentations=current_class_ids)
            current_image_np = augmented['image']
            current_boxes_pixels = augmented['bboxes']
            current_class_ids = augmented['class_labels_for_albumentations']
            if not current_boxes_pixels: objects = []
        except Exception as e_aug:
            print(
                f"DEBUG_PY_FUNC WARNING: Ошибка аугментации для {image_path}: {e_aug}. Используется оригинальное изображение.")

    image_tensor_in_py = tf.convert_to_tensor(current_image_np.astype(np.float32), dtype=tf.float32)
    boxes_tensor_pixels_in_py = tf.constant(current_boxes_pixels,
                                            dtype=tf.float32) if current_boxes_pixels else tf.zeros((0, 4),
                                                                                                    dtype=tf.float32)

    image_processed_tensor, scaled_gt_boxes_norm_tensor = preprocess_image_and_boxes(
        image_tensor_in_py, boxes_tensor_pixels_in_py,
        tf.constant(py_target_height, dtype=tf.int32),
        tf.constant(py_target_width, dtype=tf.int32)
    )

    y_true_target_np = np.zeros(y_true_output_shape, dtype=np.float32)

    if objects and tf.shape(scaled_gt_boxes_norm_tensor)[
        0] > 0:  # Используем objects, а не current_boxes_pixels, так как objects обнуляется если боксы удалены
        gt_boxes_xywh_norm_list = []
        # ... (остальная логика формирования gt_boxes_xywh_norm_list как была) ...
        for i_s in range(scaled_gt_boxes_norm_tensor.shape[0]):
            box_norm_np = scaled_gt_boxes_norm_tensor[i_s].numpy();
            width_n = float(box_norm_np[2]) - float(box_norm_np[0]);
            height_n = float(box_norm_np[3]) - float(box_norm_np[1]);
            x_center_n = float(box_norm_np[0]) + width_n / 2.0;
            y_center_n = float(box_norm_np[1]) + height_n / 2.0;
            gt_boxes_xywh_norm_list.append([x_center_n, y_center_n, width_n, height_n])

        gt_boxes_xywh_norm_np = np.array(gt_boxes_xywh_norm_list, dtype=np.float32)
        anchor_assigned_mask = np.zeros((py_grid_h, py_grid_w, py_anchors_wh.shape[0]), dtype=bool)

        for i_obj_loop in range(gt_boxes_xywh_norm_np.shape[0]):
            if i_obj_loop >= len(current_class_ids): continue
            gt_xc_n, gt_yc_n, gt_w_n, gt_h_n = gt_boxes_xywh_norm_np[i_obj_loop]
            gt_class_id = int(current_class_ids[i_obj_loop])
            # ... (остальная логика назначения якорей и заполнения y_true_target_np как была) ...
            grid_x_center_float = gt_xc_n * float(py_grid_w);
            grid_y_center_float = gt_yc_n * float(py_grid_h)
            grid_x_idx = int(grid_x_center_float);
            grid_y_idx = int(grid_y_center_float)
            grid_x_idx = min(grid_x_idx, py_grid_w - 1);
            grid_y_idx = min(grid_y_idx, py_grid_h - 1)
            best_iou = -1.0;
            best_anchor_idx = -1
            gt_box_shape_wh_for_iou = [gt_w_n, gt_h_n]
            ious = calculate_iou_numpy(gt_box_shape_wh_for_iou, py_anchors_wh)
            best_anchor_idx = int(np.argmax(ious));
            best_iou = float(ious[best_anchor_idx])
            if best_iou >= 0.0 and not anchor_assigned_mask[grid_y_idx, grid_x_idx, best_anchor_idx]:
                anchor_assigned_mask[grid_y_idx, grid_x_idx, best_anchor_idx] = True
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 4] = 1.0
                tx = grid_x_center_float - float(grid_x_idx);
                ty = grid_y_center_float - float(grid_y_idx)
                anchor_w_norm_val = float(py_anchors_wh[best_anchor_idx, 0]);
                anchor_h_norm_val = float(py_anchors_wh[best_anchor_idx, 1])
                safe_gt_w_n = max(gt_w_n, 1e-9);
                safe_gt_h_n = max(gt_h_n, 1e-9)
                safe_anchor_w = max(anchor_w_norm_val, 1e-9);
                safe_anchor_h = max(anchor_h_norm_val, 1e-9)
                tw = np.log(safe_gt_w_n / safe_anchor_w);
                th = np.log(safe_gt_h_n / safe_anchor_h)
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 0:4] = [tx, ty, tw, th]
                y_true_target_np[grid_y_idx, grid_x_idx, best_anchor_idx, 5 + gt_class_id] = 1.0

    return image_processed_tensor.numpy(), y_true_target_np


def load_and_prepare_detector_y_true_tf_wrapper(image_path_tensor, xml_path_tensor, augment_tensor):
    img_processed_np, y_true_np = tf.py_function(
        func=load_and_prepare_detector_y_true_py_func,
        inp=[image_path_tensor, xml_path_tensor,
             TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH,
             GRID_HEIGHT, GRID_WIDTH,
             ANCHORS_WH_NORMALIZED,
             NUM_CLASSES_DETECTOR,
             augment_tensor],  # <--- Передаем флаг аугментации
        Tout=[tf.float32, tf.float32]
    )

    img_processed_np.set_shape([TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3])
    y_true_output_shape_tf = (GRID_HEIGHT, GRID_WIDTH, NUM_ANCHORS_PER_LOCATION, 5 + NUM_CLASSES_DETECTOR)
    y_true_np.set_shape(y_true_output_shape_tf)

    return img_processed_np, y_true_np


def create_detector_tf_dataset(image_paths_list, xml_paths_list, batch_size,
                               shuffle=True, augment=False):
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(xml_paths_list, (list, tuple)):
        raise ValueError("image_paths_list и xml_paths_list должны быть Python списками или кортежами.")
    if len(image_paths_list) != len(xml_paths_list):
        raise ValueError("Количество путей к изображениям и XML должно совпадать.")

    augment_flags = tf.constant([augment] * len(image_paths_list), dtype=tf.bool)

    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(image_paths_list, dtype=tf.string),
        tf.constant(xml_paths_list, dtype=tf.string),
        augment_flags
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths_list), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda img_p, xml_p, aug_f: load_and_prepare_detector_y_true_tf_wrapper(img_p, xml_p, aug_f),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Блок if __name__ == '__main__': ---
# (Остается таким же, как в твоей последней рабочей версии.
#  Убедись, что он вызывает create_detector_tf_dataset с новым аргументом `augment`,
#  например, `augment=USE_AUGMENTATION_CFG` или `augment=True` для теста.)
# Я скопирую его еще раз для полноты:
if __name__ == '__main__':
    import sys
    import random  # Добавим импорт random

    _utils_path = os.path.abspath(os.path.join(_current_script_dir, '..', 'utils'))
    if _utils_path not in sys.path:
        sys.path.insert(0, _utils_path)
    try:
        from plot_utils import visualize_data_sample

        VISUALIZATION_ENABLED = True
    except ImportError:
        VISUALIZATION_ENABLED = False
        print("ПРЕДУПРЕЖДЕНИЕ: Модуль plot_utils не найден. Визуализация будет отключена.")
    except Exception as e_imp:
        VISUALIZATION_ENABLED = False
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при импорте plot_utils: {e_imp}. Визуализация будет отключена.")

    if not CONFIG_LOAD_SUCCESS:
        print("\n!!! ВНИМАНИЕ: Конфигурационные файлы не были загружены корректно.")

    print(f"--- Тестирование detector_data_loader.py (Оригинал + Аугментация из Detector_Dataset_Ready/train/) ---")
    print(f"  Глобальный флаг USE_AUGMENTATION_CFG из конфига: {USE_AUGMENTATION_CFG}")
    print(f"  AUGMENTATION_FUNC_AVAILABLE: {AUGMENTATION_FUNC_AVAILABLE}")
    # ... (остальной print параметров сетки, якорей, классов, они берутся из глобальных переменных модуля) ...
    print(f"Параметры сетки: GRID_HEIGHT={GRID_HEIGHT}, GRID_WIDTH={GRID_WIDTH}")
    print(f"Якоря (W_norm, H_norm) используются внутри py_func:\n{ANCHORS_WH_NORMALIZED}")
    print(f"Количество якорей на ячейку: {NUM_ANCHORS_PER_LOCATION}")
    print(f"Количество классов: {NUM_CLASSES_DETECTOR} ({CLASSES_LIST_GLOBAL_FOR_DETECTOR})")

    # --- Пути к разделенному датасету (train) ---
    _detector_dataset_ready_path_rel = "data/Detector_Dataset_Ready"
    DETECTOR_DATASET_READY_ABS = os.path.join(_project_root_dir, _detector_dataset_ready_path_rel)

    TRAIN_IMAGES_DIR_FOR_TEST = os.path.join(DETECTOR_DATASET_READY_ABS, "train",
                                             _images_subdir_name_cfg)  # _images_subdir_name_cfg из глобальных
    TRAIN_ANNOTATIONS_DIR_FOR_TEST = os.path.join(DETECTOR_DATASET_READY_ABS, "train",
                                                  _annotations_subdir_name_cfg)  # _annotations_subdir_name_cfg из глобальных

    print(f"\nТестовые пути (сканируем Detector_Dataset_Ready/train/):")
    print(f"  Изображения из: {TRAIN_IMAGES_DIR_FOR_TEST}")
    print(f"  Аннотации из: {TRAIN_ANNOTATIONS_DIR_FOR_TEST}")

    example_image_paths = []
    example_xml_paths = []

    if not os.path.isdir(TRAIN_IMAGES_DIR_FOR_TEST) or not os.path.isdir(TRAIN_ANNOTATIONS_DIR_FOR_TEST):
        print(f"ОШИБКА: Директории data/Detector_Dataset_Ready/train/ (JPEGImages или Annotations) не найдены.")
        print("Пожалуйста, убедитесь, что скрипт 'create_data_splits.py' был успешно запущен.")
    else:
        valid_extensions = ['.jpg', '.jpeg', '.png']
        all_image_files_in_train_dir = []
        for ext in valid_extensions:
            all_image_files_in_train_dir.extend(glob.glob(os.path.join(TRAIN_IMAGES_DIR_FOR_TEST, f"*{ext.lower()}")))
            all_image_files_in_train_dir.extend(glob.glob(os.path.join(TRAIN_IMAGES_DIR_FOR_TEST, f"*{ext.upper()}")))

        all_image_files_in_train_dir = sorted(list(set(all_image_files_in_train_dir)))

        for img_path_abs_str in all_image_files_in_train_dir:
            base_name, _ = os.path.splitext(os.path.basename(img_path_abs_str))
            xml_file_abs_str = os.path.join(TRAIN_ANNOTATIONS_DIR_FOR_TEST, base_name + ".xml")

            if os.path.exists(xml_file_abs_str):
                example_image_paths.append(img_path_abs_str)
                example_xml_paths.append(xml_file_abs_str)

        if not example_image_paths:
            print("\nНе найдено совпадающих пар изображение/аннотация в data/Detector_Dataset_Ready/train/.")
        else:
            print(
                f"\nВсего найдено {len(example_image_paths)} пар изображение/аннотация в data/Detector_Dataset_Ready/train/.")

            num_examples_to_visualize = min(len(example_image_paths), 5)  # Показываем 2 случайных примера или меньше

            if num_examples_to_visualize == 0:
                print("Нет файлов для теста.")
            else:
                paired_files = list(zip(example_image_paths, example_xml_paths))
                random.shuffle(paired_files)
                selected_pairs = paired_files[:num_examples_to_visualize]

                print(
                    f"Будет протестировано и визуализировано (оригинал + аугментация) на {len(selected_pairs)} случайных файлах из TRAIN выборки:")
                for p_idx, (p_img_path, p_xml_path) in enumerate(selected_pairs):
                    print(f"  {p_idx + 1}. {os.path.basename(p_img_path)}")

                    current_test_batch_size = 1

                    # --- 1. Обработка ОРИГИНАЛА ---
                    print(f"\n  --- Обработка ОРИГИНАЛА для: {os.path.basename(p_img_path)} ---")
                    dataset_no_aug = create_detector_tf_dataset(
                        [p_img_path], [p_xml_path],
                        batch_size=current_test_batch_size,
                        shuffle=False,
                        augment=False
                    )
                    try:
                        for images_batch, y_true_batch in dataset_no_aug.take(1):
                            # ... (код вывода форм и информации об ответственных якорях, как в твоей предыдущей версии)
                            print("    Img shape (оригинал):", images_batch.shape, "y_true shape (оригинал):",
                                  y_true_batch.shape)
                            if images_batch.shape[0] > 0:
                                s_img, s_ytrue = images_batch[0].numpy(), y_true_batch[0].numpy()
                                obj_map = s_ytrue[..., 4];
                                resp_idx = tf.where(obj_map > 0.5)
                                if tf.size(resp_idx) > 0:
                                    print(
                                        f"    {tf.size(resp_idx) // 3} 'ответственных' якорей (Оригинал):")  # Добавил Original
                                else:
                                    print("    Не найдено объектов в y_true (Оригинал).")
                                if VISUALIZATION_ENABLED:
                                    visualize_data_sample(s_img, s_ytrue,
                                                          title=f"GT for {os.path.basename(p_img_path)} (Original)")
                    except Exception as e:
                        print(f"    ОШИБКА при обработке оригинала: {e}")

                    # --- 2. Обработка С аугментацией (если включена глобально) ---
                    # Используем глобальный флаг USE_AUGMENTATION_CFG, который читается из detector_config.yaml
                    if USE_AUGMENTATION_CFG and AUGMENTATION_FUNC_AVAILABLE:
                        print(f"\n  --- Обработка АУГМЕНТИРОВАННОЙ версии для: {os.path.basename(p_img_path)} ---")
                        dataset_with_aug = create_detector_tf_dataset(
                            [p_img_path], [p_xml_path],
                            batch_size=current_test_batch_size,
                            shuffle=False,
                            augment=True  # ЯВНО включаем аугментацию для этой ветки
                        )
                        try:
                            for images_batch_aug, y_true_batch_aug in dataset_with_aug.take(1):
                                print("    Img shape (аугм.):", images_batch_aug.shape, "y_true shape (аугм.):",
                                      y_true_batch_aug.shape)
                                if images_batch_aug.shape[0] > 0:
                                    s_img_aug, s_ytrue_aug = images_batch_aug[0].numpy(), y_true_batch_aug[0].numpy()
                                    obj_map_aug = s_ytrue_aug[..., 4];
                                    resp_idx_aug = tf.where(obj_map_aug > 0.5)
                                    if tf.size(resp_idx_aug) > 0:
                                        print(f"    {tf.size(resp_idx_aug) // 3} 'ответственных' якорей (Аугм.):")
                                    else:
                                        print("    Не найдено объектов в y_true (Аугм.).")
                                    if VISUALIZATION_ENABLED:
                                        visualize_data_sample(s_img_aug, s_ytrue_aug,
                                                              title=f"GT for {os.path.basename(p_img_path)} (AUGMENTED)")
                        except Exception as e:
                            print(f"    ОШИБКА при обработке аугментированной версии: {e}")
                    elif not AUGMENTATION_FUNC_AVAILABLE:
                        print(
                            f"\n  --- Аугментация для {os.path.basename(p_img_path)} пропущена (AUGMENTATION_FUNC_AVAILABLE = False) ---")
                    else:  # USE_AUGMENTATION_CFG is False
                        print(
                            f"\n  --- Аугментация для {os.path.basename(p_img_path)} пропущена (USE_AUGMENTATION_CFG = False в конфиге) ---")

    print("\n--- Тестирование detector_data_loader.py завершено ---")