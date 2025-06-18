# src/datasets/detector_data_loader.py
import sys
import os

# --- ЯВНОЕ ДОБАВЛЕНИЕ КОРНЯ ПРОЕКТА И ПАПКИ SRC В sys.path ---
_current_file_abs_path = os.path.abspath(__file__)
_datasets_dir_abs_path = os.path.dirname(_current_file_abs_path)
_src_dir_abs_path = os.path.dirname(_datasets_dir_abs_path)
_project_root_abs_path = os.path.dirname(_src_dir_abs_path)
if _src_dir_abs_path not in sys.path: sys.path.insert(0, str(_src_dir_abs_path))
if _project_root_abs_path not in sys.path: sys.path.insert(0, str(_project_root_abs_path))
# ----------------------------------------------------------------

import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
from functools import partial
import random
import argparse


# --- Импорт Аугментаций ---
def get_detector_train_augmentations_stub(h, w):
    # print("ЗАГЛУШКА: get_detector_train_augmentations вызвана (detector_data_loader).")
    return None


AUGMENTATION_FUNC_AVAILABLE = False
get_detector_train_augmentations = get_detector_train_augmentations_stub
try:
    from datasets.augmentations import get_detector_train_augmentations as aug_func_imported

    AUGMENTATION_FUNC_AVAILABLE = True
    get_detector_train_augmentations = aug_func_imported
    # print("INFO (detector_data_loader.py): Модуль augmentations.py успешно импортирован.")
except ImportError:
    # print("ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): datasets.augmentations не найден. Аугментация будет ЗАГЛУШКОЙ.")
    pass
except Exception as e_aug_imp:
    # print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Ошибка импорта augmentations: {e_aug_imp}. Аугментация будет ЗАГЛУШКОЙ.")
    pass


# --- Импорт Визуализации ---
def visualize_fpn_gt_assignments_stub(*args, **kwargs):
    print(
        "ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Используется ЗАГЛУШКА для visualize_fpn_gt_assignments. Реальная визуализация не будет вызвана.")


_plot_utils_imported_successfully = False
visualize_fpn_gt_assignments = visualize_fpn_gt_assignments_stub
try:
    from utils.plot_utils import visualize_fpn_gt_assignments as viz_func_imported

    _plot_utils_imported_successfully = True
    visualize_fpn_gt_assignments = viz_func_imported
    # print("INFO (detector_data_loader.py): Модуль utils.plot_utils.py успешно импортирован.")
except ImportError:
    # print("ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): utils.plot_utils не найден. Визуализация будет ЗАГЛУШКОЙ.")
    pass
except Exception as e_plot_imp:
    # print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Ошибка импорта plot_utils: {e_plot_imp}. Визуализация будет ЗАГЛУШКОЙ.")
    pass

# --- Загрузка Конфигурации ---
_base_config_path = os.path.join(_src_dir_abs_path, 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_src_dir_abs_path, 'configs', 'detector_config.yaml')
BASE_CONFIG = {};
DETECTOR_CONFIG = {};
CONFIG_LOAD_SUCCESS = True
try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS = False
except Exception:
    CONFIG_LOAD_SUCCESS = False
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS = False
except Exception:
    CONFIG_LOAD_SUCCESS = False

if not CONFIG_LOAD_SUCCESS:
    if __name__ == '__main__' or "pytest" in sys.modules:  # Выводим только при прямом запуске или тестах
        print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки конфигов в detector_data_loader. Используются дефолты.")
    # Установка дефолтов, если конфиги не загрузились
    FPN_PARAMS_CFG_DEFAULT = {
        'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
        'detector_fpn_levels': ['P3', 'P4', 'P5'],
        'detector_fpn_strides': {'P3': 8, 'P4': 16, 'P5': 32},
        'detector_fpn_anchor_configs': {
            'P3': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.05, 0.05]] * 3},
            'P4': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3},
            'P5': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.2, 0.2]] * 3}
        },
        'fpn_gt_assignment_scale_ranges': {'P3': [0, 64], 'P4': [64, 128], 'P5': [128, 10000]}
    }
    DETECTOR_CONFIG.setdefault('fpn_detector_params', FPN_PARAMS_CFG_DEFAULT)
    DETECTOR_CONFIG.setdefault('use_augmentation', False)
    BASE_CONFIG.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    BASE_CONFIG.setdefault('master_dataset_path', 'data/Master_Dataset_Fallback_DataLoader')

FPN_PARAMS_CFG = DETECTOR_CONFIG.get('fpn_detector_params', {})
_input_shape_list = FPN_PARAMS_CFG.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT = _input_shape_list[0];
TARGET_IMG_WIDTH = _input_shape_list[1]
CLASSES_LIST_GLOBAL_FOR_DETECTOR = FPN_PARAMS_CFG.get('classes', ['pit', 'crack'])
NUM_CLASSES_DETECTOR = len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
USE_AUGMENTATION_CFG = DETECTOR_CONFIG.get('use_augmentation', True) and AUGMENTATION_FUNC_AVAILABLE

FPN_LEVEL_NAMES_ORDERED = FPN_PARAMS_CFG.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
FPN_STRIDES_CONFIG_YAML = FPN_PARAMS_CFG.get('detector_fpn_strides', {})
FPN_ANCHOR_CONFIGS_YAML = FPN_PARAMS_CFG.get('detector_fpn_anchor_configs', {})
FPN_SCALE_RANGES_CFG_FROM_YAML = FPN_PARAMS_CFG.get('fpn_gt_assignment_scale_ranges',
                                                    {'P3': [0, 64], 'P4': [64, 128], 'P5': [128, 10000]})

FPN_LEVELS_CONFIG_GLOBAL = {}  # Этот словарь будет использоваться везде
for level_name in FPN_LEVEL_NAMES_ORDERED:
    level_anchor_data = FPN_ANCHOR_CONFIGS_YAML.get(level_name, {})
    level_stride_data = FPN_STRIDES_CONFIG_YAML.get(level_name)

    default_stride_val = {'P3': 8, 'P4': 16, 'P5': 32}.get(level_name, 16)
    num_anchors_val_cfg = level_anchor_data.get('num_anchors_this_level', 3)
    anchors_list_val_cfg = level_anchor_data.get('anchors_wh_normalized', [[0.1, 0.1]] * num_anchors_val_cfg)

    if level_stride_data is None: level_stride_data = default_stride_val

    current_anchors_np_init = np.array(anchors_list_val_cfg, dtype=np.float32)
    if not (current_anchors_np_init.ndim == 2 and current_anchors_np_init.shape[1] == 2 and
            current_anchors_np_init.shape[0] > 0):
        # print(f"WARN (Global Config Init): Некорректные якоря для {level_name} из YAML. Используем дефолтные 3 якоря [0.1,0.1].")
        current_anchors_np_init = np.array([[0.1, 0.1]] * 3, dtype=np.float32)

    num_anchors_final_init = current_anchors_np_init.shape[0]
    if num_anchors_val_cfg != num_anchors_final_init:
        # print(f"WARN (Global Config Init): num_anchors_this_level ({num_anchors_val_cfg}) для '{level_name}' "
        #       f"не соотв. длине anchors_wh_normalized ({num_anchors_final_init}). Используется {num_anchors_final_init}.")
        pass  # num_anchors_final_init уже правильное

    FPN_LEVELS_CONFIG_GLOBAL[level_name] = {
        'stride': level_stride_data,
        'anchors_wh_normalized': current_anchors_np_init,
        'num_anchors': num_anchors_final_init,
        'grid_h': TARGET_IMG_HEIGHT // level_stride_data,
        'grid_w': TARGET_IMG_WIDTH // level_stride_data
    }

_images_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')


# --- Конец Загрузки Конфигурации ---


# --- Вспомогательные Функции ---
def parse_xml_annotation(xml_file_path, classes_list_arg=CLASSES_LIST_GLOBAL_FOR_DETECTOR):
    # ... (КОД parse_xml_annotation КАК В ТВОЕМ ПОСЛЕДНЕМ РАБОЧЕМ ВАРИАНТЕ load_detector1.txt)
    try:
        tree = ET.parse(xml_file_path);
        root = tree.getroot();
        image_filename_node = root.find(
            'filename');
        image_filename = image_filename_node.text if image_filename_node is not None and image_filename_node.text else os.path.basename(
            xml_file_path).replace(".xml", ".jpg");
        size_node = root.find(
            'size');
        img_width_xml, img_height_xml = None, None
    except Exception:
        return None, None, None, None
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
        class_name_node = obj_node.find('name');
        if class_name_node is None or class_name_node.text is None: continue
        class_name = class_name_node.text
        if class_name not in classes_list_arg: continue
        class_id = classes_list_arg.index(class_name);
        bndbox_node = obj_node.find('bndbox')
        if bndbox_node is None: continue
        try:
            xmin, ymin, xmax, ymax = (float(bndbox_node.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax'])
        except (TypeError, ValueError, AttributeError):
            continue
        if xmin >= xmax or ymin >= ymax: continue
        if img_width_xml and img_height_xml:
            xmin = max(0.0, min(xmin, float(img_width_xml)));
            ymin = max(0.0, min(ymin, float(img_height_xml)))
            xmax = max(0.0, min(xmax, float(img_width_xml)));
            ymax = max(0.0, min(ymax, float(img_height_xml)))
            if xmin >= xmax or ymin >= ymax: continue
        objects.append(
            {"class_id": class_id, "class_name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return objects, img_width_xml, img_height_xml, image_filename


@tf.function
def preprocess_image_and_boxes(image, boxes, target_height_tf, target_width_tf):
    # ... (КОД preprocess_image_and_boxes КАК В ТВОЕМ ПОСЛЕДНЕМ РАБОЧЕМ ВАРИАНТЕ)
    original_height_f = tf.cast(tf.shape(image)[0], dtype=tf.float32);
    original_width_f = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    image_resized = tf.image.resize(image, [target_height_tf, target_width_tf]);
    image_processed = image_resized / 255.0
    num_boxes = tf.shape(boxes)[0]
    if num_boxes > 0:
        safe_original_width_f = tf.maximum(original_width_f, 1e-6);
        safe_original_height_f = tf.maximum(original_height_f, 1e-6)
        scaled_boxes_norm = tf.stack([boxes[:, 0] / safe_original_width_f, boxes[:, 1] / safe_original_height_f,
                                      boxes[:, 2] / safe_original_width_f, boxes[:, 3] / safe_original_height_f],
                                     axis=-1)
        scaled_boxes_norm = tf.clip_by_value(scaled_boxes_norm, 0.0, 1.0)
    else:
        scaled_boxes_norm = tf.zeros((0, 4), dtype=tf.float32)
    return image_processed, scaled_boxes_norm


def calculate_iou_numpy(box_wh, anchors_wh_level):
    # ... (КОД calculate_iou_numpy КАК В ТВОЕМ ПОСЛЕДНЕМ РАБОЧЕМ ВАРИАНТЕ)
    box_wh_np = np.array(box_wh);
    anchors_wh_level_np = np.array(anchors_wh_level)
    inter_w = np.minimum(box_wh_np[0], anchors_wh_level_np[:, 0]);
    inter_h = np.minimum(box_wh_np[1], anchors_wh_level_np[:, 1])
    intersection = inter_w * inter_h;
    box_area = box_wh_np[0] * box_wh_np[1];
    anchor_area = anchors_wh_level_np[:, 0] * anchors_wh_level_np[:, 1]
    union = box_area + anchor_area - intersection;
    return intersection / (union + 1e-6)


def assign_gt_to_fpn_levels_and_encode_by_scale(
        gt_boxes_pixels_list,  # Список списков: [[xmin_px, ymin_px, xmax_px, ymax_px], ...] ОРИГИНАЛЬНОГО изображения
        gt_class_ids_list,  # Список ID классов для каждого GT бокса
        original_image_width_px,  # Ширина ОРИГИНАЛЬНОГО изображения в пикселях
        original_image_height_px,  # Высота ОРИГИНАЛЬНОГО изображения в пикселях
        image_filename_for_debug="Unknown"  # Добавим для отладки
):
    """
    Назначает Ground Truth объекты на соответствующие уровни FPN на основе их ПИКСЕЛЬНОГО МАСШТАБА
    и кодирует y_true для каждого уровня.
    Использует глобальные FPN_LEVELS_CONFIG_GLOBAL, FPN_SCALE_RANGES_CFG_FROM_YAML,
    NUM_CLASSES_DETECTOR, FPN_LEVEL_NAMES_ORDERED.
    """
    y_true_all_levels = {}
    anchor_assigned_masks = {}  # Для отслеживания, чтобы один якорь не был назначен нескольким GT объектам

    # Инициализация y_true и масок для всех уровней FPN
    for level_name_init in FPN_LEVEL_NAMES_ORDERED:
        cfg_init = FPN_LEVELS_CONFIG_GLOBAL[level_name_init]
        y_true_all_levels[level_name_init] = np.zeros(
            (cfg_init['grid_h'], cfg_init['grid_w'], cfg_init['num_anchors'], 5 + NUM_CLASSES_DETECTOR),
            dtype=np.float32
        )
        anchor_assigned_masks[level_name_init] = np.zeros(
            (cfg_init['grid_h'], cfg_init['grid_w'], cfg_init['num_anchors']),
            dtype=bool
        )

    if not gt_boxes_pixels_list or original_image_width_px <= 0 or original_image_height_px <= 0:
        return tuple(y_true_all_levels[level_name_ret] for level_name_ret in FPN_LEVEL_NAMES_ORDERED)

    debug_img_name = image_filename_for_debug if image_filename_for_debug else "Unknown (assign_gt)"
    # --- Отладочный вывод: Начало обработки изображения ---
    #print(f"\nDEBUG assign_gt: Изображение '{debug_img_name}', Оригинальные размеры WxH: {original_image_width_px}x{original_image_height_px}")
    #print(f"  Диапазоны масштабов для FPN: {FPN_SCALE_RANGES_CFG_FROM_YAML}")
    # --- Конец отладочного вывода ---

    for i_obj in range(len(gt_boxes_pixels_list)):
        box_px = gt_boxes_pixels_list[i_obj]
        gt_class_id = int(gt_class_ids_list[i_obj])

        xmin_px, ymin_px, xmax_px, ymax_px = box_px
        gt_w_px = xmax_px - xmin_px
        gt_h_px = ymax_px - ymin_px

        if gt_w_px <= 0 or gt_h_px <= 0:
            # print(f"  DEBUG assign_gt: Пропущен GT объект {i_obj} с невалидными размерами: w={gt_w_px}, h={gt_h_px}")
            continue

        # 1. Вычисляем масштаб объекта (в пикселях оригинального изображения)
        object_scale_px = np.sqrt(gt_w_px * gt_h_px)
        assigned_level_name = None

        # 2. Определяем, какому уровню FPN назначить этот GT объект
        # Идем от P3 (мелкие) к P5 (крупные) и выбираем первый подходящий уровень
        for level_name in FPN_LEVEL_NAMES_ORDERED:  # ['P3', 'P4', 'P5']
            min_scale, max_scale = FPN_SCALE_RANGES_CFG_FROM_YAML[level_name]
            if min_scale <= object_scale_px < max_scale:
                assigned_level_name = level_name
                break

        if assigned_level_name is None:
            # Если не попал ни в один из основных диапазонов, проверяем, не слишком ли он большой для P5
            # (или слишком маленький для P3, если бы у P3 была нижняя граница > 0)
            if object_scale_px >= FPN_SCALE_RANGES_CFG_FROM_YAML.get('P5', [0, 0])[
                0]:  # Если больше или равен мин. для P5
                assigned_level_name = 'P5'
            # else: # Если слишком маленький даже для P3 - можно проигнорировать или назначить P3
            #     # print(f"  DEBUG assign_gt: GT объект {i_obj} (масштаб {object_scale_px:.1f}px) слишком мал. Назначаем P3.")
            #     # assigned_level_name = 'P3' # Или continue, если хотим игнорировать очень мелкие
            #     pass # Пока оставим None, если не попал и не больше P5_min

        if assigned_level_name is None:
            # print(f"  DEBUG assign_gt: GT объект {i_obj} (масштаб {object_scale_px:.1f}px) НЕ НАЗНАЧЕН ни одному уровню. Пропускается.")
            continue

        # --- Отладочный вывод для назначенного объекта ---
        #print(f"  GT Объект {i_obj}: Класс ID={gt_class_id} ({CLASSES_LIST_GLOBAL_FOR_DETECTOR[gt_class_id] if 0<=gt_class_id<len(CLASSES_LIST_GLOBAL_FOR_DETECTOR) else 'Unknown'}), "
             # f"Пикс. WxH: {gt_w_px:.1f}x{gt_h_px:.1f}, Масштаб_px: {object_scale_px:.1f} -> НАЗНАЧЕН УРОВНЮ: {assigned_level_name}")
        # --- Конец отладочного вывода ---

        # 3. Нормализуем координаты GT бокса относительно ВСЕГО изображения
        gt_xc_norm = (xmin_px + gt_w_px / 2.0) / original_image_width_px
        gt_yc_norm = (ymin_px + gt_h_px / 2.0) / original_image_height_px
        gt_w_norm = gt_w_px / original_image_width_px
        gt_h_norm = gt_h_px / original_image_height_px

        config_assigned = FPN_LEVELS_CONFIG_GLOBAL[assigned_level_name]
        y_true_target_level_np = y_true_all_levels[assigned_level_name]
        anchor_mask_target_level = anchor_assigned_masks[assigned_level_name]

        grid_h_level, grid_w_level = config_assigned['grid_h'], config_assigned['grid_w']
        anchors_wh_norm_level = config_assigned['anchors_wh_normalized']  # (num_anchors_level, 2)

        # Определяем ячейку сетки
        grid_x_center_float = gt_xc_norm * float(grid_w_level)
        grid_y_center_float = gt_yc_norm * float(grid_h_level)
        grid_x_idx = min(int(grid_x_center_float), grid_w_level - 1)
        grid_y_idx = min(int(grid_y_center_float), grid_h_level - 1)

        if anchors_wh_norm_level.shape[0] == 0:
            # print(f"  DEBUG assign_gt: Нет якорей для уровня {assigned_level_name} для объекта {i_obj}. Пропускаем сопоставление.")
            continue

        # 4. Сопоставление с якорями этого уровня
        gt_box_shape_wh_for_iou = [gt_w_norm, gt_h_norm]
        ious = calculate_iou_numpy(gt_box_shape_wh_for_iou, anchors_wh_norm_level)
        best_anchor_idx = int(np.argmax(ious))
        best_iou_val = float(ious[best_anchor_idx])  # Сохраним значение IoU для отладки

        # Назначаем GT этому якорю, если он еще не занят другим GT
        # (простая стратегия: один GT объект назначается только одному лучшему незанятому якорю в ячейке)
        if not anchor_mask_target_level[grid_y_idx, grid_x_idx, best_anchor_idx]:
            anchor_mask_target_level[grid_y_idx, grid_x_idx, best_anchor_idx] = True

            # 5. Кодируем координаты и класс для y_true
            y_true_target_level_np[grid_y_idx, grid_x_idx, best_anchor_idx, 4] = 1.0  # objectness = 1

            tx = grid_x_center_float - float(grid_x_idx)
            ty = grid_y_center_float - float(grid_y_idx)

            anchor_w_n_val = float(anchors_wh_norm_level[best_anchor_idx, 0])
            anchor_h_n_val = float(anchors_wh_norm_level[best_anchor_idx, 1])

            safe_gt_w_n = max(gt_w_norm, 1e-9);
            safe_gt_h_n = max(gt_h_norm, 1e-9)
            safe_anchor_w = max(anchor_w_n_val, 1e-9);
            safe_anchor_h = max(anchor_h_n_val, 1e-9)

            tw = np.log(safe_gt_w_n / safe_anchor_w);
            th = np.log(safe_gt_h_n / safe_anchor_h)

            y_true_target_level_np[grid_y_idx, grid_x_idx, best_anchor_idx, 0:4] = [tx, ty, tw, th]
            y_true_target_level_np[grid_y_idx, grid_x_idx, best_anchor_idx, 5 + gt_class_id] = 1.0

            # --- Отладочный вывод для назначенного якоря ---
            #print(f"    -> Ячейка ({grid_y_idx},{grid_x_idx}), Якорь_idx {best_anchor_idx} (IoU: {best_iou_val:.3f})")
            #print(f"       Закодировано: Objectness=1, Class={gt_class_id}, Box(tx,ty,tw,th)=[{tx:.3f},{ty:.3f},{tw:.3f},{th:.3f}]")
            # --- Конец отладочного вывода ---
        #else:
            #print(f"  DEBUG assign_gt: Якорь ({grid_y_idx},{grid_x_idx}, idx={best_anchor_idx}) на уровне {assigned_level_name} уже занят. GT объект {i_obj} не назначен этому якорю.")

    return tuple(y_true_all_levels[level_name_ret] for level_name_ret in FPN_LEVEL_NAMES_ORDERED)


# -------------------------------------------------------------------------------------

# --- Ключевые функции для загрузки (с вызовом assign_gt_to_fpn_levels_and_encode_by_scale) ---
def load_and_prepare_detector_fpn_py_func(image_path_tensor, xml_path_tensor, py_apply_augmentation_arg):

    image_path_str = image_path_tensor.numpy().decode('utf-8');
    xml_path_str = xml_path_tensor.numpy().decode('utf-8')
    apply_augmentation = bool(py_apply_augmentation_arg.numpy()) if hasattr(py_apply_augmentation_arg,
                                                                            'numpy') else bool(
        py_apply_augmentation_arg)
    y_true_fallbacks_list = [];
    img_fallback_shape = (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3)
    for level_name_fb in FPN_LEVEL_NAMES_ORDERED:
        cfg_fallback = FPN_LEVELS_CONFIG_GLOBAL.get(level_name_fb);
        if cfg_fallback:
            shape_fb = (cfg_fallback['grid_h'], cfg_fallback['grid_w'], cfg_fallback['num_anchors'],
                        5 + NUM_CLASSES_DETECTOR); y_true_fallbacks_list.append(np.zeros(shape_fb, dtype=np.float32))
        else:
            y_true_fallbacks_list.append(np.zeros((1, 1, 1, 5 + NUM_CLASSES_DETECTOR), dtype=np.float32))
    dummy_image_np = np.zeros(img_fallback_shape, dtype=np.float32);
    dummy_y_true_tuple = tuple(y_true_fallbacks_list)
    dummy_scaled_boxes_viz_np = np.zeros((0, 4), dtype=np.float32);
    dummy_class_ids_viz_np = np.zeros((0), dtype=np.int32)
    error_return_tuple = tuple(
        [dummy_image_np] + list(dummy_y_true_tuple) + [dummy_scaled_boxes_viz_np, dummy_class_ids_viz_np])
    try:
        from PIL import Image as PILImage;
        pil_image = PILImage.open(image_path_str).convert('RGB');
        image_original_np_uint8 = np.array(pil_image, dtype=np.uint8);
        original_img_width_px, original_img_height_px = pil_image.size
        if original_img_width_px <= 0 or original_img_height_px <= 0: return error_return_tuple
    except Exception:
        return error_return_tuple
    objects_from_xml, xml_img_w, xml_img_h, _ = parse_xml_annotation(
        xml_path_str);  # Использует глобальный CLASSES_LIST_GLOBAL_FOR_DETECTOR
    if objects_from_xml is None: return error_return_tuple
    current_processing_width = xml_img_w if xml_img_w and xml_img_w > 0 else original_img_width_px
    current_processing_height = xml_img_h if xml_img_h and xml_img_h > 0 else original_img_height_px
    boxes_pixels_before_aug = [[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']] for obj in
                               objects_from_xml] if objects_from_xml else []
    class_ids_before_aug = [obj['class_id'] for obj in objects_from_xml] if objects_from_xml else []
    image_to_process_np_uint8 = image_original_np_uint8.copy();
    boxes_after_potential_aug_pixels = list(boxes_pixels_before_aug);
    class_ids_after_potential_aug = list(class_ids_before_aug)
    img_w_for_assign_gt = current_processing_width;
    img_h_for_assign_gt = current_processing_height
    if apply_augmentation and AUGMENTATION_FUNC_AVAILABLE and objects_from_xml:
        try:
            augmenter = get_detector_train_augmentations(original_img_height_px, original_img_width_px)
            if augmenter:
                augmented_data = augmenter(image=image_original_np_uint8, bboxes=boxes_pixels_before_aug,
                                           class_labels_for_albumentations=class_ids_before_aug)
                image_to_process_np_uint8 = augmented_data['image'];
                boxes_after_potential_aug_pixels = augmented_data['bboxes'];
                class_ids_after_potential_aug = augmented_data['class_labels_for_albumentations']
                img_w_for_assign_gt = image_to_process_np_uint8.shape[1];
                img_h_for_assign_gt = image_to_process_np_uint8.shape[0]
                if not boxes_after_potential_aug_pixels: class_ids_after_potential_aug = []  # Синхронизируем
        except Exception:
            pass
    image_tensor_for_tf = tf.convert_to_tensor(image_to_process_np_uint8.astype(np.float32), dtype=tf.float32)
    temp_boxes_tensor_pixels_val_for_viz = tf.constant(boxes_after_potential_aug_pixels,
                                                       dtype=tf.float32) if boxes_after_potential_aug_pixels else tf.zeros(
        (0, 4), dtype=tf.float32)
    image_processed_for_model_tf, scaled_gt_boxes_norm_for_viz_tf = preprocess_image_and_boxes(image_tensor_for_tf,
                                                                                               temp_boxes_tensor_pixels_val_for_viz,
                                                                                               tf.constant(
                                                                                                   TARGET_IMG_HEIGHT,
                                                                                                   dtype=tf.int32),
                                                                                               tf.constant(
                                                                                                   TARGET_IMG_WIDTH,
                                                                                                   dtype=tf.int32))
    if len(boxes_after_potential_aug_pixels) != len(
        class_ids_after_potential_aug) and boxes_after_potential_aug_pixels: return error_return_tuple
    y_true_fpn_tuple_np = assign_gt_to_fpn_levels_and_encode_by_scale(boxes_after_potential_aug_pixels,
                                                                      class_ids_after_potential_aug,
                                                                      img_w_for_assign_gt, img_h_for_assign_gt)
    final_class_ids_for_viz_np = np.array(class_ids_after_potential_aug,
                                          dtype=np.int32) if class_ids_after_potential_aug else np.zeros((0),
                                                                                                         dtype=np.int32)
    return tuple(
        [image_processed_for_model_tf.numpy()] + list(y_true_fpn_tuple_np) + [scaled_gt_boxes_norm_for_viz_tf.numpy(),
                                                                              final_class_ids_for_viz_np])


# ------------------------------------------------------------------------------------------

def load_and_prepare_detector_fpn_tf_wrapper(image_path_tensor, xml_path_tensor, augment_tensor):
    # ... (Эта функция остается такой же, как в твоем load_detector1.txt или моем последнем ответе)
    tout_y_true_individual_specs = []
    for level_name_tfw in FPN_LEVEL_NAMES_ORDERED:
        level_cfg_tfw = FPN_LEVELS_CONFIG_GLOBAL[level_name_tfw]
        tout_y_true_individual_specs.append(tf.TensorSpec(shape=(
        level_cfg_tfw['grid_h'], level_cfg_tfw['grid_w'], level_cfg_tfw['num_anchors'], 5 + NUM_CLASSES_DETECTOR),
                                                          dtype=tf.float32))
    flat_tout_list = [tf.float32] + tout_y_true_individual_specs + [tf.float32,
                                                                    tf.int32]  # img, p3, p4, p5, boxes_viz, classes_viz
    flat_outputs_tf = tf.py_function(func=load_and_prepare_detector_fpn_py_func,
                                     inp=[image_path_tensor, xml_path_tensor, augment_tensor], Tout=flat_tout_list)
    img_processed_tf = flat_outputs_tf[0];
    img_processed_tf.set_shape([TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3])
    y_true_fpn_levels_tf_list = []
    num_fpn_outputs_from_py_func = len(FPN_LEVEL_NAMES_ORDERED)  # Должно быть 3
    for i in range(num_fpn_outputs_from_py_func):
        y_true_level_tensor = flat_outputs_tf[1 + i]  # Индексы 1, 2, 3 для y_true P3, P4, P5
        level_name_set_shape = FPN_LEVEL_NAMES_ORDERED[i];
        level_cfg_set_shape = FPN_LEVELS_CONFIG_GLOBAL[level_name_set_shape]
        current_shape = (
        level_cfg_set_shape['grid_h'], level_cfg_set_shape['grid_w'], level_cfg_set_shape['num_anchors'],
        5 + NUM_CLASSES_DETECTOR)
        y_true_level_tensor.set_shape(current_shape);
        y_true_fpn_levels_tf_list.append(y_true_level_tensor)
    return img_processed_tf, tuple(y_true_fpn_levels_tf_list)  # Возвращаем (img, (y_true_p3, y_true_p4, y_true_p5))


def create_detector_tf_dataset(image_paths_list, xml_paths_list, batch_size, shuffle=True, augment=False):
    # ... (Эта функция остается такой же, как в твоем load_detector1.txt) ...
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(xml_paths_list,
                                                                         (list, tuple)): raise ValueError(
        "paths must be lists/tuples.")
    if len(image_paths_list) != len(xml_paths_list): raise ValueError("Image and XML path counts differ.")
    if not image_paths_list: return None  # Если список пуст, возвращаем None
    augment_flags = tf.constant([augment] * len(image_paths_list), dtype=tf.bool)
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(image_paths_list, dtype=tf.string), tf.constant(xml_paths_list, dtype=tf.string), augment_flags))
    if shuffle: dataset = dataset.shuffle(buffer_size=max(1, len(image_paths_list)), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda img_p, xml_p, aug_f: load_and_prepare_detector_fpn_tf_wrapper(img_p, xml_p, aug_f),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# В самом конце файла src/datasets/detector_data_loader.py

if __name__ == '__main__':
    import sys
    import random
    import argparse

    # matplotlib импортируется внутри visualize_fpn_gt_assignments, если VISUALIZATION_ENABLED_FOR_MAIN_TEST is True

    parser = argparse.ArgumentParser(description="Тестирование и визуализация detector_data_loader.py для FPN.")
    parser.add_argument("--num_examples", type=int, default=3, help="Количество случайных примеров для показа.")
    parser.add_argument("--enable_visualization", action='store_true', help="Включить matplotlib визуализацию.")
    parser.add_argument("--seed", type=int, default=None, help="Seed для ГСЧ (для воспроизводимости).")
    args = parser.parse_args()

    VISUALIZATION_ENABLED_FOR_MAIN_TEST = False
    if args.enable_visualization:
        if _plot_utils_imported_successfully:
            VISUALIZATION_ENABLED_FOR_MAIN_TEST = True
            print("INFO (__main__): Визуализация ВКЛЮЧЕНА.")
        else:
            print("ПРЕДУПРЕЖДЕНИЕ (__main__): Визуализация запрошена, но plot_utils не импортирован. ОТКЛЮЧЕНА.")

    if not CONFIG_LOAD_SUCCESS:
        print("\n!!! ВНИМАНИЕ: Ошибка загрузки конфигов в detector_data_loader.py.")

    print(f"\n--- Тестирование detector_data_loader.py (FPN с назначением по МАСШТАБУ) ---")
    print(f"  Количество примеров для обработки: {args.num_examples}")
    print(f"  Визуализация во время этого теста: {VISUALIZATION_ENABLED_FOR_MAIN_TEST}")
    print(f"  Используется SEED: {args.seed if args.seed is not None else 'Случайный (None)'}")
    # ... (остальной вывод информации о конфигурации FPN) ...

    # --- ЯВНЫЙ СБОР ФАЙЛОВ ---
    detector_dataset_ready_path = os.path.join(_project_root_abs_path, "data", "Detector_Dataset_Ready")
    train_images_dir = os.path.join(detector_dataset_ready_path, "train", _images_subdir_name_cfg)
    train_annotations_dir = os.path.join(detector_dataset_ready_path, "train", _annotations_subdir_name_cfg)

    print(f"\nСканирование обучающих данных:")
    print(f"  Папка изображений: {train_images_dir}")
    print(f"  Папка аннотаций: {train_annotations_dir}")

    all_found_image_paths = []
    if os.path.isdir(train_images_dir):
        for ext_pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_found_image_paths.extend(glob.glob(os.path.join(train_images_dir, ext_pattern)))
    else:
        print(f"ОШИБКА: Директория изображений не найдена: {train_images_dir}")

    if not all_found_image_paths:
        print("Изображения в обучающей директории не найдены.")
    else:
        print(f"Найдено всего {len(all_found_image_paths)} файлов изображений в {train_images_dir}.")

        valid_pairs = []
        for img_path_str in all_found_image_paths:
            base_name, _ = os.path.splitext(os.path.basename(img_path_str))
            xml_path_str = os.path.join(train_annotations_dir, base_name + ".xml")
            if os.path.exists(xml_path_str):
                valid_pairs.append((img_path_str, xml_path_str))
            # else:
            # print(f"  Предупреждение: XML для {os.path.basename(img_path_str)} не найден.")

        print(f"Найдено {len(valid_pairs)} валидных пар (изображение + XML аннотация).")

        if not valid_pairs:
            print("Не найдено ни одной валидной пары для теста.")
        else:
            num_to_sample = min(args.num_examples, len(valid_pairs))

            if num_to_sample == 0:
                print("Не выбрано ни одного примера для теста (num_examples=0 или нет доступных валидных пар).")
            else:
                # Устанавливаем seed для воспроизводимости ВЫБОРКИ
                if args.seed is not None:
                    random.seed(args.seed)
                # Если args.seed is None, то random.sample будет использовать текущее состояние ГСЧ
                # (которое обычно инициализируется системным временем при первом вызове random).
                # Для полной уверенности в новой случайности при каждом запуске без --seed,
                # можно было бы добавить random.seed(None) здесь.

                selected_pairs_for_test = random.sample(valid_pairs, num_to_sample)

                print(
                    f"\nБудет обработано и (если включено) визуализировано {len(selected_pairs_for_test)} случайных файлов:")

                for i_example_test, (img_path_test, xml_path_test) in enumerate(selected_pairs_for_test):
                    print(
                        f"\n--- Пример {i_example_test + 1}/{len(selected_pairs_for_test)}: {os.path.basename(img_path_test)} ---")

                    # 1. Оригинал
                    print(f"  Обработка ОРИГИНАЛА...")
                    try:
                        outputs_orig_test = load_and_prepare_detector_fpn_py_func(
                            tf.constant(img_path_test, dtype=tf.string),
                            tf.constant(xml_path_test, dtype=tf.string),
                            tf.constant(False, dtype=tf.bool)  # augment = False
                        )
                        img_np_o, y_true_tup_o, boxes_viz_o, classes_viz_o = \
                            outputs_orig_test[0], tuple(outputs_orig_test[1:4]), outputs_orig_test[4], \
                            outputs_orig_test[5]

                        # ... (Вывод форм и информации, как раньше)
                        print(f"    Изображение: {img_np_o.shape}, GT объектов для виз: {boxes_viz_o.shape[0]}")

                        if VISUALIZATION_ENABLED_FOR_MAIN_TEST and _plot_utils_imported_successfully:
                            visualize_fpn_gt_assignments(
                                img_np_o, y_true_tup_o,
                                fpn_level_names=FPN_LEVEL_NAMES_ORDERED,
                                fpn_configs=FPN_LEVELS_CONFIG_GLOBAL,
                                classes_list=CLASSES_LIST_GLOBAL_FOR_DETECTOR,
                                title_prefix=f"GT (Scale) for {os.path.basename(img_path_test)} [ORIGINAL]"
                            )
                    except Exception as e_orig_test:
                        print(f"    ОШИБКА при обработке оригинала '{os.path.basename(img_path_test)}': {e_orig_test}")
                        import traceback;

                        traceback.print_exc()

                    # 2. Аугментированная версия
                    if USE_AUGMENTATION_CFG and AUGMENTATION_FUNC_AVAILABLE:
                        print(f"  Обработка АУГМЕНТИРОВАННОЙ...")
                        try:
                            outputs_aug_test = load_and_prepare_detector_fpn_py_func(
                                tf.constant(img_path_test, dtype=tf.string),
                                tf.constant(xml_path_test, dtype=tf.string),
                                tf.constant(True, dtype=tf.bool)  # augment = True
                            )
                            img_np_a, y_true_tup_a, boxes_viz_a, classes_viz_a = \
                                outputs_aug_test[0], tuple(outputs_aug_test[1:4]), outputs_aug_test[4], \
                                outputs_aug_test[5]

                            # ... (Вывод форм и информации)
                            print(
                                f"    Изображение (аугм): {img_np_a.shape}, GT объектов для виз (аугм): {boxes_viz_a.shape[0]}")

                            if VISUALIZATION_ENABLED_FOR_MAIN_TEST and _plot_utils_imported_successfully:
                                visualize_fpn_gt_assignments(
                                    img_np_a, y_true_tup_a,
                                    fpn_level_names=FPN_LEVEL_NAMES_ORDERED,
                                    fpn_configs=FPN_LEVELS_CONFIG_GLOBAL,
                                    classes_list=CLASSES_LIST_GLOBAL_FOR_DETECTOR,
                                    title_prefix=f"GT (Scale) for {os.path.basename(img_path_test)} [AUGMENTED]"
                                )
                        except Exception as e_aug_test:
                            print(
                                f"    ОШИБКА при обработке аугментированной '{os.path.basename(img_path_test)}': {e_aug_test}")
                            import traceback;

                            traceback.print_exc()
    print("\n--- Тестирование detector_data_loader.py завершено ---")