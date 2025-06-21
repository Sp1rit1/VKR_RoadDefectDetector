
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
from pathlib import Path
import random  # Добавлен для __main__, но может быть полезен и здесь
import xml.etree.ElementTree as ET
from PIL import Image as PILImage

# --- Настройка sys.path для импорта augmentations и plot_utils ---
_current_script_dir_sdl_v2 = Path(__file__).resolve().parent
_src_dir_sdl_v2 = _current_script_dir_sdl_v2.parent
_project_root_sdl_v2 = _src_dir_sdl_v2.parent
if str(_src_dir_sdl_v2) not in sys.path: sys.path.insert(0, str(_src_dir_sdl_v2))
if str(_project_root_sdl_v2) not in sys.path: sys.path.insert(0, str(_project_root_sdl_v2))

# --- Импорт Аугментаций ---
AUGMENTATION_FUNC_AVAILABLE_SDL_V2 = False
get_detector_train_augmentations_sdl_v2 = lambda h, w: None  # Заглушка по умолчанию
try:
    from datasets.augmentations import get_detector_train_augmentations as aug_func_imported_sdl_v2

    get_detector_train_augmentations_sdl_v2 = aug_func_imported_sdl_v2
    AUGMENTATION_FUNC_AVAILABLE_SDL_V2 = True
    # print("INFO (sdl_loader_v2): Модуль augmentations.py успешно импортирован.") # Раскомментируй для отладки импорта
except ImportError:
    # print("ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): augmentations.py не найден. Аугментация будет отключена.")
    pass
except Exception as e_aug_sdl_v2_imp:  # Более общее исключение
    print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): Ошибка импорта augmentations: {e_aug_sdl_v2_imp}. Аугментация отключена.")
    pass

# --- Импорт Визуализации (для использования в __main__) ---
# Глобальный флаг для проверки, удалось ли импортировать plot_utils
_plot_utils_v2_imported_successfully = False  # Инициализируем как False
# Заглушка для функции визуализации, если реальная не импортируется
visualize_single_level_gt_assignments_sdl_v2 = lambda *args, **kwargs: print(
    "ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): Используется ЗАГЛУШКА для visualize_single_level_gt_assignments. Реальная функция из plot_utils_v2 не импортирована.")
try:
    from utils.other_utils.plot_utils_v2 import visualize_single_level_gt_assignments as viz_func_v2_imported_real

    visualize_single_level_gt_assignments_sdl_v2 = viz_func_v2_imported_real
    _plot_utils_v2_imported_successfully = True  # Устанавливаем флаг успеха
except ImportError:
    pass
except Exception:
    pass

# --- Загрузка Конфигурации (предпочтение _v2.yaml) ---
_base_config_path_sdl_v2 = _src_dir_sdl_v2 / 'configs' / 'base_config.yaml'
_detector_config_path_sdl_v2_primary = _src_dir_sdl_v2 / 'configs' / 'detector_config_single_level_v2.yaml'
_detector_config_path_sdl_v2_fallback = _src_dir_sdl_v2 / 'configs' / 'detector_config_single_level_debug.yaml'

BASE_CONFIG_SDL_V2_GLOBAL = {}
DETECTOR_CONFIG_SDL_V2_GLOBAL = {}  # Будет содержать содержимое _v2.yaml или _debug.yaml
CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL = True

try:
    with open(_base_config_path_sdl_v2, 'r', encoding='utf-8') as f:
        BASE_CONFIG_SDL_V2_GLOBAL = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG_SDL_V2_GLOBAL, dict) or not BASE_CONFIG_SDL_V2_GLOBAL:
        CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL = False;
        BASE_CONFIG_SDL_V2_GLOBAL = {}
except Exception:
    CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL = False; BASE_CONFIG_SDL_V2_GLOBAL = {}

_detector_config_to_load = _detector_config_path_sdl_v2_primary
if not _detector_config_to_load.exists():
    # print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): Файл {_detector_config_to_load.name} не найден. Попытка загрузить {_detector_config_path_sdl_v2_fallback.name}")
    _detector_config_to_load = _detector_config_path_sdl_v2_fallback

try:
    with open(_detector_config_to_load, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_SDL_V2_GLOBAL = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_SDL_V2_GLOBAL, dict) or not DETECTOR_CONFIG_SDL_V2_GLOBAL:
        CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL = False;
        DETECTOR_CONFIG_SDL_V2_GLOBAL = {}
except Exception:
    CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL = False; DETECTOR_CONFIG_SDL_V2_GLOBAL = {}

if not CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL:
    print("ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): Используются АВАРИЙНЫЕ ДЕФОЛТЫ для detector_config.")
    BASE_CONFIG_SDL_V2_GLOBAL.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    BASE_CONFIG_SDL_V2_GLOBAL.setdefault('master_dataset_path', 'data/Master_Dataset_Fallback_SDL_V2')
    _fpn_params_default_sdl_v2 = {'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
                                  'detector_fpn_levels': ['P4_debug'], 'detector_fpn_strides': {'P4_debug': 16},
                                  'detector_fpn_anchor_configs': {'P4_debug': {'num_anchors_this_level': 7,
                                                                               'anchors_wh_normalized': [
                                                                                   [0.1832, 0.1104], [0.0967, 0.2753],
                                                                                   [0.4083, 0.0925], [0.0921, 0.4968],
                                                                                   [0.2919, 0.1936], [0.7358, 0.0843],
                                                                                   [0.0743, 0.8969]]}},
                                  'iou_positive_threshold': 0.7, 'iou_ignore_threshold': 0.4}  # Добавлены пороги IoU
    DETECTOR_CONFIG_SDL_V2_GLOBAL.setdefault('fpn_detector_params', _fpn_params_default_sdl_v2);
    DETECTOR_CONFIG_SDL_V2_GLOBAL.setdefault('use_augmentation', True)

# --- Глобальные Параметры из Конфига v2 ---
_fpn_params_sdl_v2 = DETECTOR_CONFIG_SDL_V2_GLOBAL.get('fpn_detector_params', {})
INPUT_SHAPE_SDL_V2_G = tuple(_fpn_params_sdl_v2.get('input_shape', [416, 416, 3]))
TARGET_IMG_HEIGHT_SDL_V2_G, TARGET_IMG_WIDTH_SDL_V2_G = INPUT_SHAPE_SDL_V2_G[0], INPUT_SHAPE_SDL_V2_G[1]
CLASSES_LIST_SDL_V2_G = _fpn_params_sdl_v2.get('classes', ['pit', 'crack'])
NUM_CLASSES_SDL_V2_G = len(CLASSES_LIST_SDL_V2_G)
USE_AUGMENTATION_SDL_V2_FLAG_G = DETECTOR_CONFIG_SDL_V2_GLOBAL.get('use_augmentation',
                                                                   True) and AUGMENTATION_FUNC_AVAILABLE_SDL_V2

FPN_LEVEL_NAME_DEBUG_SDL_V2_G = _fpn_params_sdl_v2.get('detector_fpn_levels', ['P4_debug'])[0]
_p4_debug_stride_cfg_v2 = _fpn_params_sdl_v2.get('detector_fpn_strides', {}).get(FPN_LEVEL_NAME_DEBUG_SDL_V2_G, 16)
_p4_debug_anchor_cfg_yaml_v2 = _fpn_params_sdl_v2.get('detector_fpn_anchor_configs', {}).get(
    FPN_LEVEL_NAME_DEBUG_SDL_V2_G, {})

# Дефолтные якоря, если в конфиге пусто или не тот формат
default_anchors_for_p4_debug = np.array([[0.15, 0.15]] * 7, dtype=np.float32)  # Пример дефолта на 7 якорей
ANCHORS_WH_P4_DEBUG_SDL_V2_G = np.array(
    _p4_debug_anchor_cfg_yaml_v2.get('anchors_wh_normalized', default_anchors_for_p4_debug), dtype=np.float32)
NUM_ANCHORS_P4_DEBUG_SDL_V2_G = _p4_debug_anchor_cfg_yaml_v2.get('num_anchors_this_level',
                                                                 ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[0])

if ANCHORS_WH_P4_DEBUG_SDL_V2_G.ndim == 1 and ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[
    0] == 2:  # Если прочитался как один якорь [w,h]
    ANCHORS_WH_P4_DEBUG_SDL_V2_G = np.expand_dims(ANCHORS_WH_P4_DEBUG_SDL_V2_G, axis=0)
if ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[
    0] != NUM_ANCHORS_P4_DEBUG_SDL_V2_G:  # Синхронизация, если num_anchors_this_level не совпадает с anchors_wh_normalized
    print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): num_anchors_this_level ({NUM_ANCHORS_P4_DEBUG_SDL_V2_G}) "
          f"не соответствует количеству пар в anchors_wh_normalized ({ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[0]}). "
          f"Используется количество из anchors_wh_normalized.")
    NUM_ANCHORS_P4_DEBUG_SDL_V2_G = ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[0]
if NUM_ANCHORS_P4_DEBUG_SDL_V2_G == 0 and ANCHORS_WH_P4_DEBUG_SDL_V2_G.shape[0] == 0:  # Если якорей вообще нет
    print(
        f"КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ (sdl_loader_v2): Якоря для уровня {FPN_LEVEL_NAME_DEBUG_SDL_V2_G} не определены! Добавлен один дефолтный якорь.")
    ANCHORS_WH_P4_DEBUG_SDL_V2_G = np.array([[0.1, 0.1]], dtype=np.float32)
    NUM_ANCHORS_P4_DEBUG_SDL_V2_G = 1

GRID_H_P4_DEBUG_SDL_V2_G = TARGET_IMG_HEIGHT_SDL_V2_G // _p4_debug_stride_cfg_v2
GRID_W_P4_DEBUG_SDL_V2_G = TARGET_IMG_WIDTH_SDL_V2_G // _p4_debug_stride_cfg_v2

IOU_POSITIVE_THRESHOLD_SDL_G = _fpn_params_sdl_v2.get('iou_positive_threshold', 0.7)
IOU_IGNORE_THRESHOLD_SDL_G = _fpn_params_sdl_v2.get('iou_ignore_threshold', 0.4)
ANCHOR_SHAPE_MATCHING_THRESHOLD_SDL_G = _fpn_params_sdl_v2.get('anchor_shape_matching_threshold', 4.0)

SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G = {
    'grid_h': GRID_H_P4_DEBUG_SDL_V2_G, 'grid_w': GRID_W_P4_DEBUG_SDL_V2_G,
    'anchors_wh_normalized': ANCHORS_WH_P4_DEBUG_SDL_V2_G,
    'num_anchors': NUM_ANCHORS_P4_DEBUG_SDL_V2_G,
    'stride': _p4_debug_stride_cfg_v2
}


# --- Вспомогательные функции ---
def parse_xml_annotation(xml_file_path, classes_list_arg=CLASSES_LIST_SDL_V2_G):
    if not xml_file_path or not os.path.exists(xml_file_path):  # Проверка на пустой путь или несуществующий файл
        return [], None, None, os.path.basename(xml_file_path or "unknown.xml").replace(".xml", ".jpg")
    try:
        tree = ET.parse(xml_file_path);
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
            class_name_node = obj_node.find('name');
            if class_name_node is None or class_name_node.text is None: continue
            class_name = class_name_node.text
            if class_name not in classes_list_arg: continue
            class_id = classes_list_arg.index(class_name);
            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None: continue
            try:
                xmin = float(bndbox_node.find('xmin').text);
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text);
                ymax = float(bndbox_node.find('ymax').text)
            except (TypeError, ValueError, AttributeError):
                continue
            if xmin >= xmax or ymin >= ymax: continue
            if img_width_xml and img_height_xml:  # Клиппинг
                xmin = max(0.0, min(xmin, float(img_width_xml)));
                ymin = max(0.0, min(ymin, float(img_height_xml)))
                xmax = max(0.0, min(xmax, float(img_width_xml)));
                ymax = max(0.0, min(ymax, float(img_height_xml)))
                if xmin >= xmax or ymin >= ymax: continue
            objects.append({"class_id": class_id, "class_name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax,
                            "ymax": ymax})
        return objects, img_width_xml, img_height_xml, image_filename
    except ET.ParseError:
        pass
    except Exception:
        pass
    return [], None, None, os.path.basename(xml_file_path or "unknown.xml").replace(".xml",
                                                                                    ".jpg")  # Возвращаем пустой список объектов при ошибке


@tf.function
def preprocess_image_and_boxes(image_tensor, boxes_tensor_pixels, target_height_tf, target_width_tf):
    original_height_f = tf.cast(tf.shape(image_tensor)[0], dtype=tf.float32)
    original_width_f = tf.cast(tf.shape(image_tensor)[1], dtype=tf.float32)
    image_resized_tf = tf.image.resize(image_tensor, [target_height_tf, target_width_tf])
    image_processed_tf = image_resized_tf / 255.0
    num_boxes = tf.shape(boxes_tensor_pixels)[0]
    if num_boxes > 0:
        safe_original_width_f = tf.maximum(original_width_f, 1e-6)
        safe_original_height_f = tf.maximum(original_height_f, 1e-6)
        scaled_boxes_normalized = tf.stack([
            boxes_tensor_pixels[:, 0] / safe_original_width_f, boxes_tensor_pixels[:, 1] / safe_original_height_f,
            boxes_tensor_pixels[:, 2] / safe_original_width_f, boxes_tensor_pixels[:, 3] / safe_original_height_f
        ], axis=-1)
        scaled_boxes_normalized = tf.clip_by_value(scaled_boxes_normalized, 0.0, 1.0)
    else:
        scaled_boxes_normalized = tf.zeros((0, 4), dtype=tf.float32)
    return image_processed_tf, scaled_boxes_normalized


def calculate_iou_xyxy_numpy(box1_xyxy, box2_xyxy):
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1_xyxy;
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2_xyxy
    inter_xmin = max(b1_xmin, b2_xmin);
    inter_ymin = max(b1_ymin, b2_ymin);
    inter_xmax = min(b1_xmax, b2_xmax);
    inter_ymax = min(b1_ymax, b2_ymax)
    inter_w = max(0, inter_xmax - inter_xmin);
    inter_h = max(0, inter_ymax - inter_ymin);
    intersection_area = inter_w * inter_h
    box1_area = (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin);
    box2_area = (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin)
    union_area = box1_area + box2_area - intersection_area;
    return intersection_area / (union_area + 1e-6)


# ==============================================================================
# НОВАЯ ВЕРСИЯ assign_gt_to_single_level_v2 (YOLOv5/v7-like assignment)
# ==============================================================================
def assign_gt_to_anchors_hybrid_strategy(
        gt_boxes_pixels_list_arg,
        gt_class_ids_list_arg,
        original_img_width_px_arg,
        original_img_height_px_arg,
        level_config_arg,
        iou_positive_threshold_arg,
        iou_ignore_threshold_arg,
        anchor_shape_match_thresh_arg,
        image_filename_for_debug="unknown_image"
):
    grid_h = level_config_arg['grid_h']
    grid_w = level_config_arg['grid_w']
    anchors_wh_level_config_np = np.array(level_config_arg['anchors_wh_normalized'], dtype=np.float32)
    num_anchors_at_this_level = level_config_arg['num_anchors']

    y_true_target_array = np.zeros((grid_h, grid_w, num_anchors_at_this_level, 5 + NUM_CLASSES_SDL_V2_G),
                                   dtype=np.float32)
    y_true_target_array[..., 4] = -1.0

    if not gt_boxes_pixels_list_arg or original_img_width_px_arg <= 0 or original_img_height_px_arg <= 0:
        y_true_target_array[..., 4] = 0.0
        return y_true_target_array

    # --- Шаг 1: Подготовка информации о Ground Truth объектах ---
    targets_info = []  # <--- ОПРЕДЕЛЕНИЕ targets_info
    gt_boxes_xyxy_norm_all_for_ignore_check = []  # Для Шага 4

    for i_obj_gt, box_px in enumerate(gt_boxes_pixels_list_arg):
        xmin_px, ymin_px, xmax_px, ymax_px = box_px
        gt_w_px = xmax_px - xmin_px;
        gt_h_px = ymax_px - ymin_px
        if gt_w_px <= 0 or gt_h_px <= 0: continue

        xc_n = (xmin_px + xmax_px) / 2.0 / original_img_width_px_arg
        yc_n = (ymin_px + ymax_px) / 2.0 / original_img_height_px_arg
        w_n = gt_w_px / original_img_width_px_arg;
        h_n = gt_h_px / original_img_height_px_arg

        targets_info.append({  # <--- ЗАПОЛНЕНИЕ targets_info
            'class_id': int(gt_class_ids_list_arg[i_obj_gt]),
            'xywh_norm': [xc_n, yc_n, w_n, h_n],
            'xyxy_norm': [xc_n - w_n / 2, yc_n - h_n / 2, xc_n + w_n / 2, yc_n + h_n / 2],
            'original_idx': i_obj_gt,
            'assigned_anchor_count': 0
        })
        gt_boxes_xyxy_norm_all_for_ignore_check.append([xc_n - w_n / 2, yc_n - h_n / 2, xc_n + w_n / 2, yc_n + h_n / 2])

    if not targets_info:
        y_true_target_array[..., 4] = 0.0;
        return y_true_target_array

    gt_boxes_xyxy_norm_all_np_for_ignore = np.array(gt_boxes_xyxy_norm_all_for_ignore_check, dtype=np.float32)

    anchor_assignment_map_final = np.full((grid_h, grid_w, num_anchors_at_this_level, 2), -1.0, dtype=np.float32)
    gt_wh_all_np = np.array([t['xywh_norm'][2:4] for t in targets_info])
    anchors_bcast_shape = anchors_wh_level_config_np[np.newaxis, :, :]
    gt_wh_bcast_shape = gt_wh_all_np[:, np.newaxis, :]
    safe_anchors_bcast_s = np.maximum(anchors_bcast_shape, 1e-9)
    safe_gt_wh_bcast_s = np.maximum(gt_wh_bcast_shape, 1e-9)
    ratios_s = np.maximum(safe_gt_wh_bcast_s / safe_anchors_bcast_s, safe_anchors_bcast_s / safe_gt_wh_bcast_s)
    shape_match_mask_gt_anchor = np.all(ratios_s < anchor_shape_match_thresh_arg, axis=2)

    # --- Шаг 2: Назначение "строго позитивных" якорей ---
    for gt_idx_loop, current_gt_info in enumerate(targets_info):  # Используем enumerate для индекса в targets_info
        gy = min(int(current_gt_info['xywh_norm'][1] * grid_h), grid_h - 1)
        gx = min(int(current_gt_info['xywh_norm'][0] * grid_w), grid_w - 1)

        for anchor_idx in range(num_anchors_at_this_level):
            if not shape_match_mask_gt_anchor[gt_idx_loop, anchor_idx]:  # Используем gt_idx_loop
                continue

            anchor_w_n, anchor_h_n = anchors_wh_level_config_np[anchor_idx]
            anchor_xc_n = (gx + 0.5) / grid_w;
            anchor_yc_n = (gy + 0.5) / grid_h
            anchor_xyxy_n_current = [anchor_xc_n - anchor_w_n / 2, anchor_yc_n - anchor_h_n / 2,
                                     anchor_xc_n + anchor_w_n / 2, anchor_yc_n + anchor_h_n / 2]

            iou = calculate_iou_xyxy_numpy(current_gt_info["xyxy_norm"], anchor_xyxy_n_current)

            if iou > iou_positive_threshold_arg:
                previous_iou_for_this_anchor = anchor_assignment_map_final[gy, gx, anchor_idx, 1]
                if iou > previous_iou_for_this_anchor:
                    anchor_assignment_map_final[gy, gx, anchor_idx, 0] = current_gt_info['original_idx']
                    anchor_assignment_map_final[gy, gx, anchor_idx, 1] = iou
                    # Обновляем счетчик назначений для оригинального GT объекта
                    # targets_info теперь список словарей, нужно найти правильный по original_idx
                    original_gt_idx_to_update = current_gt_info['original_idx']
                    for gt_to_update in targets_info:
                        if gt_to_update['original_idx'] == original_gt_idx_to_update:
                            gt_to_update[
                                'assigned_anchor_count'] += 1  # Это не совсем точно, если якорь переназначается, но для флага хватит
                            break

    # --- Шаг 3: Гарантированное назначение для "осиротевших" GT ---
    for gt_info_orphan in targets_info:  # Итерируемся по targets_info
        if gt_info_orphan['assigned_anchor_count'] > 0:
            continue

        gt_xyxy_n_orphan = gt_info_orphan['xyxy_norm']
        gt_wh_n_orphan = gt_info_orphan['xywh_norm'][2:4]
        gt_original_idx_orphan = gt_info_orphan['original_idx']
        best_iou_overall_for_orphan = -1.0
        best_gy_overall, best_gx_overall, best_anchor_idx_overall = -1, -1, -1

        for r_fb in range(grid_h):
            for c_fb in range(grid_w):
                for a_idx_fb in range(num_anchors_at_this_level):
                    anchor_w_n_fb, anchor_h_n_fb = anchors_wh_level_config_np[a_idx_fb]
                    # Проверка формы для fallback (опционально, но может быть полезно)
                    ratio_w_fb = gt_wh_n_orphan[0] / (anchor_w_n_fb + 1e-9);
                    ratio_h_fb = gt_wh_n_orphan[1] / (anchor_h_n_fb + 1e-9)
                    if not (max(ratio_w_fb, 1.0 / ratio_w_fb) < anchor_shape_match_thresh_arg and \
                            max(ratio_h_fb, 1.0 / ratio_h_fb) < anchor_shape_match_thresh_arg):
                        continue

                    anchor_xc_n_fb = (c_fb + 0.5) / grid_w;
                    anchor_yc_n_fb = (r_fb + 0.5) / grid_h
                    anchor_xyxy_n_fb = [anchor_xc_n_fb - anchor_w_n_fb / 2, anchor_yc_n_fb - anchor_h_n_fb / 2,
                                        anchor_xc_n_fb + anchor_w_n_fb / 2, anchor_yc_n_fb + anchor_h_n_fb / 2]
                    iou_fb = calculate_iou_xyxy_numpy(gt_xyxy_n_orphan, anchor_xyxy_n_fb)
                    if iou_fb > best_iou_overall_for_orphan:
                        best_iou_overall_for_orphan = iou_fb
                        best_gy_overall, best_gx_overall, best_anchor_idx_overall = r_fb, c_fb, a_idx_fb

        if best_iou_overall_for_orphan > 1e-3:
            previous_iou_for_this_best_anchor = anchor_assignment_map_final[
                best_gy_overall, best_gx_overall, best_anchor_idx_overall, 1]
            if best_iou_overall_for_orphan > previous_iou_for_this_best_anchor:
                anchor_assignment_map_final[
                    best_gy_overall, best_gx_overall, best_anchor_idx_overall, 0] = gt_original_idx_orphan
                anchor_assignment_map_final[
                    best_gy_overall, best_gx_overall, best_anchor_idx_overall, 1] = best_iou_overall_for_orphan

    # --- Шаг 4: Финальное формирование y_true и обработка ignore/background ---
    for r_final in range(grid_h):
        for c_final in range(grid_w):
            for a_idx_final in range(num_anchors_at_this_level):
                assigned_gt_original_idx_final = int(anchor_assignment_map_final[r_final, c_final, a_idx_final, 0])

                if assigned_gt_original_idx_final != -1:
                    y_true_target_array[r_final, c_final, a_idx_final, 4] = 1.0
                    # Используем targets_info, который был определен в Шаге 1
                    gt_data_for_encoding = next(
                        (t for t in targets_info if t['original_idx'] == assigned_gt_original_idx_final), None)
                    if gt_data_for_encoding is None:  # На всякий случай, хотя не должно случиться
                        y_true_target_array[
                            r_final, c_final, a_idx_final, 4] = 0.0  # Делаем фоном, если что-то пошло не так
                        continue

                    gt_class_id_enc = gt_data_for_encoding['class_id']
                    gt_xc_n_enc, gt_yc_n_enc = gt_data_for_encoding['xywh_norm'][0], gt_data_for_encoding['xywh_norm'][
                        1]
                    gt_w_n_enc, gt_h_n_enc = gt_data_for_encoding['xywh_norm'][2], gt_data_for_encoding['xywh_norm'][3]
                    tx_enc = (gt_xc_n_enc * grid_w) - c_final;
                    ty_enc = (gt_yc_n_enc * grid_h) - r_final
                    anchor_w_n_sel_enc, anchor_h_n_sel_enc = anchors_wh_level_config_np[a_idx_final]
                    safe_gt_w_enc, safe_gt_h_enc = max(gt_w_n_enc, 1e-9), max(gt_h_n_enc, 1e-9)
                    safe_anchor_w_enc, safe_anchor_h_enc = max(anchor_w_n_sel_enc, 1e-9), max(anchor_h_n_sel_enc, 1e-9)
                    tw_enc = np.log(safe_gt_w_enc / safe_anchor_w_enc);
                    th_enc = np.log(safe_gt_h_enc / safe_anchor_h_enc)
                    y_true_target_array[r_final, c_final, a_idx_final, 0:4] = [tx_enc, ty_enc, tw_enc, th_enc]
                    y_true_target_array[r_final, c_final, a_idx_final, 5:] = 0.0
                    y_true_target_array[r_final, c_final, a_idx_final, 5 + gt_class_id_enc] = 1.0
                else:
                    max_iou_with_any_gt_for_this_anchor = 0.0
                    # Используем gt_boxes_xyxy_norm_all_np_for_ignore, который был определен в Шаге 1
                    if gt_boxes_xyxy_norm_all_np_for_ignore.shape[0] > 0:
                        anchor_w_n_empty, anchor_h_n_empty = anchors_wh_level_config_np[a_idx_final]
                        anchor_xc_n_empty = (c_final + 0.5) / grid_w;
                        anchor_yc_n_empty = (r_final + 0.5) / grid_h
                        anc_xyxy_n_empty = [anchor_xc_n_empty - anchor_w_n_empty / 2,
                                            anchor_yc_n_empty - anchor_h_n_empty / 2,
                                            anchor_xc_n_empty + anchor_w_n_empty / 2,
                                            anchor_yc_n_empty + anchor_h_n_empty / 2]
                        for gt_xyxy_norm_ign_check in gt_boxes_xyxy_norm_all_np_for_ignore:  # <--- ИСПРАВЛЕНО
                            iou_val_ign = calculate_iou_xyxy_numpy(gt_xyxy_norm_ign_check, anc_xyxy_n_empty)
                            if iou_val_ign > max_iou_with_any_gt_for_this_anchor:
                                max_iou_with_any_gt_for_this_anchor = iou_val_ign

                    if max_iou_with_any_gt_for_this_anchor > iou_ignore_threshold_arg:
                        pass  # objectness остается -1.0
                    else:
                        y_true_target_array[r_final, c_final, a_idx_final, 4] = 0.0

    return y_true_target_array


# --- load_and_prepare_single_level_v2_py_func ---
# Должен вызывать assign_gt_to_anchors_hybrid_strategy
def load_and_prepare_single_level_v2_py_func(image_path_tensor, xml_path_tensor, py_apply_augmentation_tensor):
    # ... (начальная часть загрузки изображения и XML, аугментация, предобработка изображения - как раньше) ...
    # ... (ВАЖНО: w_for_assign, h_for_assign должны быть размерами изображения, НА КОТОРОМ находятся
    #      boxes_after_aug_or_original_pixels перед передачей в assign_gt_to_anchors_hybrid_strategy)
    image_path_str = image_path_tensor.numpy().decode('utf-8');
    xml_path_str = xml_path_tensor.numpy().decode('utf-8');
    apply_aug = bool(py_apply_augmentation_tensor.numpy()) if hasattr(py_apply_augmentation_tensor, 'numpy') else bool(
        py_apply_augmentation_tensor);
    py_target_h, py_target_w = TARGET_IMG_HEIGHT_SDL_V2_G, TARGET_IMG_WIDTH_SDL_V2_G;
    level_cfg_for_y_true = SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G;
    img_fallback = np.zeros((py_target_h, py_target_w, 3), dtype=np.float32);
    y_true_fallback = np.zeros((level_cfg_for_y_true['grid_h'], level_cfg_for_y_true['grid_w'],
                                level_cfg_for_y_true['num_anchors'], 5 + NUM_CLASSES_SDL_V2_G), dtype=np.float32);
    scaled_boxes_viz_fallback = np.zeros((0, 4), dtype=np.float32);
    class_ids_viz_fallback = np.zeros((0), dtype=np.int32);
    error_return_tuple_sdl_v2 = (img_fallback, y_true_fallback, scaled_boxes_viz_fallback, class_ids_viz_fallback)
    try:
        pil_image = PILImage.open(image_path_str).convert('RGB');
        image_original_np_uint8 = np.array(pil_image, dtype=np.uint8);
        original_pil_w, original_pil_h = pil_image.size
        if original_pil_w <= 0 or original_pil_h <= 0: return error_return_tuple_sdl_v2
    except Exception:
        return error_return_tuple_sdl_v2
    image_filename_debug = os.path.basename(image_path_str)
    objects, xml_w, xml_h, _ = parse_xml_annotation(xml_path_str if xml_path_str else None, CLASSES_LIST_SDL_V2_G)
    if objects is None: objects = []
    effective_original_w = xml_w if xml_w and xml_w > 0 else original_pil_w;
    effective_original_h = xml_h if xml_h and xml_h > 0 else original_pil_h
    if effective_original_w <= 0 or effective_original_h <= 0: return error_return_tuple_sdl_v2
    boxes_pixels_from_xml = [[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']] for obj in
                             objects] if objects else [];
    class_ids_from_xml = [obj['class_id'] for obj in objects] if objects else []
    image_after_aug_or_original_uint8 = image_original_np_uint8;
    boxes_after_aug_or_original_pixels = list(boxes_pixels_from_xml);
    class_ids_after_aug_or_original = list(class_ids_from_xml)

    # w_for_assign, h_for_assign - это размеры изображения, к которому относятся boxes_after_aug_or_original_pixels
    w_for_assign, h_for_assign = effective_original_w, effective_original_h  # По умолчанию - оригинальные размеры

    if apply_aug and AUGMENTATION_FUNC_AVAILABLE_SDL_V2 and objects:
        try:
            # Аугментируем изображение ДО изменения его размера на TARGET_IMG_HEIGHT/WIDTH
            augmenter = get_detector_train_augmentations_sdl_v2(original_pil_h, original_pil_w)
            if augmenter:
                temp_class_labels_for_aug = [str(cid) for cid in class_ids_from_xml]
                augmented_data = augmenter(image=image_original_np_uint8, bboxes=boxes_pixels_from_xml,
                                           class_labels_for_albumentations=temp_class_labels_for_aug)
                image_after_aug_or_original_uint8 = augmented_data['image']
                boxes_after_aug_or_original_pixels = augmented_data[
                    'bboxes']  # Это пиксельные координаты для аугментированного изображения

                # Обновляем размеры, если аугментация их изменила (некоторые могут)
                h_for_assign, w_for_assign = image_after_aug_or_original_uint8.shape[:2]

                final_class_ids_after_aug = []
                if 'class_labels_for_albumentations' in augmented_data and augmented_data['bboxes']:
                    try:
                        final_class_ids_after_aug = [int(cls_label) for cls_label in
                                                     augmented_data['class_labels_for_albumentations']]
                    except ValueError:
                        final_class_ids_after_aug = class_ids_from_xml if len(augmented_data['bboxes']) == len(
                            class_ids_from_xml) else []
                elif not augmented_data['bboxes']:
                    final_class_ids_after_aug = []
                else:
                    final_class_ids_after_aug = list(
                        class_ids_from_xml)  # Если метки классов не вернулись, но боксы есть
                class_ids_after_aug_or_original = final_class_ids_after_aug

                if not boxes_after_aug_or_original_pixels: class_ids_after_aug_or_original = []
        except Exception as e_aug_exc:
            print(
                f"WARNING: Ошибка аугментации для {image_filename_debug}: {e_aug_exc}. Используются оригинальные данные.")
            image_after_aug_or_original_uint8, boxes_after_aug_or_original_pixels, class_ids_after_aug_or_original = image_original_np_uint8, list(
                boxes_pixels_from_xml), list(class_ids_from_xml)
            w_for_assign, h_for_assign = effective_original_w, effective_original_h

    image_tensor_for_tf = tf.convert_to_tensor(image_after_aug_or_original_uint8.astype(np.float32), dtype=tf.float32)
    temp_boxes_tensor_pixels = tf.constant(boxes_after_aug_or_original_pixels,
                                           dtype=tf.float32) if boxes_after_aug_or_original_pixels else tf.zeros((0, 4),
                                                                                                                 dtype=tf.float32)

    # preprocess_image_and_boxes теперь вызывается с изображением, которое МОЖЕТ быть уже аугментировано
    # и его РЕАЛЬНЫМИ текущими размерами (h_for_assign, w_for_assign), если аугментация их изменила.
    # Однако, preprocess_image_and_boxes сама берет tf.shape(image_tensor_for_tf), так что w_for_assign, h_for_assign не нужны ей явно.
    # Но они нужны для assign_gt_to_anchors_hybrid_strategy!

    image_processed_for_model_tf, scaled_gt_boxes_norm_for_viz_tf = preprocess_image_and_boxes(
        image_tensor_for_tf, temp_boxes_tensor_pixels,
        tf.constant(TARGET_IMG_HEIGHT_SDL_V2_G, dtype=tf.int32),
        tf.constant(TARGET_IMG_WIDTH_SDL_V2_G, dtype=tf.int32)
    )

    if len(boxes_after_aug_or_original_pixels) != len(
            class_ids_after_aug_or_original) and boxes_after_aug_or_original_pixels:
        return error_return_tuple_sdl_v2

    y_true_single_level_np = assign_gt_to_anchors_hybrid_strategy(
        boxes_after_aug_or_original_pixels,  # Пиксельные координаты (возможно, после аугментации)
        class_ids_after_aug_or_original,  # Соответствующие им ID классов
        w_for_assign, h_for_assign,  # Размеры изображения, к которому относятся эти пиксельные координаты
        level_config_arg=SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G,
        iou_positive_threshold_arg=IOU_POSITIVE_THRESHOLD_SDL_G,
        iou_ignore_threshold_arg=IOU_IGNORE_THRESHOLD_SDL_G,
        anchor_shape_match_thresh_arg=ANCHOR_SHAPE_MATCHING_THRESHOLD_SDL_G,
        image_filename_for_debug=image_filename_debug
    )

    final_class_ids_for_viz_np = np.array(class_ids_after_aug_or_original,
                                          dtype=np.int32) if class_ids_after_aug_or_original else np.zeros((0),
                                                                                                           dtype=np.int32)

    return image_processed_for_model_tf.numpy(), y_true_single_level_np, scaled_gt_boxes_norm_for_viz_tf.numpy(), final_class_ids_for_viz_np


# --- load_and_prepare_single_level_v2_tf_wrapper и create_detector_single_level_v2_tf_dataset ---
# (Остаются такими же)
# ... (твой код этих функций) ...
def load_and_prepare_single_level_v2_tf_wrapper(image_path_tensor, xml_path_tensor, augment_tensor):
    img_processed_np, y_true_np_out, scaled_boxes_viz_np, class_ids_viz_np = tf.py_function(
        func=load_and_prepare_single_level_v2_py_func, inp=[image_path_tensor, xml_path_tensor, augment_tensor],
        Tout=[tf.float32, tf.float32, tf.float32, tf.int32])
    img_processed_np.set_shape([TARGET_IMG_HEIGHT_SDL_V2_G, TARGET_IMG_WIDTH_SDL_V2_G, 3]);
    y_true_shape_tf = (
    GRID_H_P4_DEBUG_SDL_V2_G, GRID_W_P4_DEBUG_SDL_V2_G, NUM_ANCHORS_P4_DEBUG_SDL_V2_G, 5 + NUM_CLASSES_SDL_V2_G)
    y_true_np_out.set_shape(y_true_shape_tf);
    return img_processed_np, y_true_np_out


def create_detector_single_level_v2_tf_dataset(image_paths_list_arg, xml_paths_list_arg, batch_size_arg,
                                               shuffle_arg=True, augment_arg=False):
    if not image_paths_list_arg or not xml_paths_list_arg: return None
    if len(image_paths_list_arg) != len(xml_paths_list_arg): raise ValueError("Counts differ for images and xmls.")
    augment_flags_ds = tf.constant([augment_arg] * len(image_paths_list_arg), dtype=tf.bool)
    dataset_created = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths_list_arg, dtype=tf.string),
                                                          tf.constant(xml_paths_list_arg, dtype=tf.string),
                                                          augment_flags_ds))
    if shuffle_arg: dataset_created = dataset_created.shuffle(buffer_size=max(1, len(image_paths_list_arg)),
                                                              reshuffle_each_iteration=True)
    dataset_created = dataset_created.map(load_and_prepare_single_level_v2_tf_wrapper,
                                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset_created = dataset_created.batch(batch_size_arg, drop_remainder=True);
    dataset_created = dataset_created.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_created

# Блок if __name__ == '__main__': будет добавлен в следующем ответе, когда ты подтвердишь этот код.
if __name__ == '__main__':
    import argparse  # Убедись, что random и sys импортированы в начале detector_data_loader_single_level_v2.py

    # import random # Должен быть импортирован глобально
    # import sys    # Должен быть импортирован глобально

    # Локальная переменная для этого блока, чтобы не конфликтовать с глобальной, если она есть
    _matplotlib_available_for_this_main_test = False
    try:
        import matplotlib.pyplot as plt  # Переименовал для ясности области видимости

        _matplotlib_available_for_this_main_test = True
    except ImportError:
        pass  # Если не установлен, просто не будем вызывать plt.close()

    parser = argparse.ArgumentParser(
        description="Тестирование и визуализация detector_data_loader_single_level_v2.py с новой логикой GT.")
    parser.add_argument(
        "--num_examples", type=int, default=2,
        help="Количество случайных примеров для обработки и визуализации (дефолт: 2)."
    )
    parser.add_argument(
        "--enable_visualization", action='store_true',
        help="Включить matplotlib визуализацию."
    )
    parser.add_argument(
        "--force_augmentation_test", action='store_true',
        help="Принудительно включить аугментацию для этого теста."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed для ГСЧ Python random (для воспроизводимости выбора файлов)."
    )
    args = parser.parse_args()

    # Эта переменная будет управлять визуализацией внутри этого блока __main__
    _visualization_enabled_for_this_run_main = False
    if args.enable_visualization:
        if _plot_utils_v2_imported_successfully:  # Используем глобальный флаг из начала файла
            _visualization_enabled_for_this_run_main = True
            print("INFO (__main__ sdl_v2): Визуализация ВКЛЮЧЕНА (--enable_visualization) и plot_utils_v2 доступен.")
        else:
            print(
                "ПРЕДУПРЕЖДЕНИЕ (__main__ sdl_v2): Визуализация запрошена (--enable_visualization), НО plot_utils_v2 НЕ импортирован успешно. Визуализация ОТКЛЮЧЕНА.")
    else:
        if _plot_utils_v2_imported_successfully:
            print(
                "INFO (__main__ sdl_v2): Визуализация ОТКЛЮЧЕНА (флаг --enable_visualization не указан, но plot_utils_v2 доступен).")
        else:
            print(
                "INFO (__main__ sdl_v2): Визуализация ОТКЛЮЧЕНА (plot_utils_v2 не импортирован И флаг --enable_visualization не указан).")

    if not CONFIG_LOAD_SUCCESS_SDL_V2_GLOBAL:
        print("\n!!! ВНИМАНИЕ: Один или оба конфигурационных файла не были загружены корректно. "
              "Тестирование может использовать неактуальные или аварийные дефолтные параметры.")

    print(f"\n--- Тестирование detector_data_loader_single_level_v2.py ---")
    print(f"  Количество примеров для показа: {args.num_examples}")
    print(f"  Визуализация во время этого теста: {_visualization_enabled_for_this_run_main}")  # Используем локальную
    if args.seed is not None:
        print(f"  Используется SEED для выбора файлов: {args.seed}")
    else:
        print(f"  Используется SEED для выбора файлов: Случайный (None)")

    # Эта переменная будет управлять аугментацией внутри этого блока __main__
    _current_test_augmentation_flag_main = USE_AUGMENTATION_SDL_V2_FLAG_G  # Из конфига по умолчанию
    if args.force_augmentation_test:
        if AUGMENTATION_FUNC_AVAILABLE_SDL_V2:
            _current_test_augmentation_flag_main = True
            print(f"  Аугментация для теста в __main__ ПРИНУДИТЕЛЬНО ВКЛЮЧЕНА (--force_augmentation_test).")
        else:
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Запрошена принудительная аугментация, но функция не доступна. Аугментация останется ВЫКЛЮЧЕННОЙ.")
    else:
        print(f"  Аугментация для теста в __main__ (из конфига/доступности): {_current_test_augmentation_flag_main}")

    print(f"\n  Параметры для уровня '{FPN_LEVEL_NAME_DEBUG_SDL_V2_G}': "
          f"Сетка({GRID_H_P4_DEBUG_SDL_V2_G}x{GRID_W_P4_DEBUG_SDL_V2_G}), "
          f"Якорей={NUM_ANCHORS_P4_DEBUG_SDL_V2_G}, Классов={NUM_CLASSES_SDL_V2_G}")
    print(
        f"  Пороги IoU для назначения GT: Positive >= {IOU_POSITIVE_THRESHOLD_SDL_G}, Ignore > {IOU_IGNORE_THRESHOLD_SDL_G}")

    _master_dataset_path = BASE_CONFIG_SDL_V2_GLOBAL.get('master_dataset_path', 'data/Master_Dataset_Fallback_SDL')
    if not Path(_master_dataset_path).is_absolute():
        master_dataset_abs_path = (_project_root_sdl_v2 / _master_dataset_path).resolve()
    else:
        master_dataset_abs_path = Path(_master_dataset_path).resolve()
    _source_subdirs_keys = [
        BASE_CONFIG_SDL_V2_GLOBAL.get('source_defective_road_img_parent_subdir', 'Defective_Road_Images'),
        BASE_CONFIG_SDL_V2_GLOBAL.get('source_normal_road_img_parent_subdir', 'Normal_Road_Images'),
        BASE_CONFIG_SDL_V2_GLOBAL.get('source_not_road_img_parent_subdir', 'Not_Road_Images')]
    _images_subdir_main = BASE_CONFIG_SDL_V2_GLOBAL.get('dataset', {}).get('images_dir', 'JPEGImages')
    _annotations_subdir_main = BASE_CONFIG_SDL_V2_GLOBAL.get('dataset', {}).get('annotations_dir', 'Annotations')
    all_image_annotation_pairs_main = []
    print(f"\nСканирование мастер-датасета: {master_dataset_abs_path}")
    for parent_subfolder_main in _source_subdirs_keys:
        if not parent_subfolder_main: continue
        images_dir_to_scan = master_dataset_abs_path / parent_subfolder_main / _images_subdir_main
        annotations_dir_to_scan = master_dataset_abs_path / parent_subfolder_main / _annotations_subdir_main
        if not images_dir_to_scan.is_dir(): continue
        temp_image_files = [];
        ext_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext_pat in ext_patterns: temp_image_files.extend(list(images_dir_to_scan.glob(ext_pat)))
        if not temp_image_files: continue
        seen_stems_loop = set();
        unique_img_paths = []
        for f_path in temp_image_files:
            resolved_p = f_path.resolve()
            if resolved_p.stem.lower() not in seen_stems_loop: unique_img_paths.append(resolved_p); seen_stems_loop.add(
                resolved_p.stem.lower())
        for img_p_obj in unique_img_paths:
            xml_p_obj = annotations_dir_to_scan / (img_p_obj.stem + ".xml")
            all_image_annotation_pairs_main.append((str(img_p_obj), str(xml_p_obj) if xml_p_obj.exists() else ""))

    if not all_image_annotation_pairs_main:
        print("ОШИБКА: Не найдено изображений в мастер-датасете для теста.")
    else:
        print(f"Найдено {len(all_image_annotation_pairs_main)} изображений (с потец. аннотациями).")
        num_to_process = min(args.num_examples, len(all_image_annotation_pairs_main))
        if num_to_process == 0:
            print("Нет файлов для теста.")
        else:
            if args.seed is not None: random.seed(args.seed)
            selected_pairs = random.sample(all_image_annotation_pairs_main, num_to_process)
            print(f"\nБудет обработано и (если вкл.) визуализировано {len(selected_pairs)} случ. файлов:")
            for i_example, (img_path, xml_path) in enumerate(selected_pairs):
                print(f"\n--- Пример {i_example + 1}/{len(selected_pairs)}: {os.path.basename(img_path)} ---")
                if not xml_path: print(f"    ПРЕДУПРЕЖДЕНИЕ: XML для {os.path.basename(img_path)} не найден.")
                print(f"  --- Загрузка ОРИГИНАЛА (аугментация: False) ---")
                try:
                    img_np_o, y_true_o, scaled_boxes_viz_o, class_ids_viz_o = \
                        load_and_prepare_single_level_v2_py_func(
                            tf.constant(img_path, dtype=tf.string),
                            tf.constant(xml_path if xml_path else "dummy.xml", dtype=tf.string),
                            tf.constant(False, dtype=tf.bool))
                    print(
                        f"    Изображение: {img_np_o.shape}, y_true: {y_true_o.shape}, GT_viz_boxes: {scaled_boxes_viz_o.shape[0]}")
                    obj_map_o = y_true_o[..., 4];
                    pos_o = tf.where(obj_map_o > 0.5);
                    ign_o = tf.where(obj_map_o < -0.5)
                    print(
                        f"    Позитивных якорей: {tf.shape(pos_o)[0].numpy()}, Игнорируемых: {tf.shape(ign_o)[0].numpy()} (Оригинал)")
                    if _visualization_enabled_for_this_run_main:  # Используем локальную переменную
                        visualize_single_level_gt_assignments_sdl_v2(
                            img_np_o, y_true_o,
                            level_config_for_drawing=SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G,
                            classes_list_for_drawing=CLASSES_LIST_SDL_V2_G,
                            original_gt_boxes_for_ref=(scaled_boxes_viz_o, class_ids_viz_o),
                            title_prefix=f"V2 GT {os.path.basename(img_path)} [ORIGINAL]")
                        if _matplotlib_available_for_this_main_test: plt.close('all')  # Используем локальную
                except Exception as e:
                    print(f"    ОШИБКА (оригинал): {e}"); import traceback; traceback.print_exc()

                if _current_test_augmentation_flag_main:  # Используем локальную переменную
                    print(f"  --- Загрузка АУГМЕНТИРОВАННОЙ версии ---")
                    try:
                        img_np_a, y_true_a, scaled_boxes_viz_a, class_ids_viz_a = \
                            load_and_prepare_single_level_v2_py_func(
                                tf.constant(img_path, dtype=tf.string),
                                tf.constant(xml_path if xml_path else "dummy.xml", dtype=tf.string),
                                tf.constant(True, dtype=tf.bool))
                        print(
                            f"    Изображение: {img_np_a.shape}, y_true: {y_true_a.shape}, GT_viz_boxes: {scaled_boxes_viz_a.shape[0]}")
                        obj_map_a = y_true_a[..., 4];
                        pos_a = tf.where(obj_map_a > 0.5);
                        ign_a = tf.where(obj_map_a < -0.5)
                        print(
                            f"    Позитивных якорей: {tf.shape(pos_a)[0].numpy()}, Игнорируемых: {tf.shape(ign_a)[0].numpy()} (Аугм.)")
                        if _visualization_enabled_for_this_run_main:
                            visualize_single_level_gt_assignments_sdl_v2(
                                img_np_a, y_true_a,
                                level_config_for_drawing=SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G,
                                classes_list_for_drawing=CLASSES_LIST_SDL_V2_G,
                                original_gt_boxes_for_ref=(scaled_boxes_viz_a, class_ids_viz_a),
                                title_prefix=f"V2 GT {os.path.basename(img_path)} [AUGMENTED]")
                            if _matplotlib_available_for_this_main_test: plt.close('all')
                    except Exception as e:
                        print(f"    ОШИБКА (аугментация): {e}"); import traceback; traceback.print_exc()
    print("\n--- Тестирование detector_data_loader_single_level_v2.py завершено ---")