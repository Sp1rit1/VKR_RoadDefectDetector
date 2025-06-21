# src/datasets/detector_data_loader_single_level_debug.py
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image as PILImage

# --- Настройка sys.path для импорта augmentations и plot_utils ---
_current_script_dir_sdl = Path(__file__).resolve().parent  # src/datasets/
_src_dir_sdl = _current_script_dir_sdl.parent  # src/
_project_root_sdl = _src_dir_sdl.parent  # Корень проекта

if str(_src_dir_sdl) not in sys.path: sys.path.insert(0, str(_src_dir_sdl))
if str(_project_root_sdl) not in sys.path: sys.path.insert(0, str(_project_root_sdl))

# --- Импорт Аугментаций (с заглушкой, так как в этом скрипте аугментация обычно отключается) ---
AUGMENTATION_FUNC_AVAILABLE_SDL = False
get_detector_train_augmentations_sdl = lambda h, w: None  # Заглушка
try:
    from datasets.augmentations import get_detector_train_augmentations as aug_func_imported_sdl

    get_detector_train_augmentations_sdl = aug_func_imported_sdl
    AUGMENTATION_FUNC_AVAILABLE_SDL = True
    # print("INFO (sdl_loader): Модуль augmentations.py успешно импортирован.")
except ImportError:
    # print("ПРЕДУПРЕЖДЕНИЕ (sdl_loader): augmentations.py не найден. Аугментация будет отключена.")
    pass
except Exception as e_aug_sdl:
    print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader): Ошибка импорта augmentations: {e_aug_sdl}. Аугментация отключена.")

# --- Загрузка ОТЛАДОЧНОЙ Конфигурации ---
# Сначала загружаем ОБЩИЙ базовый конфиг, затем отладочный детекторный
_base_config_path_sdl_global = _src_dir_sdl / 'configs' / 'base_config.yaml'
_debug_config_path_sdl_global = _src_dir_sdl / 'configs' / 'detector_config_single_level_debug.yaml'

BASE_CONFIG_SDL_GLOBAL = {}
DEBUG_SDL_CONFIG_GLOBAL = {}  # Этот будет содержать содержимое detector_config_single_level_debug.yaml
CONFIG_LOAD_SUCCESS_SDL_GLOBAL = True

try:
    with open(_base_config_path_sdl_global, 'r', encoding='utf-8') as f:
        BASE_CONFIG_SDL_GLOBAL = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG_SDL_GLOBAL, dict) or not BASE_CONFIG_SDL_GLOBAL:
        # print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader): Файл {_base_config_path_sdl_global.name} пуст или неверный формат.")
        CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
        BASE_CONFIG_SDL_GLOBAL = {}
except FileNotFoundError:
    # print(f"ОШИБКА (sdl_loader): Файл {_base_config_path_sdl_global.name} не найден.")
    CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
    BASE_CONFIG_SDL_GLOBAL = {}
except yaml.YAMLError as e_cfg_base_sdl:
    # print(f"ОШИБКА YAML (sdl_loader): Не удалось прочитать {_base_config_path_sdl_global.name}: {e_cfg_base_sdl}")
    CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
    BASE_CONFIG_SDL_GLOBAL = {}

try:
    with open(_debug_config_path_sdl_global, 'r', encoding='utf-8') as f:
        DEBUG_SDL_CONFIG_GLOBAL = yaml.safe_load(f)  # Загружаем отладочный конфиг
    if not isinstance(DEBUG_SDL_CONFIG_GLOBAL, dict) or not DEBUG_SDL_CONFIG_GLOBAL:
        # print(f"ПРЕДУПРЕЖДЕНИЕ (sdl_loader): Файл {_debug_config_path_sdl_global.name} пуст или неверный формат.")
        CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
        DEBUG_SDL_CONFIG_GLOBAL = {}
except FileNotFoundError:
    # print(f"ОШИБКА (sdl_loader): Файл {_debug_config_path_sdl_global.name} не найден.")
    CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
    DEBUG_SDL_CONFIG_GLOBAL = {}
except yaml.YAMLError as e_cfg_debug_sdl:
    # print(f"ОШИБКА YAML (sdl_loader): Не удалось прочитать {_debug_config_path_sdl_global.name}: {e_cfg_debug_sdl}")
    CONFIG_LOAD_SUCCESS_SDL_GLOBAL = False;
    DEBUG_SDL_CONFIG_GLOBAL = {}

if not CONFIG_LOAD_SUCCESS_SDL_GLOBAL:
    print("ПРЕДУПРЕЖДЕНИЕ (sdl_loader): Используются АВАРИЙНЫЕ ДЕФОЛТЫ (один или оба конфига не загружены).")
    BASE_CONFIG_SDL_GLOBAL.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    BASE_CONFIG_SDL_GLOBAL.setdefault('master_dataset_path', 'data/Master_Dataset_Fallback_SDL')

    # Аварийные дефолты для DEBUG_SDL_CONFIG_GLOBAL (отладочного конфига)
    _fpn_params_default_sdl = {
        'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
        'detector_fpn_levels': ['P4_debug'],
        'detector_fpn_strides': {'P4_debug': 16},
        'detector_fpn_anchor_configs': {
            'P4_debug': {'num_anchors_this_level': 1, 'anchors_wh_normalized': [[0.15, 0.15]]}}
    }
    DEBUG_SDL_CONFIG_GLOBAL.setdefault('fpn_detector_params', _fpn_params_default_sdl)
    DEBUG_SDL_CONFIG_GLOBAL.setdefault('use_augmentation', False)
    DEBUG_SDL_CONFIG_GLOBAL.setdefault('initial_learning_rate',
                                       0.0001)  # Добавим дефолты для параметров, которые могут использоваться в __main__
    DEBUG_SDL_CONFIG_GLOBAL.setdefault('epochs_for_debug', 200)
    DEBUG_SDL_CONFIG_GLOBAL.setdefault('predict_params',
                                       {'confidence_threshold': 0.05, 'iou_threshold': 0.3, 'max_detections': 50})

# --- Глобальные Параметры из ОТЛАДОЧНОГО Конфига DEBUG_SDL_CONFIG_GLOBAL ---
_fpn_params_sdl = DEBUG_SDL_CONFIG_GLOBAL.get('fpn_detector_params', {})
INPUT_SHAPE_SDL_G = tuple(_fpn_params_sdl.get('input_shape', [416, 416, 3]))
TARGET_IMG_HEIGHT_SDL_G, TARGET_IMG_WIDTH_SDL_G = INPUT_SHAPE_SDL_G[0], INPUT_SHAPE_SDL_G[1]
CLASSES_LIST_SDL_G = _fpn_params_sdl.get('classes', ['pit', 'crack'])
NUM_CLASSES_SDL_G = len(CLASSES_LIST_SDL_G)
# Флаг аугментации для этого загрузчика берется из ЕГО отладочного конфига
USE_AUGMENTATION_SDL_LOADER_FLAG_G = DEBUG_SDL_CONFIG_GLOBAL.get('use_augmentation',
                                                                 False) and AUGMENTATION_FUNC_AVAILABLE_SDL

# Параметры для единственного отладочного уровня (например, 'P4_debug')
FPN_LEVEL_NAME_DEBUG_SDL_G = _fpn_params_sdl.get('detector_fpn_levels', ['P4_debug'])[0]
_p4_debug_stride_cfg = _fpn_params_sdl.get('detector_fpn_strides', {}).get(FPN_LEVEL_NAME_DEBUG_SDL_G, 16)
_p4_debug_anchor_cfg_yaml = _fpn_params_sdl.get('detector_fpn_anchor_configs', {}).get(FPN_LEVEL_NAME_DEBUG_SDL_G, {})

ANCHORS_WH_P4_DEBUG_SDL_G = np.array(
    _p4_debug_anchor_cfg_yaml.get('anchors_wh_normalized', [[0.15, 0.15]]), dtype=np.float32
)
NUM_ANCHORS_P4_DEBUG_SDL_G = _p4_debug_anchor_cfg_yaml.get('num_anchors_this_level', ANCHORS_WH_P4_DEBUG_SDL_G.shape[0])

if ANCHORS_WH_P4_DEBUG_SDL_G.ndim == 1 and ANCHORS_WH_P4_DEBUG_SDL_G.shape[
    0] == 2:  # Если якорь [[0.15, 0.15]] прочитался как [0.15, 0.15]
    ANCHORS_WH_P4_DEBUG_SDL_G = np.expand_dims(ANCHORS_WH_P4_DEBUG_SDL_G, axis=0)

if ANCHORS_WH_P4_DEBUG_SDL_G.shape[0] != NUM_ANCHORS_P4_DEBUG_SDL_G:  # Синхронизация
    NUM_ANCHORS_P4_DEBUG_SDL_G = ANCHORS_WH_P4_DEBUG_SDL_G.shape[0]

GRID_H_P4_DEBUG_SDL_G = TARGET_IMG_HEIGHT_SDL_G // _p4_debug_stride_cfg
GRID_W_P4_DEBUG_SDL_G = TARGET_IMG_WIDTH_SDL_G // _p4_debug_stride_cfg

# Словарь с конфигурацией для ОДНОГО отладочного уровня
SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G = {
    'grid_h': GRID_H_P4_DEBUG_SDL_G, 'grid_w': GRID_W_P4_DEBUG_SDL_G,
    'anchors_wh_normalized': ANCHORS_WH_P4_DEBUG_SDL_G,  # Это должен быть 2D numpy array
    'num_anchors': NUM_ANCHORS_P4_DEBUG_SDL_G,
    'stride': _p4_debug_stride_cfg
}


# --- Вспомогательные функции ---
# parse_xml_annotation, preprocess_image_and_boxes, calculate_iou_numpy, assign_gt_to_single_level_and_encode
# (Код этих функций остается таким же, как в твоем файле load_detector1.txt или моем предыдущем ответе, где мы их отладили.
#  Убедись, что они используют глобальные переменные, определенные ВЫШЕ в этом файле, если они им нужны,
#  или получают все необходимые параметры через аргументы.)

# Для примера, я вставлю их версии, которые мы обсуждали:
def parse_xml_annotation(xml_file_path, classes_list_arg=CLASSES_LIST_SDL_G):
    try:
        tree = ET.parse(xml_file_path);
        root = tree.getroot();
        image_filename_node = root.find('filename')
        image_filename = image_filename_node.text if image_filename_node is not None and image_filename_node.text else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")
        size_node = root.find('size');
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
        if img_width_xml and img_height_xml:  # Клиппинг
            xmin = max(0.0, min(xmin, float(img_width_xml)));
            ymin = max(0.0, min(ymin, float(img_height_xml)))
            xmax = max(0.0, min(xmax, float(img_width_xml)));
            ymax = max(0.0, min(ymax, float(img_height_xml)))
            if xmin >= xmax or ymin >= ymax: continue
        objects.append(
            {"class_id": class_id, "class_name": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return objects, img_width_xml, img_height_xml, image_filename


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


def calculate_iou_numpy(box_wh_norm, anchors_wh_norm_level):
    box_wh_np = np.array(box_wh_norm);
    anchors_wh_level_np = np.array(anchors_wh_norm_level)
    inter_w = np.minimum(box_wh_np[0], anchors_wh_level_np[:, 0]);
    inter_h = np.minimum(box_wh_np[1], anchors_wh_level_np[:, 1])
    intersection = inter_w * inter_h;
    box_area = box_wh_np[0] * box_wh_np[1];
    anchor_area = anchors_wh_level_np[:, 0] * anchors_wh_level_np[:, 1]
    union = box_area + anchor_area - intersection;
    return intersection / (union + 1e-6)


def assign_gt_to_single_level_and_encode(
        gt_boxes_pixels_list_arg, gt_class_ids_list_arg,
        original_img_width_px_arg, original_img_height_px_arg):
    level_config = SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G  # Используем глобальный отладочный конфиг для P4
    grid_h = level_config['grid_h'];
    grid_w = level_config['grid_w']
    anchors_wh_this_level = level_config['anchors_wh_normalized']
    num_anchors_this_level = level_config['num_anchors']
    y_true_output_array = np.zeros((grid_h, grid_w, num_anchors_this_level, 5 + NUM_CLASSES_SDL_G), dtype=np.float32)
    anchor_assigned_on_grid = np.zeros((grid_h, grid_w, num_anchors_this_level), dtype=bool)
    if not gt_boxes_pixels_list_arg or original_img_width_px_arg <= 0 or original_img_height_px_arg <= 0:
        return y_true_output_array
    for i_obj in range(len(gt_boxes_pixels_list_arg)):
        xmin_px, ymin_px, xmax_px, ymax_px = gt_boxes_pixels_list_arg[i_obj]
        gt_class_id_current = int(gt_class_ids_list_arg[i_obj])
        gt_box_w_px = xmax_px - xmin_px;
        gt_box_h_px = ymax_px - ymin_px
        if gt_box_w_px <= 0 or gt_box_h_px <= 0: continue
        gt_center_x_norm = (xmin_px + gt_box_w_px / 2.0) / original_img_width_px_arg
        gt_center_y_norm = (ymin_px + gt_box_h_px / 2.0) / original_img_height_px_arg
        gt_width_norm = gt_box_w_px / original_img_width_px_arg
        gt_height_norm = gt_box_h_px / original_img_height_px_arg
        grid_x_center_float = gt_center_x_norm * float(grid_w)
        grid_y_center_float = gt_center_y_norm * float(grid_h)
        grid_x_idx = min(int(grid_x_center_float), grid_w - 1)
        grid_y_idx = min(int(grid_y_center_float), grid_h - 1)
        if num_anchors_this_level == 0 or anchors_wh_this_level.shape[0] == 0: continue
        ious = calculate_iou_numpy([gt_width_norm, gt_height_norm], anchors_wh_this_level)
        best_anchor_idx = int(np.argmax(ious))
        if not anchor_assigned_on_grid[grid_y_idx, grid_x_idx, best_anchor_idx]:
            anchor_assigned_on_grid[grid_y_idx, grid_x_idx, best_anchor_idx] = True
            y_true_output_array[grid_y_idx, grid_x_idx, best_anchor_idx, 4] = 1.0
            tx = grid_x_center_float - float(grid_x_idx);
            ty = grid_y_center_float - float(grid_y_idx)
            anchor_w_n_sel, anchor_h_n_sel = anchors_wh_this_level[best_anchor_idx]
            safe_gt_w_n = max(gt_width_norm, 1e-9);
            safe_gt_h_n = max(gt_height_norm, 1e-9)
            safe_anchor_w_n = max(anchor_w_n_sel, 1e-9);
            safe_anchor_h_n = max(anchor_h_n_sel, 1e-9)
            tw = np.log(safe_gt_w_n / safe_anchor_w_n);
            th = np.log(safe_gt_h_n / safe_anchor_h_n)
            y_true_output_array[grid_y_idx, grid_x_idx, best_anchor_idx, 0:4] = [tx, ty, tw, th]
            y_true_output_array[grid_y_idx, grid_x_idx, best_anchor_idx, 5 + gt_class_id_current] = 1.0
    return y_true_output_array


# --- Ключевые функции для загрузки (адаптированные под single_level_debug) ---
def load_and_prepare_single_level_debug_py_func(image_path_tensor, xml_path_tensor, py_apply_augmentation_tensor):
    # ... (Твой код load_and_prepare_single_level_debug_py_func из предыдущего ответа,
    #      который возвращает 4 значения: image_processed_numpy, y_true_single_level_numpy,
    #      scaled_gt_boxes_norm_numpy, final_class_ids_for_viz_numpy.
    #      Убедись, что он использует ГЛОБАЛЬНЫЕ переменные этого модуля для параметров,
    #      такие как TARGET_IMG_HEIGHT_SDL_G, CLASSES_LIST_SDL_G,
    #      и вызывает assign_gt_to_single_level_and_encode)
    # Я скопирую твою последнюю рабочую версию этого блока:
    image_path_str = image_path_tensor.numpy().decode('utf-8')
    xml_path_str = xml_path_tensor.numpy().decode('utf-8')
    apply_aug = bool(py_apply_augmentation_tensor.numpy()) if hasattr(py_apply_augmentation_tensor, 'numpy') else bool(
        py_apply_augmentation_tensor)
    py_target_h = TARGET_IMG_HEIGHT_SDL_G;
    py_target_w = TARGET_IMG_WIDTH_SDL_G
    py_grid_h = GRID_H_P4_DEBUG_SDL_G;
    py_grid_w = GRID_W_P4_DEBUG_SDL_G
    py_num_anchors = NUM_ANCHORS_P4_DEBUG_SDL_G;
    py_num_classes = NUM_CLASSES_SDL_G
    img_fallback = np.zeros((py_target_h, py_target_w, 3), dtype=np.float32)
    y_true_fallback = np.zeros((py_grid_h, py_grid_w, py_num_anchors, 5 + py_num_classes), dtype=np.float32)
    scaled_boxes_viz_fallback = np.zeros((0, 4), dtype=np.float32)
    class_ids_viz_fallback = np.zeros((0), dtype=np.int32)
    error_return_tuple_sdl = (img_fallback, y_true_fallback, scaled_boxes_viz_fallback, class_ids_viz_fallback)
    try:
        pil_image = PILImage.open(image_path_str).convert('RGB')
        image_original_np_uint8 = np.array(pil_image, dtype=np.uint8)
        original_pil_w, original_pil_h = pil_image.size
        if original_pil_w <= 0 or original_pil_h <= 0: return error_return_tuple_sdl
    except Exception:
        return error_return_tuple_sdl
    objects, xml_w, xml_h, _ = parse_xml_annotation(xml_path_str)  # Использует CLASSES_LIST_SDL_G
    if objects is None: return error_return_tuple_sdl
    effective_original_w = xml_w if xml_w and xml_w > 0 else original_pil_w
    effective_original_h = xml_h if xml_h and xml_h > 0 else original_pil_h
    boxes_pixels_from_xml = [[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']] for obj in objects] if objects else []
    class_ids_from_xml = [obj['class_id'] for obj in objects] if objects else []
    image_after_aug_or_original_uint8 = image_original_np_uint8
    boxes_after_aug_or_original_pixels = list(boxes_pixels_from_xml)
    class_ids_after_aug_or_original = list(class_ids_from_xml)
    w_for_assign = effective_original_w;
    h_for_assign = effective_original_h
    if apply_aug and AUGMENTATION_FUNC_AVAILABLE_SDL and objects:
        try:
            augmenter = get_detector_train_augmentations_sdl(original_pil_h, original_pil_w)
            if augmenter:
                augmented_data = augmenter(image=image_original_np_uint8, bboxes=boxes_pixels_from_xml,
                                           class_labels_for_albumentations=class_ids_from_xml)
                image_after_aug_or_original_uint8 = augmented_data['image']
                boxes_after_aug_or_original_pixels = augmented_data['bboxes']
                class_ids_after_aug_or_original = augmented_data['class_labels_for_albumentations']
                h_for_assign = image_after_aug_or_original_uint8.shape[0];
                w_for_assign = image_after_aug_or_original_uint8.shape[1]
                if not boxes_after_aug_or_original_pixels: class_ids_after_aug_or_original = []
        except Exception:
            pass
    image_tensor_for_tf = tf.convert_to_tensor(image_after_aug_or_original_uint8.astype(np.float32), dtype=tf.float32)
    temp_boxes_tensor_pixels = tf.constant(boxes_after_aug_or_original_pixels,
                                           dtype=tf.float32) if boxes_after_aug_or_original_pixels else tf.zeros((0, 4),
                                                                                                                 dtype=tf.float32)
    image_processed_for_model_tf, scaled_gt_boxes_norm_for_viz_tf = preprocess_image_and_boxes(
        image_tensor_for_tf, temp_boxes_tensor_pixels,
        tf.constant(TARGET_IMG_HEIGHT_SDL_G, dtype=tf.int32), tf.constant(TARGET_IMG_WIDTH_SDL_G, dtype=tf.int32))
    if len(boxes_after_aug_or_original_pixels) != len(
        class_ids_after_aug_or_original) and boxes_after_aug_or_original_pixels: return error_return_tuple_sdl
    y_true_single_level_np = assign_gt_to_single_level_and_encode(boxes_after_aug_or_original_pixels,
                                                                  class_ids_after_aug_or_original, w_for_assign,
                                                                  h_for_assign)
    final_class_ids_for_viz_np = np.array(class_ids_after_aug_or_original,
                                          dtype=np.int32) if class_ids_after_aug_or_original else np.zeros((0),
                                                                                                           dtype=np.int32)
    return image_processed_for_model_tf.numpy(), y_true_single_level_np, scaled_gt_boxes_norm_for_viz_tf.numpy(), final_class_ids_for_viz_np


def load_and_prepare_single_level_debug_tf_wrapper(image_path_tensor, xml_path_tensor, augment_tensor):
    img_processed_np, y_true_p4_np_out, scaled_boxes_viz_np, class_ids_viz_np = tf.py_function(
        func=load_and_prepare_single_level_debug_py_func,
        inp=[image_path_tensor, xml_path_tensor, augment_tensor],
        Tout=[tf.float32, tf.float32, tf.float32, tf.int32]
    )
    img_processed_np.set_shape([TARGET_IMG_HEIGHT_SDL_G, TARGET_IMG_WIDTH_SDL_G, 3])
    y_true_p4_shape_tf = (
    GRID_H_P4_DEBUG_SDL_G, GRID_W_P4_DEBUG_SDL_G, NUM_ANCHORS_P4_DEBUG_SDL_G, 5 + NUM_CLASSES_SDL_G)
    y_true_p4_np_out.set_shape(y_true_p4_shape_tf)
    # Для обучения возвращаем только изображение и y_true
    return img_processed_np, y_true_p4_np_out


def create_detector_single_level_tf_dataset(image_paths_list_arg, xml_paths_list_arg, batch_size_arg,
                                            shuffle_arg=True, augment_arg=False):
    # ... (Твой код create_detector_single_level_tf_dataset - он выглядит корректным)
    if not image_paths_list_arg or not xml_paths_list_arg: return None
    if len(image_paths_list_arg) != len(xml_paths_list_arg): raise ValueError("Counts differ.")
    augment_flags_ds = tf.constant([augment_arg] * len(image_paths_list_arg), dtype=tf.bool)
    dataset_created = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths_list_arg, dtype=tf.string),
                                                          tf.constant(xml_paths_list_arg, dtype=tf.string),
                                                          augment_flags_ds))
    if shuffle_arg: dataset_created = dataset_created.shuffle(buffer_size=max(1, len(image_paths_list_arg)),
                                                              reshuffle_each_iteration=True)
    dataset_created = dataset_created.map(
        lambda img_p_map, xml_p_map, aug_f_map: load_and_prepare_single_level_debug_tf_wrapper(img_p_map, xml_p_map,
                                                                                               aug_f_map),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset_created = dataset_created.batch(batch_size_arg, drop_remainder=True)
    dataset_created = dataset_created.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_created


if __name__ == '__main__':
    import sys
    import random
    import argparse

    # matplotlib импортируется внутри visualize_fpn_gt_assignments, если VISUALIZATION_ENABLED_FOR_MAIN_TEST is True
    # Но для plt.close('all') он может понадобиться здесь
    try:
        import matplotlib.pyplot as plt_main_test_sdl

        MATPLOTLIB_AVAILABLE_MAIN_TEST_SDL = True
    except ImportError:
        MATPLOTLIB_AVAILABLE_MAIN_TEST_SDL = False

    parser_main_sdl = argparse.ArgumentParser(
        description="Тестирование и визуализация detector_data_loader_single_level_debug.py.")
    parser_main_sdl.add_argument(
        "--num_examples", type=int, default=3,  # Покажем 3 случайных примера по умолчанию
        help="Количество случайных примеров для обработки и визуализации."
    )
    parser_main_sdl.add_argument(
        "--enable_visualization", action='store_true',
        help="Включить matplotlib визуализацию (может требовать ручного закрытия окон)."
    )
    parser_main_sdl.add_argument(
        "--force_augmentation_test", action='store_true',
        help="Принудительно включить аугментацию для этого теста (даже если в отладочном конфиге выключена)."
    )
    parser_main_sdl.add_argument(
        "--seed", type=int, default=None,  # Оставляем возможность для воспроизводимого выбора
        help="Seed для ГСЧ Python random (для воспроизводимости выбора файлов)."
    )
    args_main_sdl = parser_main_sdl.parse_args()

    VISUALIZATION_ENABLED_FOR_MAIN_TEST_SDL = False
    if args_main_sdl.enable_visualization:
        # _plot_utils_module_imported_successfully была глобальной в предыдущей версии,
        # теперь это _plot_utils_module_imported_successfully (если ты ее импортируешь в начале файла)
        # Давай предположим, что у нас есть способ проверить импорт plot_utils
        _temp_plot_utils_imported_check = False
        try:
            from utils.main_utils.plot_utils import visualize_fpn_gt_assignments  # Попытка импорта

            _temp_plot_utils_imported_check = True
        except ImportError:
            pass

        if _temp_plot_utils_imported_check:
            VISUALIZATION_ENABLED_FOR_MAIN_TEST_SDL = True
            print("INFO (__main__ sdl_loader): Визуализация ВКЛЮЧЕНА (--enable_visualization).")
        else:
            print(
                "ПРЕДУПРЕЖДЕНИЕ (__main__ sdl_loader): Визуализация запрошена, но plot_utils не импортирован. ОТКЛЮЧЕНА.")
    # ... (остальной код проверки визуализации как был)

    if not CONFIG_LOAD_SUCCESS_SDL_GLOBAL:
        print(
            "\n!!! ВНИМАНИЕ: Один или оба конфигурационных файла (base_config.yaml, detector_config_single_level_debug.yaml) "
            "не были загружены корректно. Тестирование может использовать неактуальные или аварийные дефолтные параметры.")

    print(f"\n--- Тестирование detector_data_loader_single_level_debug.py ---")
    print(f"  Количество примеров для показа: {args_main_sdl.num_examples}")
    print(f"  Визуализация во время этого теста: {VISUALIZATION_ENABLED_FOR_MAIN_TEST_SDL}")
    print(
        f"  Используется SEED для выбора файлов: {args_main_sdl.seed if args_main_sdl.seed is not None else 'Случайный (None)'}")

    current_test_augmentation_flag_sdl = USE_AUGMENTATION_SDL_LOADER_FLAG_G  # Из отладочного конфига
    if args_main_sdl.force_augmentation_test:
        if AUGMENTATION_FUNC_AVAILABLE_SDL:
            current_test_augmentation_flag_sdl = True
            print(f"  Аугментация для теста в __main__ ПРИНУДИТЕЛЬНО ВКЛЮЧЕНА (--force_augmentation_test).")
        else:
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Запрошена принудительная аугментация, но функция не доступна. Аугментация останется ВЫКЛЮЧЕННОЙ.")
    else:
        print(f"  Аугментация для теста в __main__ (из отладочного конфига): {current_test_augmentation_flag_sdl}")

    print(f"  Параметры для отладочного уровня '{FPN_LEVEL_NAME_DEBUG_SDL_G}': "
          f"Сетка({GRID_H_P4_DEBUG_SDL_G}x{GRID_W_P4_DEBUG_SDL_G}), "
          f"Якорей={NUM_ANCHORS_P4_DEBUG_SDL_G}, Классов={NUM_CLASSES_SDL_G}")

    # --- Сбор ВСЕХ файлов из "мастер-датасета" согласно base_config.yaml ---
    # (Используем логику, похожую на collect_all_master_data_paths из create_data_splits.py)
    _master_dataset_path_from_base_cfg = BASE_CONFIG_SDL_GLOBAL.get('master_dataset_path',
                                                                    'data/Master_Dataset_Fallback_SDL')
    if not Path(_master_dataset_path_from_base_cfg).is_absolute():
        master_dataset_abs_path_for_main = (_project_root_sdl / _master_dataset_path_from_base_cfg).resolve()
    else:
        master_dataset_abs_path_for_main = Path(_master_dataset_path_from_base_cfg).resolve()

    _source_subdirs_keys_main = [
        'source_defective_road_img_parent_subdir',
        'source_normal_road_img_parent_subdir',
        'source_not_road_img_parent_subdir'
    ]
    _default_source_subdir_names_main = {
        'source_defective_road_img_parent_subdir': "Defective_Road_Images",
        'source_normal_road_img_parent_subdir': "Normal_Road_Images",
        'source_not_road_img_parent_subdir': "Not_Road_Images"
    }
    _images_subdir_name_main = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('images_dir', 'JPEGImages')
    _annotations_subdir_name_main = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('annotations_dir', 'Annotations')

    all_image_annotation_pairs_main = []
    print(f"\nСканирование мастер-датасета: {master_dataset_abs_path_for_main}")

    for subfolder_key_main in _source_subdirs_keys_main:
        parent_subfolder_name_main = BASE_CONFIG_SDL_GLOBAL.get(subfolder_key_main,
                                                                _default_source_subdir_names_main.get(
                                                                    subfolder_key_main))
        if not parent_subfolder_name_main: continue

        images_dir_to_scan_main = master_dataset_abs_path_for_main / parent_subfolder_name_main / _images_subdir_name_main
        annotations_dir_to_scan_main = master_dataset_abs_path_for_main / parent_subfolder_name_main / _annotations_subdir_name_main

        if not images_dir_to_scan_main.is_dir(): continue

        temp_image_files_main = []
        for ext_pat_main in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            temp_image_files_main.extend(list(images_dir_to_scan_main.glob(ext_pat_main)))

        seen_stems_loop_main = set()
        unique_imgs_loop_main = []
        for f_path_loop_main in temp_image_files_main:
            resolved_p_loop = f_path_loop_main.resolve()
            if resolved_p_loop.stem.lower() not in seen_stems_loop_main:
                unique_imgs_loop_main.append(resolved_p_loop)
                seen_stems_loop_main.add(resolved_p_loop.stem.lower())

        for img_p_obj_main_loop in unique_imgs_loop_main:
            xml_p_obj_main_loop = annotations_dir_to_scan_main / (img_p_obj_main_loop.stem + ".xml")
            if xml_p_obj_main_loop.exists():
                all_image_annotation_pairs_main.append((str(img_p_obj_main_loop), str(xml_p_obj_main_loop)))

    if not all_image_annotation_pairs_main:
        print("ОШИБКА: Не найдено ни одной валидной пары изображение/XML в мастер-датасете для теста.")
    else:
        print(f"Найдено {len(all_image_annotation_pairs_main)} валидных пар в мастер-датасете.")
        num_to_process_main_sdl = min(args_main_sdl.num_examples, len(all_image_annotation_pairs_main))

        if num_to_process_main_sdl == 0:
            print("Нет файлов для теста (num_examples=0 или нет доступных валидных пар).")
        else:
            if args_main_sdl.seed is not None:
                random.seed(args_main_sdl.seed)

            selected_pairs_for_main_test = random.sample(all_image_annotation_pairs_main, num_to_process_main_sdl)
            print(
                f"\nБудет обработано и (если включено) визуализировано {len(selected_pairs_for_main_test)} случайных файлов:")

            for i_example_main_loop, (img_path_main_loop, xml_path_main_loop) in enumerate(
                    selected_pairs_for_main_test):
                print(
                    f"\n--- Пример {i_example_main_loop + 1}/{len(selected_pairs_for_main_test)}: {os.path.basename(img_path_main_loop)} ---")

                # 1. Обработка ОРИГИНАЛА (без аугментации)
                print(f"  --- Загрузка ОРИГИНАЛА (аугментация принудительно False) ---")
                try:
                    img_np_o_main, y_true_o_main, scaled_boxes_viz_o_main, class_ids_viz_o_main = \
                        load_and_prepare_single_level_debug_py_func(
                            tf.constant(img_path_main_loop, dtype=tf.string),
                            tf.constant(xml_path_main_loop, dtype=tf.string),
                            tf.constant(False, dtype=tf.bool)
                        )
                    print(
                        f"    Изображение (обработано): {img_np_o_main.shape}, y_true (P4_debug): {y_true_o_main.shape}")
                    print(f"    GT рамок для визуализации (из py_func): {scaled_boxes_viz_o_main.shape[0]}")

                    if VISUALIZATION_ENABLED_FOR_MAIN_TEST_SDL and _temp_plot_utils_imported_check:
                        # Используем visualize_fpn_gt_assignments, адаптируя для одного уровня
                        visualize_fpn_gt_assignments(
                            img_np_o_main, (y_true_o_main,),
                            fpn_level_names=[FPN_LEVEL_NAME_DEBUG_SDL_G],
                            fpn_configs={FPN_LEVEL_NAME_DEBUG_SDL_G: SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G},
                            classes_list=CLASSES_LIST_SDL_G,
                            title_prefix=f"GT (Single Level) for {os.path.basename(img_path_main_loop)} [ORIGINAL]"
                        )
                        if MATPLOTLIB_AVAILABLE_MAIN_TEST_SDL: plt_main_test_sdl.close('all')
                except Exception as e_orig_test_main_loop:
                    print(
                        f"    ОШИБКА при обработке оригинала '{os.path.basename(img_path_main_loop)}': {e_orig_test_main_loop}")
                    import traceback;

                    traceback.print_exc()

                # 2. Обработка С АУГМЕНТАЦИЕЙ (если включена флагом current_test_augmentation_flag_sdl)
                if current_test_augmentation_flag_sdl:
                    print(
                        f"  --- Загрузка АУГМЕНТИРОВАННОЙ версии (current_test_augmentation_flag_sdl={current_test_augmentation_flag_sdl}) ---")
                    try:
                        img_np_a_main, y_true_a_main, scaled_boxes_viz_a_main, class_ids_viz_a_main = \
                            load_and_prepare_single_level_debug_py_func(
                                tf.constant(img_path_main_loop, dtype=tf.string),
                                tf.constant(xml_path_main_loop, dtype=tf.string),
                                tf.constant(True, dtype=tf.bool)  # augment = True
                            )
                        print(f"    Изображение (аугм.): {img_np_a_main.shape}, y_true (аугм.): {y_true_a_main.shape}")
                        print(f"    GT рамок для виз (аугм., из py_func): {scaled_boxes_viz_a_main.shape[0]}")

                        if VISUALIZATION_ENABLED_FOR_MAIN_TEST_SDL and _temp_plot_utils_imported_check:
                            visualize_fpn_gt_assignments(
                                img_np_a_main, (y_true_a_main,),
                                fpn_level_names=[FPN_LEVEL_NAME_DEBUG_SDL_G],
                                fpn_configs={FPN_LEVEL_NAME_DEBUG_SDL_G: SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G},
                                classes_list=CLASSES_LIST_SDL_G,
                                title_prefix=f"GT (Single Level) for {os.path.basename(img_path_main_loop)} [AUGMENTED]"
                            )
                            if MATPLOTLIB_AVAILABLE_MAIN_TEST_SDL: plt_main_test_sdl.close('all')
                    except Exception as e_aug_test_main_loop:
                        print(
                            f"    ОШИБКА при обработке аугментированной '{os.path.basename(img_path_main_loop)}': {e_aug_test_main_loop}")
                        import traceback;

                        traceback.print_exc()
                elif AUGMENTATION_FUNC_AVAILABLE_SDL:
                    print(
                        f"  --- Аугментация пропущена (флаг USE_AUGMENTATION_SDL_LOADER_FLAG_G в False или --force_augmentation_test не указан) ---")
                else:
                    print(f"  --- Аугментация пропущена (функция аугментации не доступна) ---")
    print("\n--- Тестирование detector_data_loader_single_level_debug.py завершено ---")