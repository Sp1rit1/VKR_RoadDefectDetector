# src/datasets/detector_data_loader.py
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
import sys
import random

# --- Импорт Аугментаций ---
try:
    if __name__ == '__main__' and __package__ is None:
        _parent_dir_for_direct_run = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if _parent_dir_for_direct_run not in sys.path:
            sys.path.insert(0, _parent_dir_for_direct_run)
        from datasets.augmentations import get_detector_train_augmentations
    else:
        from .augmentations import get_detector_train_augmentations
    AUGMENTATION_FUNC_AVAILABLE = True
    # print("INFO (detector_data_loader.py): Модуль augmentations.py успешно импортирован.")
except ImportError:
    # print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Не удалось импортировать augmentations. Аугментация будет отключена.")
    AUGMENTATION_FUNC_AVAILABLE = False


    def get_detector_train_augmentations(h, w):
        return None
except Exception as e_other_imp:
    # print(f"ПРЕДУПРЕЖДЕНИЕ (detector_data_loader.py): Другая ошибка при импорте augmentations: {e_other_imp}")
    AUGMENTATION_FUNC_AVAILABLE = False


    def get_detector_train_augmentations(h, w):
        return None

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
    CONFIG_LOAD_SUCCESS = False;
    print(f"ПРЕДУПРЕЖДЕНИЕ: Файл base_config.yaml не найден: {_base_config_path}.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS = False; print(f"ОШИБКА YAML в base_config.yaml.")
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS = False;
    print(f"ПРЕДУПРЕЖДЕНИЕ: Файл detector_config.yaml не найден: {_detector_config_path}.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS = False; print(f"ОШИБКА YAML в detector_config.yaml.")

if not CONFIG_LOAD_SUCCESS:
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки конфигов в detector_data_loader.py...")
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3]);
    DETECTOR_CONFIG.setdefault('classes', ['pit', 'crack'])
    DETECTOR_CONFIG.setdefault('fpn_anchor_configs', {
        'P3': {'anchors_wh_normalized': [[0.03, 0.03]] * 3, 'num_anchors_this_level': 3, 'stride': 8},
        'P4': {'anchors_wh_normalized': [[0.08, 0.08]] * 3, 'num_anchors_this_level': 3, 'stride': 16},
        'P5': {'anchors_wh_normalized': [[0.2, 0.2]] * 3, 'num_anchors_this_level': 3, 'stride': 32}})
    DETECTOR_CONFIG.setdefault('use_augmentation', False)
    BASE_CONFIG.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    BASE_CONFIG.setdefault('master_dataset_path', 'data/Master_Dataset_Fallback')

_input_shape_list = DETECTOR_CONFIG.get('input_shape', [416, 416, 3]);
TARGET_IMG_HEIGHT = _input_shape_list[0];
TARGET_IMG_WIDTH = _input_shape_list[1]
CLASSES_LIST_GLOBAL_FOR_DETECTOR = DETECTOR_CONFIG.get('classes', ['pit', 'crack']);
NUM_CLASSES_DETECTOR = len(CLASSES_LIST_GLOBAL_FOR_DETECTOR)
USE_AUGMENTATION_CFG = DETECTOR_CONFIG.get('use_augmentation', False) and AUGMENTATION_FUNC_AVAILABLE
FPN_ANCHOR_CONFIGS = DETECTOR_CONFIG.get('fpn_anchor_configs', {})
FPN_LEVELS_CONFIG = {}
for level_name in ["P3", "P4", "P5"]:
    default_stride = {'P3': 8, 'P4': 16, 'P5': 32}.get(level_name, 16);
    default_anchors = [[0.1, 0.1]] * 3
    level_cfg = FPN_ANCHOR_CONFIGS.get(level_name, {});
    FPN_LEVELS_CONFIG[level_name] = {
        'anchors_wh': np.array(level_cfg.get('anchors_wh_normalized', default_anchors), dtype=np.float32),
        'num_anchors': level_cfg.get('num_anchors_this_level', 3),
        'grid_h': TARGET_IMG_HEIGHT // level_cfg.get('stride', default_stride),
        'grid_w': TARGET_IMG_WIDTH // level_cfg.get('stride', default_stride),
        'stride': level_cfg.get('stride', default_stride)}
_master_dataset_path_from_cfg = BASE_CONFIG.get('master_dataset_path', 'data/Master_Dataset_Fallback')
if not os.path.isabs(_master_dataset_path_from_cfg):
    MASTER_DATASET_PATH_ABS = os.path.join(_project_root_dir, _master_dataset_path_from_cfg)
else:
    MASTER_DATASET_PATH_ABS = _master_dataset_path_from_cfg
_images_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')


# --- Конец Загрузки Конфигурации ---

def parse_xml_annotation(xml_file_path, classes_list):
    try:
        tree = ET.parse(xml_file_path);
        root = tree.getroot()
        img_fn_node = root.find('filename');
        img_fn = img_fn_node.text if img_fn_node is not None and img_fn_node.text else os.path.basename(
            xml_file_path).replace(".xml", ".jpg")
        size_node = root.find('size');
        w_xml, h_xml = None, None
        if size_node is not None:
            w_n, h_n = size_node.find('width'), size_node.find('height')
            if w_n is not None and h_n is not None and w_n.text and h_n.text:
                try:
                    w_xml, h_xml = int(w_n.text), int(h_n.text)
                except ValueError:
                    pass
                if w_xml is not None and (w_xml <= 0 or h_xml <= 0): w_xml, h_xml = None, None  # Исправлено
        objs = []
        for obj_node in root.findall('object'):
            cls_nm_node = obj_node.find('name')
            if cls_nm_node is None or cls_nm_node.text is None: continue
            cls_nm = cls_nm_node.text
            if cls_nm not in classes_list: continue
            cls_id = classes_list.index(cls_nm)
            bndb_node = obj_node.find('bndbox')
            if bndb_node is None: continue
            try:
                xmin, ymin, xmax, ymax = (float(bndb_node.find(t).text) for t in ['xmin', 'ymin', 'xmax', 'ymax'])
            except:
                continue
            if xmin >= xmax or ymin >= ymax: continue
            if w_xml and h_xml: xmin, ymin, xmax, ymax = max(0, min(xmin, w_xml)), max(0, min(ymin, h_xml)), max(0,
                                                                                                                 min(xmax,
                                                                                                                     w_xml)), max(
                0, min(ymax, h_xml))
            if xmin >= xmax or ymin >= ymax: continue
            objs.append(
                {"class_id": cls_id, "class_name": cls_nm, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        return objs, w_xml, h_xml, img_fn
    except ET.ParseError:  # e_parse
        # print(f"XML_PARSE_ERROR: {os.path.basename(xml_file_path)}: {e_parse}") # Убрано для краткости
        pass
    except Exception:  # e_generic
        # print(f"XML_OTHER_ERROR: {os.path.basename(xml_file_path)}: {e_generic}") # Убрано для краткости
        pass
    return None, None, None, None


@tf.function
def preprocess_image_and_boxes(image, boxes, target_height_tf, target_width_tf):
    oh_f = tf.cast(tf.shape(image)[0], tf.float32);
    ow_f = tf.cast(tf.shape(image)[1], tf.float32)
    img_r = tf.image.resize(image, [target_height_tf, target_width_tf]);
    img_p = img_r / 255.0
    n_box = tf.shape(boxes)[0]
    if n_box > 0:
        sow_f = tf.maximum(ow_f, 1e-6);
        soh_f = tf.maximum(oh_f, 1e-6)
        s_b_n = tf.stack([boxes[:, 0] / sow_f, boxes[:, 1] / soh_f, boxes[:, 2] / sow_f, boxes[:, 3] / soh_f], axis=-1)
        s_b_n = tf.clip_by_value(s_b_n, 0.0, 1.0)
    else:
        s_b_n = tf.zeros((0, 4), tf.float32)
    return img_p, s_b_n


def calculate_iou_numpy(box_wh, anchors_wh):
    box_wh = np.array(box_wh);
    anchors_wh = np.array(anchors_wh)
    iw = np.minimum(box_wh[0], anchors_wh[:, 0]);
    ih = np.minimum(box_wh[1], anchors_wh[:, 1])
    inter = iw * ih;
    b_area = box_wh[0] * box_wh[1];
    a_area = anchors_wh[:, 0] * anchors_wh[:, 1]
    union = b_area + a_area - inter;
    return inter / (union + 1e-6)


def assign_gt_to_fpn_levels_and_encode(gt_boxes_xywh_norm_np, gt_class_ids_list, fpn_levels_config_dict_py):
    y_true_all_levels = {};
    anchor_assigned_masks = {}
    for ln, cfg in fpn_levels_config_dict_py.items():  # Используем переданный аргумент
        y_true_all_levels[ln] = np.zeros((cfg['grid_h'], cfg['grid_w'], cfg['num_anchors'], 5 + NUM_CLASSES_DETECTOR),
                                         dtype=np.float32)
        anchor_assigned_masks[ln] = np.zeros((cfg['grid_h'], cfg['grid_w'], cfg['num_anchors']), dtype=bool)
    if gt_boxes_xywh_norm_np.shape[0] == 0: return tuple(y_true_all_levels[ln] for ln in ["P3", "P4", "P5"])

    # Эвристика для назначения GT уровням FPN (НУЖНО УЛУЧШАТЬ И КАЛИБРОВАТЬ!)
    # Пока очень грубо по площади объекта, сравнивая с площадями якорей.
    # Можно ориентироваться на абсолютные размеры объектов в пикселях и страйды FPN.
    # Например, объекты < 32x32 на P3, 32x32-96x96 на P4, >96x96 на P5 (из RetinaNet)
    # Эти значения нужно адаптировать к вашему TARGET_IMG_SIZE
    areas_p3_anchors = fpn_levels_config_dict_py["P3"]['anchors_wh'][:, 0] * fpn_levels_config_dict_py["P3"][
                                                                                 'anchors_wh'][:, 1]
    areas_p4_anchors = fpn_levels_config_dict_py["P4"]['anchors_wh'][:, 0] * fpn_levels_config_dict_py["P4"][
                                                                                 'anchors_wh'][:, 1]
    # areas_p5_anchors = fpn_levels_config_dict_py["P5"]['anchors_wh'][:,0] * fpn_levels_config_dict_py["P5"]['anchors_wh'][:,1]

    # Нормализованные площади
    mean_area_p3_norm = np.mean(areas_p3_anchors)
    mean_area_p4_norm = np.mean(areas_p4_anchors)
    # mean_area_p5_norm = np.mean(areas_p5_anchors)

    # Пороговые значения для площадей объектов (нормализованных)
    thresh_p3_p4_area = (mean_area_p3_norm + mean_area_p4_norm) / 2.0
    # thresh_p4_p5_area = (mean_area_p4_norm + mean_area_p5_norm) / 2.0
    # Это очень грубая эвристика, для улучшения см. статьи по FPN/RetinaNet

    for i_obj in range(gt_boxes_xywh_norm_np.shape[0]):
        gt_xc_n, gt_yc_n, gt_w_n, gt_h_n = gt_boxes_xywh_norm_np[i_obj];
        gt_class_id = int(gt_class_ids_list[i_obj])
        gt_area_norm = gt_w_n * gt_h_n

        assigned_level_name = "P5"  # По умолчанию - самый грубый уровень для больших
        if gt_area_norm < thresh_p3_p4_area:
            assigned_level_name = "P3"  # Самые маленькие
        elif gt_w_n < 0.35 and gt_h_n < 0.35:  # Эвристика для средних, чтобы не все на P5
            assigned_level_name = "P4"

        config = fpn_levels_config_dict_py[assigned_level_name];
        y_true_level_np = y_true_all_levels[assigned_level_name];
        anchor_mask_level = anchor_assigned_masks[assigned_level_name]
        ghl, gwl = config['grid_h'], config['grid_w'];
        anchors_whl = config['anchors_wh']
        gxc_f = gt_xc_n * float(gwl);
        gyc_f = gt_yc_n * float(ghl)
        gxi = int(gxc_f);
        gyi = int(gyc_f)
        gxi = min(gxi, gwl - 1);
        gyi = min(gyi, ghl - 1)
        best_iou = -1.0;
        best_anchor_idx = -1
        gt_box_shape_wh_iou = [gt_w_n, gt_h_n];
        ious = calculate_iou_numpy(gt_box_shape_wh_iou, anchors_whl)
        best_anchor_idx = int(np.argmax(ious));
        best_iou = float(ious[best_anchor_idx])
        if best_iou >= 0.0 and not anchor_mask_level[
            gyi, gxi, best_anchor_idx]:  # Используем порог 0.0, чтобы хотя бы один якорь был назначен, если IoU хоть какой-то есть
            anchor_mask_level[gyi, gxi, best_anchor_idx] = True;
            y_true_level_np[gyi, gxi, best_anchor_idx, 4] = 1.0
            tx = gxc_f - float(gxi);
            ty = gyc_f - float(gyi)
            anchor_w_n_val = float(anchors_whl[best_anchor_idx, 0]);
            anchor_h_n_val = float(anchors_whl[best_anchor_idx, 1])
            sgw_n = max(gt_w_n, 1e-9);
            sgh_n = max(gt_h_n, 1e-9)
            saw_n = max(anchor_w_n_val, 1e-9);
            sah_n = max(anchor_h_n_val, 1e-9)
            tw = np.log(sgw_n / saw_n);
            th = np.log(sgh_n / sah_n)
            y_true_level_np[gyi, gxi, best_anchor_idx, 0:4] = [tx, ty, tw, th]
            y_true_level_np[gyi, gxi, best_anchor_idx, 5 + gt_class_id] = 1.0
    return tuple(y_true_all_levels[ln] for ln in ["P3", "P4", "P5"])


def load_and_prepare_detector_fpn_py_func(image_path_tensor, xml_path_tensor,
                                          py_target_height, py_target_width,
                                          # py_fpn_levels_config теперь будет браться из глобальной FPN_LEVELS_CONFIG
                                          py_apply_augmentation_arg):
    image_path = image_path_tensor.numpy().decode('utf-8');
    xml_path = xml_path_tensor.numpy().decode('utf-8')
    py_apply_aug = bool(py_apply_augmentation_arg.numpy()) if hasattr(py_apply_augmentation_arg, 'numpy') else bool(
        py_apply_augmentation_arg)
    y_true_fallbacks = [];
    img_fallback_shape = (py_target_height, py_target_width, 3)
    for ln_fallback in ["P3", "P4", "P5"]:
        cfg_fallback = FPN_LEVELS_CONFIG[ln_fallback];
        y_true_fallbacks.append(np.zeros(
            (cfg_fallback['grid_h'], cfg_fallback['grid_w'], cfg_fallback['num_anchors'], 5 + NUM_CLASSES_DETECTOR),
            dtype=np.float32))
    try:
        from PIL import Image as PILImage;
        pil_img = PILImage.open(image_path).convert('RGB');
        img_np_orig_u8 = np.array(pil_img, dtype=np.uint8)
    except:
        return np.zeros(img_fallback_shape, dtype=np.float32), y_true_fallbacks[0], y_true_fallbacks[1], \
        y_true_fallbacks[2]

    objs, _, _, _ = parse_xml_annotation(xml_path, CLASSES_LIST_GLOBAL_FOR_DETECTOR)
    if objs is None: return np.zeros(img_fallback_shape, dtype=np.float32), y_true_fallbacks[0], y_true_fallbacks[1], \
    y_true_fallbacks[2]

    boxes_px_orig = [[o['xmin'], o['ymin'], o['xmax'], o['ymax']] for o in objs] if objs else []
    cls_ids_orig = [o['class_id'] for o in objs] if objs else []
    curr_img_np = img_np_orig_u8;
    curr_boxes_px = boxes_px_orig;
    curr_cls_ids = cls_ids_orig
    if py_apply_aug and AUGMENTATION_FUNC_AVAILABLE and objs:
        try:
            augger = get_detector_train_augmentations(py_target_height,
                                                      py_target_width)  # Используем глобальные TARGET_IMG_H/W
            augd = augger(image=curr_img_np, bboxes=curr_boxes_px,
                          class_labels_for_albumentations=curr_cls_ids)  # class_labels_for_albumentations - это новое имя поля
            curr_img_np = augd['image'];
            curr_boxes_px = augd['bboxes'];
            curr_cls_ids = augd['class_labels_for_albumentations']
            if not curr_boxes_px: objs = []
        except:
            pass

    img_tensor_py = tf.convert_to_tensor(curr_img_np.astype(np.float32), dtype=tf.float32)
    boxes_tensor_px_py = tf.constant(curr_boxes_px, dtype=tf.float32) if curr_boxes_px else tf.zeros((0, 4),
                                                                                                     dtype=tf.float32)
    img_proc_tensor, scaled_gt_boxes_norm_t = preprocess_image_and_boxes(img_tensor_py, boxes_tensor_px_py,
                                                                         tf.constant(py_target_height, dtype=tf.int32),
                                                                         tf.constant(py_target_width, dtype=tf.int32))

    gt_boxes_xywh_norm_l = []
    if objs and tf.shape(scaled_gt_boxes_norm_t)[0] > 0 and len(curr_cls_ids) == tf.shape(scaled_gt_boxes_norm_t)[
        0]:  # Убедимся, что длины совпадают
        for i_s in range(scaled_gt_boxes_norm_t.shape[0]):
            bn = scaled_gt_boxes_norm_t[i_s].numpy();
            w_n = float(bn[2]) - float(bn[0]);
            h_n = float(bn[3]) - float(bn[1])
            xc_n = float(bn[0]) + w_n / 2.0;
            yc_n = float(bn[1]) + h_n / 2.0;
            gt_boxes_xywh_norm_l.append([xc_n, yc_n, w_n, h_n])
    gt_boxes_xywh_norm_np_assign = np.array(gt_boxes_xywh_norm_l,
                                            dtype=np.float32) if gt_boxes_xywh_norm_l else np.zeros((0, 4),
                                                                                                    dtype=np.float32)

    # Передаем ГЛОБАЛЬНЫЙ FPN_LEVELS_CONFIG
    y_true_p3, y_true_p4, y_true_p5 = assign_gt_to_fpn_levels_and_encode(gt_boxes_xywh_norm_np_assign, curr_cls_ids,
                                                                         FPN_LEVELS_CONFIG)
    return img_proc_tensor.numpy(), y_true_p3, y_true_p4, y_true_p5  # Возвращаем плоский кортеж


def load_and_prepare_detector_fpn_tf_wrapper(image_path_tensor, xml_path_tensor, augment_tensor):
    img_processed_np, y_true_p3_np, y_true_p4_np, y_true_p5_np = tf.py_function(
        func=load_and_prepare_detector_fpn_py_func,
        inp=[image_path_tensor, xml_path_tensor,
             TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH,  # py_target_height, py_target_width
             # FPN_LEVELS_CONFIG, # Убрали из inp, он будет браться из глобальной области видимости Python функции
             augment_tensor],  # py_apply_augmentation_arg
        Tout=[tf.float32, tf.float32, tf.float32, tf.float32]  # Изображение + 3 y_true тензора
    )
    img_processed_np.set_shape([TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3])
    y_true_fpn_tuple = (y_true_p3_np, y_true_p4_np, y_true_p5_np)
    for level_name, y_true_tensor_level in zip(["P3", "P4", "P5"], y_true_fpn_tuple):
        cfg = FPN_LEVELS_CONFIG[level_name]
        shape = (cfg['grid_h'], cfg['grid_w'], cfg['num_anchors'], 5 + NUM_CLASSES_DETECTOR)
        y_true_tensor_level.set_shape(shape)
    return img_processed_np, y_true_fpn_tuple


def create_detector_tf_dataset(image_paths_list, xml_paths_list, batch_size, shuffle=True, augment=False):
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(xml_paths_list,
                                                                         (list, tuple)): raise ValueError(
        "paths must be lists/tuples.")
    if len(image_paths_list) != len(xml_paths_list): raise ValueError("Image and XML path counts differ.")
    aug_flags = tf.constant([augment] * len(image_paths_list), dtype=tf.bool)
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(image_paths_list, dtype=tf.string), tf.constant(xml_paths_list, dtype=tf.string), aug_flags))
    if shuffle: dataset = dataset.shuffle(buffer_size=len(image_paths_list), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda img_p, xml_p, aug_f: load_and_prepare_detector_fpn_tf_wrapper(img_p, xml_p, aug_f),
                          num_parallel_calls=tf.data.AUTOTUNE)
    # Для FPN очень важно, чтобы все элементы в батче имели одинаковую структуру y_true (кортеж из 3х тензоров с фиксированными формами)
    # drop_remainder=True может быть безопаснее, если есть риск неполных последних батчей, которые могут вызвать проблемы с формами
    # но если все y_true тензоры всегда имеют фиксированную форму, то False тоже должен работать.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    import sys
    import random  # Убедись, что random импортирован в начале файла

    _utils_path = os.path.abspath(os.path.join(_current_script_dir, '..', 'utils'))
    if _utils_path not in sys.path:
        sys.path.insert(0, _utils_path)
    try:
        from plot_utils import visualize_fpn_data_sample

        VISUALIZATION_ENABLED = True
        print("INFO (detector_data_loader.py __main__): plot_utils.visualize_fpn_data_sample успешно импортирован.")
    except ImportError:
        VISUALIZATION_ENABLED = False
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Модуль plot_utils или функция visualize_fpn_data_sample не найдены. Визуализация будет отключена.")
    except Exception as e_imp_plot:
        VISUALIZATION_ENABLED = False
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при импорте plot_utils: {e_imp_plot}. Визуализация будет отключена.")

    if not CONFIG_LOAD_SUCCESS:
        print("\n!!! ВНИМАНИЕ: Конфигурационные файлы не были загружены корректно. "
              "Тестирование может использовать неактуальные или дефолтные параметры.")

    print(f"\n--- Тестирование detector_data_loader.py (FPN) ---")
    print(f"  Глобальный флаг USE_AUGMENTATION_CFG из конфига: {USE_AUGMENTATION_CFG}")
    print(f"  AUGMENTATION_FUNC_AVAILABLE: {AUGMENTATION_FUNC_AVAILABLE}")
    print(f"Параметры FPN уровней из конфига:")
    for level, cfg_test in FPN_LEVELS_CONFIG.items():  # Используем FPN_LEVELS_CONFIG, определенный в модуле
        print(
            f"  Ур.{level}: Сетка({cfg_test['grid_h']}x{cfg_test['grid_w']}), Якорей {cfg_test['num_anchors']}, Страйд {cfg_test['stride']}")

    # Пути к данным (берем из TRAIN сета Detector_Dataset_Ready)
    _detector_dataset_ready_path_rel_test = "data/Detector_Dataset_Ready"  # Как определено в create_data_splits.py
    DETECTOR_DATASET_READY_ABS_TEST = os.path.join(_project_root_dir, _detector_dataset_ready_path_rel_test)

    # Используем _images_subdir_name_cfg и _annotations_subdir_name_cfg, определенные в начале файла
    IMAGES_DIR_FOR_TEST_MAIN = os.path.join(DETECTOR_DATASET_READY_ABS_TEST, "train", _images_subdir_name_cfg)
    ANNOTATIONS_DIR_FOR_TEST_MAIN = os.path.join(DETECTOR_DATASET_READY_ABS_TEST, "train", _annotations_subdir_name_cfg)

    print(f"\nТестовые пути (сканируем Detector_Dataset_Ready/train/):")
    print(f"  Изображения из: {IMAGES_DIR_FOR_TEST_MAIN}")
    print(f"  Аннотации из: {ANNOTATIONS_DIR_FOR_TEST_MAIN}")

    example_image_paths_main = []
    example_xml_paths_main = []

    if not os.path.isdir(IMAGES_DIR_FOR_TEST_MAIN) or not os.path.isdir(ANNOTATIONS_DIR_FOR_TEST_MAIN):
        print(f"ОШИБКА: Директории data/Detector_Dataset_Ready/train/ (JPEGImages или Annotations) не найдены.")
        print("Пожалуйста, убедитесь, что скрипт 'create_data_splits.py' был успешно запущен.")
    else:
        valid_extensions = ['.jpg', '.jpeg', '.png']
        all_image_files_in_train_dir_main = []
        for ext_main in valid_extensions:
            all_image_files_in_train_dir_main.extend(
                glob.glob(os.path.join(IMAGES_DIR_FOR_TEST_MAIN, f"*{ext_main.lower()}")))
            all_image_files_in_train_dir_main.extend(
                glob.glob(os.path.join(IMAGES_DIR_FOR_TEST_MAIN, f"*{ext_main.upper()}")))

        all_image_files_in_train_dir_main = sorted(list(set(all_image_files_in_train_dir_main)))

        for img_path_abs_str_main in all_image_files_in_train_dir_main:
            base_name_main, _ = os.path.splitext(os.path.basename(img_path_abs_str_main))
            xml_file_abs_str_main = os.path.join(ANNOTATIONS_DIR_FOR_TEST_MAIN, base_name_main + ".xml")

            if os.path.exists(xml_file_abs_str_main):
                example_image_paths_main.append(img_path_abs_str_main)
                example_xml_paths_main.append(xml_file_abs_str_main)

        if not example_image_paths_main:
            print("\nНе найдено совпадающих пар изображение/аннотация в data/Detector_Dataset_Ready/train/.")
        else:
            print(
                f"\nВсего найдено {len(example_image_paths_main)} пар изображение/аннотация в data/Detector_Dataset_Ready/train/.")

            num_examples_to_show = min(len(example_image_paths_main), 10)  # Показываем до 3х примеров

            if num_examples_to_show == 0:
                print("Нет файлов для теста.")
            else:
                paired_files_main = list(zip(example_image_paths_main, example_xml_paths_main))
                random.shuffle(paired_files_main)  # Перемешиваем один раз
                selected_pairs_main = paired_files_main[:num_examples_to_show]

                print(
                    f"\nБудет протестировано и визуализировано {len(selected_pairs_main)} случайных файлов из TRAIN выборки:")

                for p_idx_main, (p_img_path_main, p_xml_path_main) in enumerate(selected_pairs_main):
                    print(f"\n--- Пример {p_idx_main + 1}: {os.path.basename(p_img_path_main)} ---")

                    current_test_batch_size_main = 1  # Всегда 1 для детального просмотра

                    # --- 1. Обработка ОРИГИНАЛА (без аугментации) ---
                    print(f"  --- Загрузка ОРИГИНАЛА ---")
                    dataset_no_aug_main = create_detector_tf_dataset(
                        [p_img_path_main], [p_xml_path_main],  # Список из одного элемента
                        batch_size=current_test_batch_size_main,
                        shuffle=False,
                        augment=False  # Явно отключаем аугментацию
                    )
                    try:
                        for images_batch_no_aug, y_true_fpn_output_no_aug in dataset_no_aug_main.take(1):
                            y_true_p3_b_no_aug, y_true_p4_b_no_aug, y_true_p5_b_no_aug = y_true_fpn_output_no_aug
                            print("    Img shape (оригинал):", images_batch_no_aug.shape)
                            print("    y_true P3 (оригинал):", y_true_p3_b_no_aug.shape)
                            # ... (можно добавить вывод информации об ответственных якорях, как раньше)
                            if VISUALIZATION_ENABLED:
                                visualize_fpn_data_sample(
                                    images_batch_no_aug[0].numpy(),
                                    (y_true_p3_b_no_aug[0].numpy(), y_true_p4_b_no_aug[0].numpy(),
                                     y_true_p5_b_no_aug[0].numpy()),
                                    FPN_LEVELS_CONFIG, CLASSES_LIST_GLOBAL_FOR_DETECTOR,
                                    TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT,
                                    title=f"GT for {os.path.basename(p_img_path_main)} (Original)"
                                )
                    except Exception as e_no_aug:
                        print(f"    ОШИБКА при обработке оригинала: {e_no_aug}")
                        import traceback;

                        traceback.print_exc()

                    # --- 2. Обработка С АУГМЕНТАЦИЕЙ (если включена глобально) ---
                    if USE_AUGMENTATION_CFG and AUGMENTATION_FUNC_AVAILABLE:
                        print(
                            f"  --- Загрузка АУГМЕНТИРОВАННОЙ версии (USE_AUGMENTATION_CFG={USE_AUGMENTATION_CFG}) ---")
                        dataset_with_aug_main = create_detector_tf_dataset(
                            [p_img_path_main], [p_xml_path_main],  # Список из одного элемента
                            batch_size=current_test_batch_size_main,
                            shuffle=False,  # Для одного файла нет смысла перемешивать
                            augment=True  # ЯВНО включаем аугментацию
                        )
                        try:
                            for images_batch_aug, y_true_fpn_output_aug in dataset_with_aug_main.take(1):
                                y_true_p3_b_aug, y_true_p4_b_aug, y_true_p5_b_aug = y_true_fpn_output_aug
                                print("    Img shape (аугм.):", images_batch_aug.shape)
                                print("    y_true P3 (аугм.):", y_true_p3_b_aug.shape)
                                # ... (можно добавить вывод информации об ответственных якорях)
                                if VISUALIZATION_ENABLED:
                                    visualize_fpn_data_sample(
                                        images_batch_aug[0].numpy(),
                                        (y_true_p3_b_aug[0].numpy(), y_true_p4_b_aug[0].numpy(),
                                         y_true_p5_b_aug[0].numpy()),
                                        FPN_LEVELS_CONFIG, CLASSES_LIST_GLOBAL_FOR_DETECTOR,
                                        TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT,
                                        title=f"GT for {os.path.basename(p_img_path_main)} (AUGMENTED)"
                                    )
                        except Exception as e_aug:
                            print(f"    ОШИБКА при обработке аугментированной версии: {e_aug}")
                            import traceback;

                            traceback.print_exc()
                    elif not AUGMENTATION_FUNC_AVAILABLE:
                        print(f"  --- Аугментация пропущена (AUGMENTATION_FUNC_AVAILABLE = False) ---")
                    else:  # USE_AUGMENTATION_CFG is False
                        print(f"  --- Аугментация пропущена (USE_AUGMENTATION_CFG = False в конфиге) ---")

    print("\n--- Тестирование detector_data_loader.py (FPN) завершено ---")