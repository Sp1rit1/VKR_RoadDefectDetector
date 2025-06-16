# RoadDefectDetector/evaluate_detector.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import glob
from pathlib import Path
import time  # Хотя здесь не так критично, как в predict
import argparse

# --- Добавляем src в sys.path ---
_project_root_eval = Path(__file__).resolve().parent
_src_path_eval = _project_root_eval / 'src'
import sys

if str(_src_path_eval) not in sys.path:
    sys.path.insert(0, str(_src_path_eval))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_EVAL = {}
try:
    from losses.detection_losses import compute_detector_loss_v2_fpn, compute_detector_loss_v1

    CUSTOM_OBJECTS_EVAL['compute_detector_loss_v2_fpn'] = compute_detector_loss_v2_fpn
    CUSTOM_OBJECTS_EVAL['compute_detector_loss_v1'] = compute_detector_loss_v1  # На случай старой модели
    print("INFO (evaluate_detector.py): Кастомные функции потерь ЗАГРУЖЕНЫ.")
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Одна или несколько кастомных функций потерь не найдены.")
except Exception as e_gen_loss:
    print(f"ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Общая ошибка при импорте функций потерь: {e_gen_loss}.")

try:
    from datasets.detector_data_loader import parse_xml_annotation as parse_xml_annotation_detector

    print("INFO (evaluate_detector.py): parse_xml_annotation успешно импортирована.")
except ImportError as e_parse_xml:
    print(f"ОШИБКА: Не удалось импортировать parse_xml_annotation из detector_data_loader: {e_parse_xml}")


    def parse_xml_annotation_detector(xml_path, classes_list):
        return None, None, None, None  # Заглушка


# --- Загрузка Конфигураций ---
def load_config_eval_strict(config_path_obj, config_name_for_log):
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(f"ОШИБКА: {config_path_obj.name} пуст или имеет неверный формат для '{config_name_for_log}'. Выход.")
            exit()
        print(f"INFO: Конфиг '{config_name_for_log}' ({config_path_obj.name}) успешно загружен.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден: {config_path_obj}. Выход.");
        exit()
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML в {config_path_obj.name}: {e}. Выход.");
        exit()


_base_config_path_obj_eval = _src_path_eval / 'configs' / 'base_config.yaml'
_detector_arch_config_path_obj_eval = _src_path_eval / 'configs' / 'detector_config.yaml'  # Основной конфиг архитектуры
_predict_config_path_obj_eval = _src_path_eval / 'configs' / 'predict_config.yaml'  # Для порогов по умолчанию

print("--- Загрузка конфигурационных файлов для evaluate_detector.py ---")
BASE_CONFIG_EVAL = load_config_eval_strict(_base_config_path_obj_eval, "Base Config")
DETECTOR_ARCH_CONFIG_EVAL = load_config_eval_strict(_detector_arch_config_path_obj_eval, "Detector Architecture Config")
PREDICT_CONFIG_EVAL = load_config_eval_strict(_predict_config_path_obj_eval, "Predict Config")

# --- Параметры из Конфигов ---
# Общие параметры детектора из predict_config (т.к. evaluate использует predict-like флоу)
DET_INPUT_SHAPE_EVAL = tuple(PREDICT_CONFIG_EVAL.get('detector_input_shape', [416, 416, 3]))
DET_TARGET_IMG_HEIGHT_EVAL, DET_TARGET_IMG_WIDTH_EVAL = DET_INPUT_SHAPE_EVAL[0], DET_INPUT_SHAPE_EVAL[1]
DET_CLASSES_LIST_EVAL = PREDICT_CONFIG_EVAL.get('detector_class_names', ['pit', 'crack'])
DET_NUM_CLASSES_EVAL = len(DET_CLASSES_LIST_EVAL)

# Определяем тип детектора на основе того, какая модель передана (или из predict_config)
# Для evaluate.py лучше всего, если тип модели определяется явно или по имени файла модели.
# Пока будем полагаться на detector_type из predict_config.yaml
DETECTOR_TYPE_EVAL = PREDICT_CONFIG_EVAL.get("detector_type", "fpn").lower()
print(f"INFO (evaluate_detector.py): Определен тип детектора как '{DETECTOR_TYPE_EVAL}' из predict_config.yaml.")

# Загрузка специфичных для архитектуры параметров из DETECTOR_ARCH_CONFIG_EVAL
if DETECTOR_TYPE_EVAL == "fpn":
    fpn_params_eval = DETECTOR_ARCH_CONFIG_EVAL.get('fpn_detector_params', {})
    if not fpn_params_eval: print(f"ОШИБКА: Секция 'fpn_detector_params' не найдена в detector_config.yaml."); exit()
    DET_FPN_LEVELS_EVAL = fpn_params_eval.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
    DET_FPN_STRIDES_EVAL = fpn_params_eval.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})
    DET_FPN_ANCHOR_CONFIGS_EVAL = fpn_params_eval.get('detector_fpn_anchor_configs', {})
    if not DET_FPN_ANCHOR_CONFIGS_EVAL or not all(lvl in DET_FPN_ANCHOR_CONFIGS_EVAL for lvl in DET_FPN_LEVELS_EVAL):
        print(f"ОШИБКА: Конфигурация якорей для FPN неполна в detector_config.yaml -> fpn_detector_params");
        exit()
elif DETECTOR_TYPE_EVAL == "single_level":
    sl_params_eval = DETECTOR_ARCH_CONFIG_EVAL.get('single_level_detector_params', {})
    if not sl_params_eval: print(
        f"ОШИБКА: Секция 'single_level_detector_params' не найдена в detector_config.yaml."); exit()
    SINGLE_ANCHORS_WH_NORM_LIST_EVAL = sl_params_eval.get('anchors_wh_normalized',
                                                          [[0.1, 0.1]] * sl_params_eval.get('num_anchors_per_location',
                                                                                            3))
    SINGLE_ANCHORS_WH_NORM_EVAL = np.array(SINGLE_ANCHORS_WH_NORM_LIST_EVAL, dtype=np.float32)
    SINGLE_NUM_ANCHORS_EVAL = SINGLE_ANCHORS_WH_NORM_EVAL.shape[0]
    SINGLE_NETWORK_STRIDE_EVAL = sl_params_eval.get('network_stride', 16)
    SINGLE_GRID_HEIGHT_EVAL = DET_TARGET_IMG_HEIGHT_EVAL // SINGLE_NETWORK_STRIDE_EVAL
    SINGLE_GRID_WIDTH_EVAL = DET_TARGET_IMG_WIDTH_EVAL // SINGLE_NETWORK_STRIDE_EVAL
else:
    print(f"ОШИБКА: Неизвестный detector_type: '{DETECTOR_TYPE_EVAL}'.");
    exit()

_images_subdir_name_eval = BASE_CONFIG_EVAL.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_eval = BASE_CONFIG_EVAL.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel_eval = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_EVAL = (_project_root_eval / _detector_dataset_ready_path_rel_eval).resolve()
VAL_IMAGE_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _images_subdir_name_eval)
VAL_ANNOT_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _annotations_subdir_name_eval)


# --- Вспомогательные Функции (копируем из predict_pipeline.py) ---
# preprocess_image_for_model_tf, decode_single_level_predictions_generic,
# apply_nms_and_filter_generic, calculate_iou_for_eval_np
# (Вставь сюда полные версии этих функций, они идентичны тем, что в последней версии run_prediction_pipeline.py)
def preprocess_image_for_eval(image_bgr, target_height, target_width):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = tf.cast(image_resized, tf.float32) / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_single_level_predictions_generic(raw_level_preds, level_anchors_wh_norm, level_grid_h, level_grid_w,
                                            num_classes):
    batch_size = tf.shape(raw_level_preds)[0]
    num_anchors_this_level = tf.shape(level_anchors_wh_norm)[0]
    pred_xy_raw = raw_level_preds[..., 0:2];
    pred_wh_raw = raw_level_preds[..., 2:4]
    pred_obj_logit = raw_level_preds[..., 4:5];
    pred_class_logits = raw_level_preds[..., 5:]
    gy = tf.tile(tf.range(level_grid_h, dtype=tf.float32)[:, tf.newaxis], [1, level_grid_w])
    gx = tf.tile(tf.range(level_grid_w, dtype=tf.float32)[tf.newaxis, :], [level_grid_h, 1])
    grid_xy = tf.stack([gx, gy], axis=-1)[tf.newaxis, :, :, tf.newaxis, :]
    grid_xy = tf.tile(grid_xy, [batch_size, 1, 1, num_anchors_this_level, 1])
    pred_xy_norm = (tf.sigmoid(pred_xy_raw) + grid_xy) / tf.constant([level_grid_w, level_grid_h], dtype=tf.float32)
    anchors_t = tf.constant(level_anchors_wh_norm, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pred_wh_norm = tf.exp(pred_wh_raw) * anchors_t
    decoded_boxes_xywh_n = tf.concat([pred_xy_norm, pred_wh_norm], axis=-1)
    obj_conf = tf.sigmoid(pred_obj_logit)
    class_probs = tf.sigmoid(pred_class_logits)
    return decoded_boxes_xywh_n, obj_conf, class_probs


def apply_nms_and_filter_generic(all_decoded_boxes_xywh_norm_list, all_obj_confidence_list, all_class_probs_list,
                                 num_classes_detector, confidence_threshold, iou_threshold, max_detections):
    batch_size = tf.shape(all_decoded_boxes_xywh_norm_list[0])[0]
    flat_boxes_xywh, flat_obj_conf, flat_class_probs = [], [], []
    for i in range(len(all_decoded_boxes_xywh_norm_list)):
        num_total_boxes_level = tf.reduce_prod(tf.shape(all_decoded_boxes_xywh_norm_list[i])[1:-1])
        flat_boxes_xywh.append(tf.reshape(all_decoded_boxes_xywh_norm_list[i], [batch_size, num_total_boxes_level, 4]))
        flat_obj_conf.append(tf.reshape(all_obj_confidence_list[i], [batch_size, num_total_boxes_level, 1]))
        flat_class_probs.append(
            tf.reshape(all_class_probs_list[i], [batch_size, num_total_boxes_level, num_classes_detector]))
    boxes_combined_xywh = tf.concat(flat_boxes_xywh, axis=1)
    obj_conf_combined = tf.concat(flat_obj_conf, axis=1)
    class_probs_combined = tf.concat(flat_class_probs, axis=1)
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        boxes_combined_xywh[..., 1:2] - boxes_combined_xywh[..., 3:4] / 2.0,
        boxes_combined_xywh[..., 0:1] - boxes_combined_xywh[..., 2:3] / 2.0,
        boxes_combined_xywh[..., 1:2] + boxes_combined_xywh[..., 3:4] / 2.0,
        boxes_combined_xywh[..., 0:1] + boxes_combined_xywh[..., 2:3] / 2.0
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax, 0.0, 1.0)
    final_scores_per_class = obj_conf_combined * class_probs_combined
    boxes_for_nms = tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2)
    max_out_per_cls = max_detections // num_classes_detector if num_classes_detector > 0 else max_detections
    if max_out_per_cls == 0: max_out_per_cls = 1
    return tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms, scores=final_scores_per_class,
        max_output_size_per_class=max_out_per_cls, max_total_size=max_detections,
        iou_threshold=iou_threshold, score_threshold=confidence_threshold, clip_boxes=False)


def calculate_iou_for_eval_np(box1_xyxy_px, box2_xyxy_px):
    x1_inter = max(box1_xyxy_px[0], box2_xyxy_px[0]);
    y1_inter = max(box1_xyxy_px[1], box2_xyxy_px[1])
    x2_inter = min(box1_xyxy_px[2], box2_xyxy_px[2]);
    y2_inter = min(box1_xyxy_px[3], box2_xyxy_px[3])
    inter_width = max(0, x2_inter - x1_inter);
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = float(inter_width * inter_height)
    box1_area = float((box1_xyxy_px[2] - box1_xyxy_px[0]) * (box1_xyxy_px[3] - box1_xyxy_px[1]))
    box2_area = float((box2_xyxy_px[2] - box2_xyxy_px[0]) * (box2_xyxy_px[3] - box2_xyxy_px[1]))
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / (union_area + 1e-7)
    return iou


# --- Основная Функция Оценки ---
def evaluate_detector_on_validation_set(model_path_arg, conf_thresh_arg, iou_thresh_nms_arg, iou_thresh_match_arg,
                                        max_dets_arg_from_cmd):
    # ... (Загрузка модели как была) ...
    model_full_path = (_project_root_eval / model_path_arg).resolve()
    print(f"\n--- Оценка Модели Детектора ---");
    print(f"Загрузка модели из: {model_full_path}")
    if not model_full_path.exists(): print(f"ОШИБКА: Файл модели не найден: {model_full_path}"); return
    try:
        model = tf.keras.models.load_model(str(model_full_path), custom_objects=CUSTOM_OBJECTS_EVAL, compile=False)
        print("Модель детектора успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели детектора: {e}"); return

    # ... (Сбор val_image_paths как был) ...
    if not Path(VAL_IMAGE_DIR_EVAL).is_dir() or not Path(VAL_ANNOT_DIR_EVAL).is_dir(): print(
        f"ОШИБКА: Директории валидационных данных не найдены."); return
    val_image_paths = sorted(list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpg")) + \
                             list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpeg")) + \
                             list(Path(VAL_IMAGE_DIR_EVAL).glob("*.png")))
    if not val_image_paths: print(f"Изображения не найдены в: {VAL_IMAGE_DIR_EVAL}"); return
    print(f"\nНайдено {len(val_image_paths)} изображений для оценки.")
    print(
        f"Параметры оценки: conf_thresh={conf_thresh_arg}, iou_nms={iou_thresh_nms_arg}, iou_match={iou_thresh_match_arg}, max_dets={max_dets_arg_from_cmd}")

    stats_per_class = {cls_name: [0, 0, 0] for cls_name in DET_CLASSES_LIST_EVAL}
    total_gt_objects_per_class = {cls_name: 0 for cls_name in DET_CLASSES_LIST_EVAL}

    for img_idx, image_path_obj in enumerate(val_image_paths):
        # ... (Логика чтения изображения и GT XML как была) ...
        if (img_idx + 1) % 50 == 0 or img_idx == 0 or (img_idx + 1) == len(val_image_paths): print(
            f"  Обработка {img_idx + 1}/{len(val_image_paths)}: {image_path_obj.name}")
        original_bgr_image = cv2.imread(str(image_path_obj));
        if original_bgr_image is None: continue
        original_h, original_w = original_bgr_image.shape[:2]
        xml_path_str = str(Path(VAL_ANNOT_DIR_EVAL) / (image_path_obj.stem + ".xml"))
        gt_objects_data, _, _, _ = parse_xml_annotation_detector(xml_path_str, DET_CLASSES_LIST_EVAL)
        if gt_objects_data is None: gt_objects_data = []
        gt_boxes_for_matching = []
        for gt_obj in gt_objects_data:
            class_id = int(gt_obj['class_id'])
            if 0 <= class_id < len(DET_CLASSES_LIST_EVAL):
                class_name = DET_CLASSES_LIST_EVAL[class_id];
                total_gt_objects_per_class[class_name] += 1
                gt_boxes_for_matching.append(
                    [int(gt_obj['xmin']), int(gt_obj['ymin']), int(gt_obj['xmax']), int(gt_obj['ymax']), class_id,
                     False])

        detector_input_batch = preprocess_image_for_eval(original_bgr_image, DET_TARGET_IMG_HEIGHT_EVAL,
                                                         DET_TARGET_IMG_WIDTH_EVAL)
        raw_detector_output = model.predict(detector_input_batch, verbose=0)

        all_level_boxes_xywh, all_level_obj_conf, all_level_class_probs = [], [], []

        if DETECTOR_TYPE_EVAL == "fpn":
            raw_detector_predictions_list = raw_detector_output  # Это уже список
            for i_lvl, level_key in enumerate(DET_FPN_LEVELS_EVAL):
                raw_level_preds = raw_detector_predictions_list[i_lvl]
                level_cfg = DET_FPN_ANCHOR_CONFIGS_EVAL.get(level_key)
                if level_cfg is None: print(f"ОШИБКА: Нет конфига якорей для FPN уровня {level_key}"); continue
                level_anchors_wh = np.array(level_cfg['anchors_wh_normalized'], dtype=np.float32)
                level_grid_h = DET_TARGET_IMG_HEIGHT_EVAL // DET_FPN_STRIDES_EVAL[level_key]
                level_grid_w = DET_TARGET_IMG_WIDTH_EVAL // DET_FPN_STRIDES_EVAL[level_key]
                decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                    raw_level_preds, level_anchors_wh, level_grid_h, level_grid_w, DET_NUM_CLASSES_EVAL)
                all_level_boxes_xywh.append(decoded_boxes);
                all_level_obj_conf.append(obj_conf);
                all_level_class_probs.append(class_probs)

        elif DETECTOR_TYPE_EVAL == "single_level":
            decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                raw_detector_output, SINGLE_ANCHORS_WH_NORM_EVAL, SINGLE_GRID_HEIGHT_EVAL, SINGLE_GRID_WIDTH_EVAL,
                DET_NUM_CLASSES_EVAL)
            all_level_boxes_xywh.append(decoded_boxes);
            all_level_obj_conf.append(obj_conf);
            all_level_class_probs.append(class_probs)

        if not all_level_boxes_xywh:
            num_detections = 0; pred_boxes_norm_yminxminymaxxmax = np.array([]); pred_scores = np.array(
                []); pred_class_ids = np.array([])
        else:
            nms_boxes_n, nms_scores_n, nms_classes_n, num_valid_dets_n = apply_nms_and_filter_generic(
                all_level_boxes_xywh, all_level_obj_conf, all_level_class_probs,
                DET_NUM_CLASSES_EVAL, conf_thresh_arg, iou_thresh_nms_arg, max_dets_arg_from_cmd)
            num_detections = int(num_valid_dets_n[0].numpy())
            pred_boxes_norm_yminxminymaxxmax = nms_boxes_n[0][:num_detections].numpy()
            pred_scores = nms_scores_n[0][:num_detections].numpy()
            pred_class_ids = nms_classes_n[0][:num_detections].numpy().astype(int)

        # ... (Логика сопоставления TP/FP как была) ...
        if num_detections > 0:
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes_norm_sorted = pred_boxes_norm_yminxminymaxxmax[sorted_indices]
            pred_class_ids_sorted = pred_class_ids[sorted_indices]
            for i_pred in range(len(pred_boxes_norm_sorted)):
                pred_box_norm = pred_boxes_norm_sorted[i_pred]
                pred_box_px = [int(pred_box_norm[1] * original_w), int(pred_box_norm[0] * original_h),
                               int(pred_box_norm[3] * original_w), int(pred_box_norm[2] * original_h)]
                pred_cls_id = pred_class_ids_sorted[i_pred]
                if not (0 <= pred_cls_id < len(DET_CLASSES_LIST_EVAL)): continue
                pred_cls_name = DET_CLASSES_LIST_EVAL[pred_cls_id];
                best_iou_for_pred = 0.0;
                best_gt_match_idx = -1
                for i_gt, gt_data in enumerate(gt_boxes_for_matching):
                    gt_box_px = gt_data[0:4];
                    gt_cls_id = gt_data[4];
                    gt_matched_flag = gt_data[5]
                    if gt_cls_id == pred_cls_id and not gt_matched_flag:
                        iou = calculate_iou_for_eval_np(pred_box_px, gt_box_px)
                        if iou > best_iou_for_pred: best_iou_for_pred = iou;best_gt_match_idx = i_gt
                if best_iou_for_pred >= iou_thresh_match_arg:
                    stats_per_class[pred_cls_name][0] += 1;
                    gt_boxes_for_matching[best_gt_match_idx][5] = True
                else:
                    stats_per_class[pred_cls_name][1] += 1

    # ... (Расчет и вывод финальных метрик как был) ...
    print("\n--- Результаты Оценки ---");
    avg_precision, avg_recall, avg_f1 = 0.0, 0.0, 0.0;
    valid_classes_for_avg = 0
    for class_name_key in DET_CLASSES_LIST_EVAL:
        tp = stats_per_class[class_name_key][0];
        fp = stats_per_class[class_name_key][1]
        num_gt = total_gt_objects_per_class[class_name_key];
        fn = num_gt - tp
        stats_per_class[class_name_key][2] = fn
        precision = tp / (tp + fp + 1e-7);
        recall = tp / (num_gt + 1e-7);
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        print(f"\nКласс: {class_name_key}");
        print(f"  GT: {int(num_gt)}, TP: {int(tp)}, FP: {int(fp)}, FN: {int(fn)}");
        print(f"  P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        if num_gt > 0: avg_precision += precision;avg_recall += recall;avg_f1 += f1;valid_classes_for_avg += 1
    if valid_classes_for_avg > 0:
        macro_precision = avg_precision / valid_classes_for_avg;
        macro_recall = avg_recall / valid_classes_for_avg;
        macro_f1 = avg_f1 / valid_classes_for_avg
        print("\n--- Макро-усредненные ---");
        print(f"  P: {macro_precision:.4f}, R: {macro_recall:.4f}, F1: {macro_f1:.4f}")
    total_tp_all = sum(stats_per_class[cls][0] for cls in DET_CLASSES_LIST_EVAL);
    total_fp_all = sum(stats_per_class[cls][1] for cls in DET_CLASSES_LIST_EVAL)
    total_gt_all = sum(total_gt_objects_per_class[cls] for cls in DET_CLASSES_LIST_EVAL);
    total_fn_all = total_gt_all - total_tp_all
    micro_precision = total_tp_all / (total_tp_all + total_fp_all + 1e-7);
    micro_recall = total_tp_all / (total_gt_all + 1e-7)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-7)
    print("\n--- Микро-усредненные ---");
    print(f"  GT: {int(total_gt_all)}, TP: {int(total_tp_all)}, FP: {int(total_fp_all)}, FN: {int(total_fn_all)}");
    print(f"  P: {micro_precision:.4f}, R: {micro_recall:.4f}, F1: {micro_f1:.4f}")
    print("\nОценка завершена.")


# --- Блок if __name__ == '__main__' ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели детектора (FPN или одноуровневой).")
    parser.add_argument("--model_path", type=str, default=PREDICT_CONFIG_EVAL.get("detector_model_path"),
                        help="Путь к .keras модели детектора.")
    parser.add_argument("--conf_thresh", type=float, default=PREDICT_CONFIG_EVAL.get("default_conf_thresh"),
                        help="Порог уверенности для NMS.")
    parser.add_argument("--iou_thresh_nms", type=float, default=PREDICT_CONFIG_EVAL.get("default_iou_thresh"),
                        help="Порог IoU для NMS.")
    parser.add_argument("--iou_thresh_match", type=float, default=0.5, help="Порог IoU для сопоставления TP/FP.")
    parser.add_argument("--max_dets", type=int, default=PREDICT_CONFIG_EVAL.get("default_max_dets"),
                        help="Макс. детекций после NMS.")

    args_eval = parser.parse_args()

    if not args_eval.model_path or not (Path(_project_root_eval) / args_eval.model_path).exists():
        print(f"ОШИБКА: Файл модели '{args_eval.model_path}' не найден или путь некорректен.")
        exit()

    # Определяем тип детектора для evaluate на основе predict_config.yaml
    # Эта переменная DETECTOR_TYPE_EVAL уже определена глобально выше
    if DETECTOR_TYPE_EVAL not in ["fpn", "single_level"]:
        print(
            f"ОШИБКА: Некорректный detector_type ('{DETECTOR_TYPE_EVAL}') в predict_config.yaml. Должен быть 'fpn' или 'single_level'.")
        exit()
    print(f"INFO: Скрипт оценки будет использовать логику для типа детектора: '{DETECTOR_TYPE_EVAL}'")

    evaluate_detector_on_validation_set(
        args_eval.model_path,
        args_eval.conf_thresh,
        args_eval.iou_thresh_nms,
        args_eval.iou_thresh_match,
        args_eval.max_dets
    )