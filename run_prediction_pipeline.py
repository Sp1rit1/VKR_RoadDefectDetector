# RoadDefectDetector/run_prediction_pipeline.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import argparse
import time
from pathlib import Path
import json

# --- Добавляем src в sys.path ---
_project_root_pipeline = Path(__file__).resolve().parent
_src_path_pipeline = _project_root_pipeline / 'src'
import sys

if str(_src_path_pipeline) not in sys.path:
    sys.path.insert(0, str(_src_path_pipeline))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_PIPELINE = {}
try:
    from losses.detection_losses import compute_detector_loss_v2_fpn, compute_detector_loss_v1

    CUSTOM_OBJECTS_PIPELINE['compute_detector_loss_v2_fpn'] = compute_detector_loss_v2_fpn
    CUSTOM_OBJECTS_PIPELINE['compute_detector_loss_v1'] = compute_detector_loss_v1
    print("INFO (run_prediction_pipeline.py): Кастомные функции потерь ЗАГРУЖЕНЫ.")
except ImportError as e_loss:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (run_prediction_pipeline.py): Одна или несколько кастомных функций потерь не найдены: {e_loss}.")
except Exception as e_gen_loss:
    print(f"ПРЕДУПРЕЖДЕНИЕ (run_prediction_pipeline.py): Общая ошибка при импорте функций потерь: {e_gen_loss}.")


# --- Загрузка Конфигураций ---
def load_config_pipeline_strict(config_path_obj, config_name_for_log):
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


_predict_config_path_obj = _src_path_pipeline / 'configs' / 'predict_config.yaml'
_detector_arch_config_path_obj = _src_path_pipeline / 'configs' / 'detector_config.yaml'

print("--- Загрузка конфигурационных файлов для пайплайна ---")
PREDICT_CONFIG = load_config_pipeline_strict(_predict_config_path_obj, "Predict Config")
DETECTOR_ARCH_CONFIG = load_config_pipeline_strict(_detector_arch_config_path_obj, "Detector Architecture Config")

# --- Параметры из PREDICT_CONFIG ---
CLS_MODEL_PATH = PREDICT_CONFIG.get("classifier_model_path")
DET_MODEL_PATH = PREDICT_CONFIG.get("detector_model_path")
DETECTOR_TYPE = PREDICT_CONFIG.get("detector_type", "fpn").lower()

CLS_INPUT_SHAPE = tuple(PREDICT_CONFIG.get('classifier_input_shape', [224, 224, 3]))
CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH = CLS_INPUT_SHAPE[0], CLS_INPUT_SHAPE[1]
CLS_CLASS_NAMES_FROM_PREDICT_CFG = PREDICT_CONFIG.get('classifier_class_names_ordered_by_keras', ['not_road', 'road'])
CLASSIFIER_ROAD_CLASS_NAME_CFG = PREDICT_CONFIG.get('classifier_road_class_name', 'road')

# Общие параметры детектора из predict_config (для проверки согласованности и базовых вещей)
DET_INPUT_SHAPE_PREDICT = tuple(PREDICT_CONFIG.get('detector_input_shape', [416, 416, 3]))
DET_TARGET_IMG_HEIGHT, DET_TARGET_IMG_WIDTH = DET_INPUT_SHAPE_PREDICT[0], DET_INPUT_SHAPE_PREDICT[1]
DET_CLASSES_LIST_PREDICT = PREDICT_CONFIG.get('detector_class_names', ['pit', 'crack'])
DET_NUM_CLASSES = len(DET_CLASSES_LIST_PREDICT)

# --- Загрузка специфичных для детектора параметров из DETECTOR_ARCH_CONFIG на основе DETECTOR_TYPE ---
if DETECTOR_TYPE == "fpn":
    fpn_params = DETECTOR_ARCH_CONFIG.get('fpn_detector_params', {})
    if not fpn_params: print(f"ОШИБКА: Секция 'fpn_detector_params' не найдена в detector_config.yaml."); exit()
    # Проверка согласованности базовых параметров
    if tuple(fpn_params.get('input_shape', DET_INPUT_SHAPE_PREDICT)) != DET_INPUT_SHAPE_PREDICT:
        print("ПРЕДУПРЕЖДЕНИЕ: input_shape в predict_config и fpn_detector_params в detector_config отличаются!")
    if fpn_params.get('classes', DET_CLASSES_LIST_PREDICT) != DET_CLASSES_LIST_PREDICT:
        print("ПРЕДУПРЕЖДЕНИЕ: classes в predict_config и fpn_detector_params в detector_config отличаются!")

    DET_FPN_LEVELS = fpn_params.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
    DET_FPN_STRIDES = fpn_params.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})
    DET_FPN_ANCHOR_CONFIGS = fpn_params.get('detector_fpn_anchor_configs', {})
    if not DET_FPN_ANCHOR_CONFIGS or not all(lvl in DET_FPN_ANCHOR_CONFIGS for lvl in DET_FPN_LEVELS):
        print(f"ОШИБКА: Конфигурация якорей для FPN не найдена/неполна в detector_config.yaml -> fpn_detector_params");
        exit()

elif DETECTOR_TYPE == "single_level":
    sl_params = DETECTOR_ARCH_CONFIG.get('single_level_detector_params', {})
    if not sl_params: print(f"ОШИБКА: Секция 'single_level_detector_params' не найдена в detector_config.yaml."); exit()
    if tuple(sl_params.get('input_shape', DET_INPUT_SHAPE_PREDICT)) != DET_INPUT_SHAPE_PREDICT:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: input_shape в predict_config и single_level_detector_params в detector_config отличаются!")
    if sl_params.get('classes', DET_CLASSES_LIST_PREDICT) != DET_CLASSES_LIST_PREDICT:
        print("ПРЕДУПРЕЖДЕНИЕ: classes в predict_config и single_level_detector_params в detector_config отличаются!")

    SINGLE_ANCHORS_WH_NORM_LIST = sl_params.get('anchors_wh_normalized',
                                                [[0.1, 0.1]] * sl_params.get('num_anchors_per_location', 3))
    SINGLE_ANCHORS_WH_NORM = np.array(SINGLE_ANCHORS_WH_NORM_LIST, dtype=np.float32)
    SINGLE_NUM_ANCHORS = SINGLE_ANCHORS_WH_NORM.shape[0]
    SINGLE_NETWORK_STRIDE = sl_params.get('network_stride', 16)
    SINGLE_GRID_HEIGHT = DET_TARGET_IMG_HEIGHT // SINGLE_NETWORK_STRIDE
    SINGLE_GRID_WIDTH = DET_TARGET_IMG_WIDTH // SINGLE_NETWORK_STRIDE
else:
    print(f"ОШИБКА: Неизвестный detector_type: '{DETECTOR_TYPE}'. Должен быть 'fpn' или 'single_level'.");
    exit()


# --- Вспомогательные Функции (копируем из evaluate_detector.py) ---
# preprocess_image_for_model_tf, decode_single_level_predictions_generic,
# apply_nms_and_filter_generic, draw_detections_on_image
# (Вставь сюда полные версии этих функций из evaluate_detector.py, они идентичны)
def preprocess_image_for_model_tf(image_bgr, target_height, target_width):
    image_rgb = tf.image.decode_image(tf.io.encode_jpeg(tf.reverse(image_bgr, axis=[-1])), channels=3)
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


def draw_detections_on_image(image_bgr_input, boxes_norm_yminxminymaxxmax, scores, classes_ids,
                             class_names_list_detector, original_img_w, original_img_h):
    image_bgr_output = image_bgr_input.copy()
    num_valid_detections_tf = tf.shape(boxes_norm_yminxminymaxxmax)[0]
    num_valid_detections = num_valid_detections_tf.numpy() if hasattr(num_valid_detections_tf,
                                                                      'numpy') else num_valid_detections_tf
    for i in range(num_valid_detections):
        if scores[i] < 0.001: continue
        ymin_n, xmin_n, ymax_n, xmax_n = boxes_norm_yminxminymaxxmax[i]
        xmin = int(xmin_n * original_img_w);
        ymin = int(ymin_n * original_img_h)
        xmax = int(xmax_n * original_img_w);
        ymax = int(ymax_n * original_img_h)
        class_id = int(classes_ids[i]);
        score_val = scores[i]
        label_text = f"Unknown({class_id}): {score_val:.2f}";
        color = (128, 128, 128)
        if 0 <= class_id < len(class_names_list_detector):
            label_text = f"{class_names_list_detector[class_id]}: {score_val:.2f}"
            color = (0, 0, 255) if class_names_list_detector[class_id] == DET_CLASSES_LIST_PREDICT[0] else (
            0, 255, 0)  # pit=red, crack=green
        cv2.rectangle(image_bgr_output, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_bgr_output, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_bgr_output


# --- Конец Вспомогательных Функций ---

# --- Основной Пайплайн ---
def run_complete_pipeline(image_path_arg, classifier_model_path_arg, detector_model_path_arg,
                          output_path_arg, conf_thresh_arg, iou_thresh_arg, max_dets_arg):
    # ... (Загрузка изображения и моделей как была) ...
    # ... (Этап Классификации как был) ...
    # Я скопирую эти блоки для полноты из твоего предыдущего полного кода run_prediction_pipeline.py
    if not os.path.exists(image_path_arg): print(f"Ошибка: Изображение не найдено: {image_path_arg}"); return
    original_bgr_image = cv2.imread(image_path_arg);
    if original_bgr_image is None: print(f"Ошибка: Не удалось прочитать: {image_path_arg}"); return
    original_h, original_w = original_bgr_image.shape[:2];
    print(f"Обработка: {image_path_arg} ({original_w}x{original_h})")
    classifier_model_full_path = _project_root_pipeline / classifier_model_path_arg
    detector_model_full_path = _project_root_pipeline / detector_model_path_arg
    try:
        classifier_model = tf.keras.models.load_model(str(classifier_model_full_path), compile=False)
        detector_model = tf.keras.models.load_model(str(detector_model_full_path),
                                                    custom_objects=CUSTOM_OBJECTS_PIPELINE, compile=False)
        print("Модели загружены.")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}"); return
    pipeline_result = {"image_path": image_path_arg, "status_message": "Обработка...", "is_road": None,
                       "classifier_confidence": None, "defects": []}
    print("\n--- Этап 1: Классификация ---");
    cls_input = preprocess_image_for_model_tf(original_bgr_image, CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH)
    cls_pred = classifier_model.predict(cls_input, verbose=0)
    if cls_pred.shape[-1] == 1:  # Бинарный
        conf_road = cls_pred[0][0]
        is_road = conf_road > 0.5
        pipeline_result["is_road"] = bool(is_road)
        pred_cls_name = CLASSIFIER_ROAD_CLASS_NAME_CFG if is_road else \
        [name for name in CLS_CLASS_NAMES_FROM_PREDICT_CFG if name != CLASSIFIER_ROAD_CLASS_NAME_CFG][0]
        pipeline_result["classifier_confidence"] = float(conf_road) if is_road else 1.0 - float(conf_road)
    else:  # softmax
        pred_cls_idx = np.argmax(cls_pred[0])
        pipeline_result["classifier_confidence"] = float(cls_pred[0][pred_cls_idx])
        pred_cls_name = CLS_CLASS_NAMES_FROM_PREDICT_CFG[pred_cls_idx]
        pipeline_result["is_road"] = (pred_cls_name == CLASSIFIER_ROAD_CLASS_NAME_CFG)
    print(f"Классификатор: '{pred_cls_name}' (уверенность: {pipeline_result['classifier_confidence']:.3f})")

    output_image_display = original_bgr_image.copy()
    if not pipeline_result["is_road"]:
        pipeline_result["status_message"] = "Дрон сбился с пути (НЕ дорога)";
        print(f"\nРЕЗУЛЬТАТ: {pipeline_result['status_message']}")
        cv2.putText(output_image_display, "DRONE OFF COURSE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        print("\n--- Этап 2: Детекция дефектов ---")
        detector_input_batch = preprocess_image_for_model_tf(original_bgr_image, DET_TARGET_IMG_HEIGHT,
                                                             DET_TARGET_IMG_WIDTH)
        raw_detector_output = detector_model.predict(detector_input_batch, verbose=0)

        all_level_boxes_xywh, all_level_obj_conf, all_level_class_probs = [], [], []

        if DETECTOR_TYPE == "fpn":
            raw_detector_predictions_list = raw_detector_output  # Это уже список
            print(f"  Обработка выхода FPN модели ({len(raw_detector_predictions_list)} уровней).")
            for i_lvl, level_key in enumerate(DET_FPN_LEVELS):  # Используем DET_FPN_LEVELS из конфига
                raw_level_preds = raw_detector_predictions_list[i_lvl]
                level_cfg_anchors = DET_FPN_ANCHOR_CONFIGS.get(level_key)  # Используем DET_FPN_ANCHOR_CONFIGS
                if level_cfg_anchors is None: print(f"ОШИБКА: Нет конфига якорей для FPN уровня {level_key}"); continue

                level_anchors_wh = np.array(level_cfg_anchors['anchors_wh_normalized'], dtype=np.float32)
                level_grid_h = DET_TARGET_IMG_HEIGHT // DET_FPN_STRIDES[level_key]  # Используем DET_FPN_STRIDES
                level_grid_w = DET_TARGET_IMG_WIDTH // DET_FPN_STRIDES[level_key]

                decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                    raw_level_preds, level_anchors_wh, level_grid_h, level_grid_w, DET_NUM_CLASSES)
                all_level_boxes_xywh.append(decoded_boxes);
                all_level_obj_conf.append(obj_conf);
                all_level_class_probs.append(class_probs)

        elif DETECTOR_TYPE == "single_level":
            print(f"  Обработка выхода одноуровневой модели.")
            decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                raw_detector_output, SINGLE_ANCHORS_WH_NORM, SINGLE_GRID_HEIGHT, SINGLE_GRID_WIDTH, DET_NUM_CLASSES)
            all_level_boxes_xywh.append(decoded_boxes);
            all_level_obj_conf.append(obj_conf);
            all_level_class_probs.append(class_probs)

        if not all_level_boxes_xywh:
            num_found_defects = 0
        else:
            final_boxes_norm, final_scores, final_classes_ids, num_valid_dets = apply_nms_and_filter_generic(
                all_level_boxes_xywh, all_level_obj_conf, all_level_class_probs,
                DET_NUM_CLASSES, conf_thresh_arg, iou_thresh_arg, max_dets_arg)
            num_found_defects = int(num_valid_dets[0].numpy())

        print(f"  Найдено {num_found_defects} дефектов после NMS.")
        if num_found_defects > 0:
            # ... (Логика формирования JSON и рисования как была, но используем DET_CLASSES_LIST_PREDICT) ...
            pipeline_result["status_message"] = "Обнаружены дефекты"
            boxes_to_draw_norm = final_boxes_norm[0][:num_found_defects].numpy()
            scores_to_draw = final_scores[0][:num_found_defects].numpy()
            classes_ids_to_draw = final_classes_ids[0][:num_found_defects].numpy()
            for k_idx_det in range(num_found_defects):
                class_id_int_det = int(classes_ids_to_draw[k_idx_det])
                pipeline_result["defects"].append({
                    "class_id": class_id_int_det,
                    "class_name": DET_CLASSES_LIST_PREDICT[class_id_int_det] if 0 <= class_id_int_det < len(
                        DET_CLASSES_LIST_PREDICT) else "Unknown",
                    "confidence": float(scores_to_draw[k_idx_det]),
                    "bbox_normalized_ymin_xmin_ymax_xmax": [float(coord) for coord in boxes_to_draw_norm[k_idx_det]]})
            output_image_display = draw_detections_on_image(original_bgr_image, boxes_to_draw_norm, scores_to_draw,
                                                            classes_ids_to_draw, DET_CLASSES_LIST_PREDICT, original_w,
                                                            original_h)  # Используем DET_CLASSES_LIST_PREDICT
            print("РЕЗУЛЬТАТ: Обнаружены дефекты.")
        else:
            status_msg = "Дорога в норме (дефекты не обнаружены)";
            pipeline_result["status_message"] = status_msg
            print(f"РЕЗУЛЬТАТ: {status_msg}");
            cv2.putText(output_image_display, "ROAD OK", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ... (Логика сохранения файла и вывода JSON как была) ...
    final_output_path_str = "";
    output_dir_abs = _project_root_pipeline / PREDICT_CONFIG.get("output_dir", "results_default")
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    if output_path_arg:
        final_output_path_str = output_path_arg
        if not os.path.isabs(final_output_path_str): final_output_path_str = str(
            output_dir_abs / Path(final_output_path_str).name)
    else:
        img_path_obj = Path(image_path_arg);
        image_name = img_path_obj.stem;
        ext = img_path_obj.suffix
        final_output_path_str = str(output_dir_abs / f"{image_name}_predicted_pipeline{ext}")
    try:
        cv2.imwrite(final_output_path_str, output_image_display)
        print(f"\nИтоговое изображение: {final_output_path_str}")
        pipeline_result["processed_image_path"] = final_output_path_str
    except Exception as e_write:
        print(f"ОШИБКА сохранения: {final_output_path_str}: {e_write}")
    print("\n--- JSON Ответ ---");
    print(json.dumps(pipeline_result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    # ... (argparse как был, но default'ы для моделей и порогов берутся из PREDICT_CONFIG) ...
    parser = argparse.ArgumentParser(description="Пайплайн: Классификатор + Детектор.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--classifier_model_path", type=str, default=PREDICT_CONFIG.get("classifier_model_path"))
    parser.add_argument("--detector_model_path", type=str, default=PREDICT_CONFIG.get("detector_model_path"))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--conf_thresh", type=float, default=PREDICT_CONFIG.get("default_conf_thresh"))
    parser.add_argument("--iou_thresh", type=float, default=PREDICT_CONFIG.get("default_iou_thresh"))
    parser.add_argument("--max_dets", type=int, default=PREDICT_CONFIG.get("default_max_dets"))
    args_pipeline = parser.parse_args()
    if not args_pipeline.classifier_model_path or not (
            _project_root_pipeline / args_pipeline.classifier_model_path).exists():
        print(f"ОШИБКА: Модель классификатора не найдена: {args_pipeline.classifier_model_path}");
        exit()
    if not args_pipeline.detector_model_path or not (
            _project_root_pipeline / args_pipeline.detector_model_path).exists():
        print(f"ОШИБКА: Модель детектора не найдена: {args_pipeline.detector_model_path}");
        exit()

    # Проверка, что DETECTOR_TYPE корректно распознан из predict_config.yaml
    if DETECTOR_TYPE not in ["fpn", "single_level"]:
        print(
            f"ОШИБКА: Некорректный detector_type ('{DETECTOR_TYPE}') в predict_config.yaml. Должен быть 'fpn' или 'single_level'.")
        exit()

    run_complete_pipeline(
        args_pipeline.image_path, args_pipeline.classifier_model_path, args_pipeline.detector_model_path,
        args_pipeline.output_path, args_pipeline.conf_thresh, args_pipeline.iou_thresh, args_pipeline.max_dets
    )