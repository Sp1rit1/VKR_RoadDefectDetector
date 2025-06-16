# run_prediction_pipeline.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import argparse
import time
from pathlib import Path
import json

_project_root_pipeline = Path(__file__).resolve().parent
_src_path_pipeline = _project_root_pipeline / 'src'
import sys

if str(_src_path_pipeline) not in sys.path:
    sys.path.insert(0, str(_src_path_pipeline))

try:
    from losses.detection_losses import compute_detector_loss_v1, compute_detector_loss_v2_fpn

    # Мы можем не знать, какая модель загружается, поэтому передадим обе, если они есть
    CUSTOM_OBJECTS_DETECTOR = {
        'compute_detector_loss_v1': compute_detector_loss_v1,
        'compute_detector_loss_v2_fpn': compute_detector_loss_v2_fpn
    }
    print("INFO (run_prediction_pipeline.py): Кастомные функции потерь детектора для custom_objects ЗАГРУЖЕНЫ.")
except ImportError:
    CUSTOM_OBJECTS_DETECTOR = {}
    print("ПРЕДУПРЕЖДЕНИЕ (run_prediction_pipeline.py): Одна или обе кастомные функции потерь не найдены.")

CLASSIFIER_MODEL = None
DETECTOR_MODEL = None
PREDICT_CONFIG = None


def load_pipeline_configs():
    global PREDICT_CONFIG
    if PREDICT_CONFIG is not None: return True
    _predict_config_path_here = _src_path_pipeline / 'configs' / 'predict_config.yaml'
    try:
        with open(_predict_config_path_here, 'r', encoding='utf-8') as f:
            PREDICT_CONFIG = yaml.safe_load(f)
        if not isinstance(PREDICT_CONFIG, dict) or not PREDICT_CONFIG:
            print(f"ОШИБКА: predict_config.yaml ({_predict_config_path_here}) пуст или имеет неверный формат.")
            PREDICT_CONFIG = None;
            return False
        print("INFO: predict_config.yaml успешно загружен.")
        return True
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить predict_config.yaml: {e}");
        PREDICT_CONFIG = None;
        return False


def load_models_once():
    # ... (код этой функции как был, загружает модели на основе PREDICT_CONFIG) ...
    global CLASSIFIER_MODEL, DETECTOR_MODEL
    if CLASSIFIER_MODEL is not None and DETECTOR_MODEL is not None: return True
    if PREDICT_CONFIG is None:
        if not load_pipeline_configs(): return False

    classifier_model_path_rel = PREDICT_CONFIG.get("classifier_model_path")
    detector_model_path_rel = PREDICT_CONFIG.get("detector_model_path")
    if not classifier_model_path_rel or not detector_model_path_rel:
        print("ОШИБКА: Пути к моделям не указаны в predict_config.yaml");
        return False

    classifier_model_full_path = (_project_root_pipeline / classifier_model_path_rel).resolve()
    detector_model_full_path = (_project_root_pipeline / detector_model_path_rel).resolve()

    try:
        print(f"Загрузка классификатора из: {classifier_model_full_path}...")
        if not classifier_model_full_path.exists(): raise FileNotFoundError(
            f"Файл классификатора не найден: {classifier_model_full_path}")
        CLASSIFIER_MODEL = tf.keras.models.load_model(str(classifier_model_full_path), compile=False)
        print("Классификатор успешно загружен.")

        print(f"Загрузка детектора из: {detector_model_full_path}...")
        if not detector_model_full_path.exists(): raise FileNotFoundError(
            f"Файл детектора не найден: {detector_model_full_path}")
        DETECTOR_MODEL = tf.keras.models.load_model(str(detector_model_full_path),
                                                    custom_objects=CUSTOM_OBJECTS_DETECTOR, compile=False)
        print("Детектор успешно загружен.")
        return True
    except Exception as e:
        print(f"ОШИБКА загрузки одной из моделей: {e}");
        CLASSIFIER_MODEL = None;
        DETECTOR_MODEL = None;
        return False


def preprocess_image_for_model(image_bgr, target_height, target_width):
    # ... (код как был) ...
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_single_head_predictions(raw_predictions_tensor, anchors_wh_normalized_list,
                                   input_shape_detector_tuple, num_classes, network_stride):
    # ... (код этой функции как был, но принимает list of lists для якорей)
    # Убедимся, что anchors_wh_normalized_list это numpy array
    anchors_wh_normalized_np = np.array(anchors_wh_normalized_list, dtype=np.float32)

    batch_size = tf.shape(raw_predictions_tensor)[0]
    grid_h = input_shape_detector_tuple[0] // network_stride
    grid_w = input_shape_detector_tuple[1] // network_stride
    num_anchors = anchors_wh_normalized_np.shape[0]

    # Если модель уже выдает (B, Gh, Gw, A, 5+C), то reshape не нужен
    # Предположим, что raw_predictions_tensor УЖЕ имеет правильную последнюю размерность якорей
    # или был решейпнут в модели
    # Если нет, то:
    # raw_predictions_tensor = tf.reshape(raw_predictions_tensor,
    #                                     [batch_size, grid_h, grid_w, num_anchors, 5 + num_classes])

    pred_xy_raw = raw_predictions_tensor[..., 0:2]
    pred_wh_raw = raw_predictions_tensor[..., 2:4]
    pred_obj_logit = raw_predictions_tensor[..., 4:5]
    pred_class_logits = raw_predictions_tensor[..., 5:]

    gy = tf.tile(tf.range(grid_h, dtype=tf.float32)[:, tf.newaxis], [1, grid_w])
    gx = tf.tile(tf.range(grid_w, dtype=tf.float32)[tf.newaxis, :], [grid_h, 1])
    grid_xy = tf.stack([gx, gy], axis=-1)[tf.newaxis, :, :, tf.newaxis, :]
    grid_xy = tf.tile(grid_xy, [batch_size, 1, 1, num_anchors, 1])

    pred_xy = (tf.sigmoid(pred_xy_raw) + grid_xy) / tf.constant([grid_w, grid_h], dtype=tf.float32)

    anchors_tf = tf.constant(anchors_wh_normalized_np, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pred_wh = tf.exp(pred_wh_raw) * anchors_tf

    boxes_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    obj_conf = tf.sigmoid(pred_obj_logit)
    cls_probs = tf.sigmoid(pred_class_logits)

    flat_boxes = tf.reshape(boxes_xywh, [batch_size, -1, 4])
    flat_obj = tf.reshape(obj_conf, [batch_size, -1, 1])
    flat_cls = tf.reshape(cls_probs, [batch_size, -1, num_classes])
    return flat_boxes, flat_obj, flat_cls


def decode_fpn_predictions_list_output(list_of_raw_level_outputs, fpn_anchor_configs_dict,
                                       input_shape_detector_tuple, num_classes):
    all_boxes_list, all_obj_list, all_cls_list = [], [], []
    fpn_levels_order = PREDICT_CONFIG.get('detector_fpn_levels', ['P3', 'P4', 'P5'])  # Берем порядок из конфига

    for i, level_name in enumerate(fpn_levels_order):
        raw_preds_level = list_of_raw_level_outputs[i]  # (B, Gh_l, Gw_l, A_l, 5+C)
        level_cfg = fpn_anchor_configs_dict.get(level_name, {})

        anchors_this_level_list = level_cfg.get('anchors_wh_normalized', [])
        stride_this_level = level_cfg.get('stride', 8 * (2 ** i))  # Дефолтный страйд

        if not anchors_this_level_list:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Якоря для уровня FPN {level_name} не найдены в конфиге.")
            continue

        anchors_this_level_np = np.array(anchors_this_level_list, dtype=np.float32)

        # Используем decode_single_head_predictions для каждого уровня
        boxes_l_flat, obj_l_flat, cls_l_flat = decode_single_head_predictions(
            raw_preds_level,  # Уже в нужной форме (B, Gh, Gw, A, 5+C)
            anchors_this_level_np,  # Передаем якоря этого уровня
            input_shape_detector_tuple,  # Общий входной размер
            num_classes,
            stride_this_level  # Страйд этого уровня
        )
        all_boxes_list.append(boxes_l_flat)
        all_obj_list.append(obj_l_flat)
        all_cls_list.append(cls_l_flat)

    return tf.concat(all_boxes_list, axis=1), \
        tf.concat(all_obj_list, axis=1), \
        tf.concat(all_cls_list, axis=1)


def apply_nms_and_filter(decoded_boxes_xywh_norm_flat, obj_confidence_flat, class_probs_flat,
                         num_classes_detector, confidence_threshold, iou_threshold, max_detections):
    # ... (код этой функции как был) ...
    # (код этой функции как был)
    batch_size = tf.shape(decoded_boxes_xywh_norm_flat)[0]
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        decoded_boxes_xywh_norm_flat[..., 1:2] - decoded_boxes_xywh_norm_flat[..., 3:4] / 2.0,
        decoded_boxes_xywh_norm_flat[..., 0:1] - decoded_boxes_xywh_norm_flat[..., 2:3] / 2.0,
        decoded_boxes_xywh_norm_flat[..., 1:2] + decoded_boxes_xywh_norm_flat[..., 3:4] / 2.0,
        decoded_boxes_xywh_norm_flat[..., 0:1] + decoded_boxes_xywh_norm_flat[..., 2:3] / 2.0
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax, 0.0, 1.0)
    final_scores_per_class = obj_confidence_flat * class_probs_flat
    max_output_per_class = max_detections // num_classes_detector if num_classes_detector > 0 else max_detections
    if max_output_per_class == 0: max_output_per_class = 1  # Добавил эту проверку

    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2), scores=final_scores_per_class,
        max_output_size_per_class=max_output_per_class,
        max_total_size=max_detections, iou_threshold=iou_threshold, score_threshold=confidence_threshold,
        clip_boxes=False
    )
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def draw_detections_on_image(image_bgr_input, boxes_norm_yminxminymaxxmax, scores, classes_ids,
                             num_valid_detections_count, class_names_list_detector):
    # ... (код этой функции как был) ...
    output_image = image_bgr_input.copy();
    original_h, original_w = output_image.shape[:2]
    if num_valid_detections_count > 0:
        # Убедимся, что берем правильные срезы
        boxes_to_draw = boxes_norm_yminxminymaxxmax[0][:num_valid_detections_count]
        scores_to_draw = scores[0][:num_valid_detections_count]
        classes_ids_to_draw = classes_ids[0][:num_valid_detections_count].numpy().astype(int)  # было .numpy() уже

        for i in range(num_valid_detections_count):
            ymin_n, xmin_n, ymax_n, xmax_n = boxes_to_draw[i].numpy()  # Добавил .numpy()
            xmin = int(xmin_n * original_w);
            ymin = int(ymin_n * original_h)
            xmax = int(xmax_n * original_w);
            ymax = int(ymax_n * original_h)
            class_id = classes_ids_to_draw[i];
            score_val = scores_to_draw[i].numpy()  # Добавил .numpy()

            label_text = f"{class_names_list_detector[class_id]}: {score_val:.2f}"
            color = (0, 0, 255) if class_names_list_detector[class_id] == 'pit' else (0, 255, 0) if \
            class_names_list_detector[class_id] == 'crack' else (255, 0, 0)

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(output_image, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output_image


def process_image(image_path, conf_thresh, iou_thresh, max_dets):
    # ... (код загрузки изображения и классификатора как был) ...
    global PREDICT_CONFIG
    if PREDICT_CONFIG is None: return "Ошибка: Конфигурация не загружена", None
    if CLASSIFIER_MODEL is None or DETECTOR_MODEL is None: return "Ошибка: Модели не загружены", None
    original_bgr_image = cv2.imread(image_path)
    if original_bgr_image is None: return f"Ошибка: Не удалось прочитать {image_path}", None

    cls_input_shape = tuple(PREDICT_CONFIG['classifier_input_shape'])
    classifier_input_batch = preprocess_image_for_model(original_bgr_image, cls_input_shape[0], cls_input_shape[1])
    classifier_prediction = CLASSIFIER_MODEL.predict(classifier_input_batch, verbose=0)
    cls_road_name = PREDICT_CONFIG['classifier_road_class_name']
    cls_names = PREDICT_CONFIG['classifier_class_names']
    pred_cls_name = cls_names[0]  # not_road по умолчанию
    if classifier_prediction.shape[-1] == 1:  # Sigmoid
        pred_cls_name = cls_road_name if classifier_prediction[0][0] > 0.5 else cls_names[0]
    else:  # Softmax
        pred_cls_name = cls_names[np.argmax(classifier_prediction[0])]

    output_image_final = original_bgr_image.copy()
    status_message = "";
    detected_objects_info = []

    if pred_cls_name != cls_road_name:
        status_message = "Дрон сбился с пути (обнаружена НЕ дорога)"
        cv2.putText(output_image_final, "NOT A ROAD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        status_message = "Дорога"
        det_input_shape = tuple(PREDICT_CONFIG['detector_input_shape'])
        detector_input_batch = preprocess_image_for_model(original_bgr_image, det_input_shape[0], det_input_shape[1])
        raw_detector_outputs = DETECTOR_MODEL.predict(detector_input_batch, verbose=0)

        # Определяем тип модели (FPN или Single Head)
        is_fpn = 'detector_fpn_levels' in PREDICT_CONFIG and PREDICT_CONFIG['detector_fpn_levels']

        num_classes_det = len(PREDICT_CONFIG['detector_class_names'])

        if is_fpn:
            print("  INFO: Используется декодер для FPN модели.")
            fpn_anchor_cfgs = PREDICT_CONFIG['detector_fpn_anchor_configs']
            # raw_detector_outputs здесь должен быть списком тензоров, если модель FPN
            if not isinstance(raw_detector_outputs, list) or len(raw_detector_outputs) != len(
                    PREDICT_CONFIG['detector_fpn_levels']):
                return "Ошибка: Выход FPN модели не является списком ожидаемой длины.", None

            decoded_boxes_flat, obj_conf_flat, class_probs_flat = decode_fpn_predictions_list_output(
                raw_detector_outputs,  # Список выходов
                fpn_anchor_cfgs,
                det_input_shape,
                num_classes_det
            )
        else:
            print("  INFO: Используется декодер для одноуровневой модели.")
            anchors_v1 = PREDICT_CONFIG.get('detector_anchors_wh_normalized', [])
            stride_v1 = PREDICT_CONFIG.get('detector_network_stride', 16)
            if not anchors_v1: return "Ошибка: Якоря для одноуровневой модели не найдены в конфиге.", None

            # raw_detector_outputs здесь один тензор (B, Gh, Gw, A, 5+C)
            decoded_boxes_flat, obj_conf_flat, class_probs_flat = decode_single_head_predictions(
                raw_detector_outputs,
                anchors_v1,
                det_input_shape,
                num_classes_det,
                stride_v1
            )

        # ... (NMS и отрисовка как были)
        nms_boxes, nms_scores, nms_classes, nms_valid_dets = apply_nms_and_filter(
            decoded_boxes_flat, obj_conf_flat, class_probs_flat,
            num_classes_det, conf_thresh, iou_thresh, max_dets)
        num_found = int(nms_valid_dets[0].numpy())
        if num_found > 0:
            status_message = "Обнаружены дефекты";
            output_image_final = draw_detections_on_image(
                original_bgr_image, nms_boxes, nms_scores, nms_classes, num_found,
                PREDICT_CONFIG['detector_class_names'])
            for i in range(num_found):
                detected_objects_info.append({
                    "class_id": int(nms_classes[0][i].numpy()),
                    "class_name": PREDICT_CONFIG['detector_class_names'][int(nms_classes[0][i].numpy())],
                    "confidence": float(nms_scores[0][i].numpy()),
                    "bbox_normalized_ymin_xmin_ymax_xmax": nms_boxes[0][i].numpy().tolist()})
        else:
            status_message = "Нормальная дорога (дефекты не обнаружены)"
            cv2.putText(output_image_final, "ROAD OK", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ... (сохранение и возврат JSON как были) ...
    output_dir_abs = _project_root_pipeline / PREDICT_CONFIG.get("output_dir", "prediction_results")
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    output_image_filename = f"{Path(image_path).stem}_predicted.jpg"
    output_image_path = str(output_dir_abs / output_image_filename)
    cv2.imwrite(output_image_path, output_image_final)
    response_data = {"status_message": status_message, "processed_image_path": output_image_path,
                     "defects": detected_objects_info if detected_objects_info else "No defects or not a road"}
    return json.dumps(response_data, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # ... (argparse и вызов как были) ...
    parser = argparse.ArgumentParser(description="Пайплайн детекции: Классификатор + Детектор.")
    parser.add_argument("--image_path", type=str, required=True, help="Путь к входному изображению.")
    parser.add_argument("--conf_thresh", type=float, help="Порог уверенности.")
    parser.add_argument("--iou_thresh", type=float, help="Порог IoU для NMS.")
    parser.add_argument("--max_dets", type=int, help="Макс. детекций.")
    args = parser.parse_args()
    if not load_pipeline_configs(): print("Крит. ошибка конфигов."); exit()
    if not load_models_once(): print("Крит. ошибка моделей."); exit()
    conf_run = args.conf_thresh if args.conf_thresh is not None else PREDICT_CONFIG.get("default_conf_thresh", 0.25)
    iou_run = args.iou_thresh if args.iou_thresh is not None else PREDICT_CONFIG.get("default_iou_thresh", 0.45)
    max_run = args.max_dets if args.max_dets is not None else PREDICT_CONFIG.get("default_max_dets", 100)
    json_response_out = process_image(args.image_path, conf_run, iou_run, max_run)
    print("\n--- JSON Ответ ---");
    print(json_response_out)