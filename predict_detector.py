# RoadDefectDetector/predict_detector.py (или predict_pipeline.py)
import tensorflow as tf
import numpy as np
import cv2  # OpenCV для загрузки/сохранения изображений и рисования
import yaml
import os
import argparse
import time
from pathlib import Path

# --- Добавляем src в sys.path ---
_project_root_predict = Path(__file__).parent.resolve()
_src_path_predict = _project_root_predict / 'src'
import sys

if str(_src_path_predict) not in sys.path:
    sys.path.insert(0, str(_src_path_predict))

# --- Импорты из твоих модулей ---
from losses.detection_losses import compute_detector_loss_v1  # Для загрузки детектора

# --- Загрузка ВСЕХ Конфигураций ---
# Пути к конфигам (относительно src/)
_base_config_path = _src_path_predict / 'configs' / 'base_config.yaml'
_classifier_config_path = _src_path_predict / 'configs' / 'classifier_config.yaml'
_detector_config_path = _src_path_predict / 'configs' / 'detector_config.yaml'
_predict_config_path = _src_path_predict / 'configs' / 'predict_config.yaml'  # Наш новый конфиг

BASE_CONFIG_PREDICT = {}
CLASSIFIER_CONFIG_PREDICT = {}
DETECTOR_CONFIG_PREDICT = {}
PREDICT_PARAMS_CONFIG = {}  # Для параметров из predict_config.yaml


def load_config(config_path, default_on_error=None):
    if default_on_error is None: default_on_error = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            print(f"ПРЕДУПРЕЖДЕНИЕ: {config_path.name} пуст или имеет неверный формат. Используются дефолты.")
            return default_on_error
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path.name} не найден по пути: {config_path}. Используются дефолты.")
        return default_on_error
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении {config_path.name}: {e}. Используются дефолты.")
        return default_on_error


BASE_CONFIG_PREDICT = load_config(_base_config_path)
CLASSIFIER_CONFIG_PREDICT = load_config(_classifier_config_path, {'input_shape': [224, 224, 3], 'num_classes': 2,
                                                                  'class_names_ordered': ['not_road', 'road']})
DETECTOR_CONFIG_PREDICT = load_config(_detector_config_path, {'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
                                                              'anchors_wh_normalized': [[0.05, 0.1], [0.1, 0.05],
                                                                                        [0.1, 0.1]],
                                                              'num_anchors_per_location': 3, 'num_classes': 2})
PREDICT_PARAMS_CONFIG = load_config(_predict_config_path,
                                    {'default_conf_thresh': 0.25, 'default_iou_thresh': 0.45, 'default_max_dets': 100})

# --- Параметры из Конфигов ---
# Для Классификатора
CLS_INPUT_SHAPE = tuple(CLASSIFIER_CONFIG_PREDICT.get('input_shape'))
CLS_TARGET_IMG_HEIGHT = CLS_INPUT_SHAPE[0]
CLS_TARGET_IMG_WIDTH = CLS_INPUT_SHAPE[1]
CLS_CLASS_NAMES = CLASSIFIER_CONFIG_PREDICT.get('class_names_ordered')
ROAD_CLASS_INDEX_FOR_CLASSIFIER = CLS_CLASS_NAMES.index('road') if CLS_CLASS_NAMES and 'road' in CLS_CLASS_NAMES else 1

# Для Детектора
DET_INPUT_SHAPE = tuple(DETECTOR_CONFIG_PREDICT.get('input_shape'))
DET_TARGET_IMG_HEIGHT = DET_INPUT_SHAPE[0]
DET_TARGET_IMG_WIDTH = DET_INPUT_SHAPE[1]
DET_CLASSES_LIST = DETECTOR_CONFIG_PREDICT.get('classes')
DET_ANCHORS_WH_NORM = np.array(DETECTOR_CONFIG_PREDICT.get('anchors_wh_normalized'), dtype=np.float32)
DET_NUM_ANCHORS = DETECTOR_CONFIG_PREDICT.get('num_anchors_per_location')
DET_NUM_CLASSES = len(DET_CLASSES_LIST)
DET_NETWORK_STRIDE = 16
DET_GRID_HEIGHT = DET_TARGET_IMG_HEIGHT // DET_NETWORK_STRIDE
DET_GRID_WIDTH = DET_TARGET_IMG_WIDTH // DET_NETWORK_STRIDE


# --- Вспомогательные Функции (копируем и адаптируем из предыдущего ответа) ---
# preprocess_image_for_model, decode_predictions, apply_nms_and_filter, draw_detections
# Эти функции остаются такими же, как в предыдущем полном коде predict_detector.py

# --- НАЧАЛО КОПИРОВАНИЯ ФУНКЦИЙ (preprocess_image_for_model, decode_predictions, apply_nms_and_filter, draw_detections) ---
def preprocess_image_for_model(image_bgr, target_height, target_width):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_predictions(raw_predictions_tensor, anchors_wh_normalized, grid_h, grid_w, num_classes_detector, stride):
    # ... (код функции decode_predictions без изменений) ...
    batch_size = tf.shape(raw_predictions_tensor)[0]
    pred_xy_raw = raw_predictions_tensor[..., 0:2]
    pred_wh_raw = raw_predictions_tensor[..., 2:4]
    pred_obj_logit = raw_predictions_tensor[..., 4:5]
    pred_class_logits = raw_predictions_tensor[..., 5:]
    gy_indices = tf.tile(tf.range(grid_h, dtype=tf.float32)[:, tf.newaxis], [1, grid_w])
    gx_indices = tf.tile(tf.range(grid_w, dtype=tf.float32)[tf.newaxis, :], [grid_h, 1])
    grid_coords_xy = tf.stack([gx_indices, gy_indices], axis=-1)
    grid_coords_xy = grid_coords_xy[tf.newaxis, :, :, tf.newaxis, :]
    grid_coords_xy = tf.tile(grid_coords_xy, [batch_size, 1, 1, anchors_wh_normalized.shape[0], 1])
    pred_xy_on_grid = (tf.sigmoid(pred_xy_raw) + grid_coords_xy)
    pred_xy_normalized = pred_xy_on_grid / tf.constant([grid_w, grid_h], dtype=tf.float32)
    anchors_tensor = tf.constant(anchors_wh_normalized, dtype=tf.float32)
    anchors_reshaped = anchors_tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pred_wh_normalized = (tf.exp(pred_wh_raw) * anchors_reshaped)
    decoded_boxes_xywh_norm = tf.concat([pred_xy_normalized, pred_wh_normalized], axis=-1)
    pred_obj_confidence = tf.sigmoid(pred_obj_logit)
    pred_class_probs = tf.sigmoid(pred_class_logits)
    return decoded_boxes_xywh_norm, pred_obj_confidence, pred_class_probs


def apply_nms_and_filter(decoded_boxes_xywh_norm, obj_confidence, class_probs,
                         gh, gw, num_anchors, num_classes_detector,
                         confidence_threshold=0.25, iou_threshold=0.45, max_detections=100):
    # ... (код функции apply_nms_and_filter без изменений) ...
    batch_size = tf.shape(decoded_boxes_xywh_norm)[0]
    num_total_boxes = gh * gw * num_anchors
    boxes_flat = tf.reshape(decoded_boxes_xywh_norm, [batch_size, num_total_boxes, 4])
    obj_conf_flat = tf.reshape(obj_confidence, [batch_size, num_total_boxes, 1])
    class_probs_flat = tf.reshape(class_probs, [batch_size, num_total_boxes, num_classes_detector])
    combined_scores = obj_conf_flat * class_probs_flat
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        boxes_flat[..., 1:2] - boxes_flat[..., 3:4] / 2.0,
        boxes_flat[..., 0:1] - boxes_flat[..., 2:3] / 2.0,
        boxes_flat[..., 1:2] + boxes_flat[..., 3:4] / 2.0,
        boxes_flat[..., 0:1] + boxes_flat[..., 2:3] / 2.0
    ], axis=-1)
    max_output_per_class = max_detections // num_classes_detector if num_classes_detector > 0 else max_detections
    if max_output_per_class == 0: max_output_per_class = 1
    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2), scores=combined_scores,
        max_output_size_per_class=max_output_per_class, max_total_size=max_detections,
        iou_threshold=iou_threshold, score_threshold=confidence_threshold, clip_boxes=False
    )
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def draw_detections(image_bgr_input, boxes_norm, scores, classes_ids, class_names_list_detector, original_w,
                    original_h):
    # ... (код функции draw_detections без изменений) ...
    image_bgr = image_bgr_input.copy()
    num_detections = boxes_norm.shape[0]
    if num_detections == 0: return image_bgr
    for i in range(num_detections):
        if scores[i] < 0.01: continue
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = boxes_norm[i]
        xmin = int(xmin_norm * original_w);
        ymin = int(ymin_norm * original_h)
        xmax = int(xmax_norm * original_w);
        ymax = int(ymax_norm * original_h)
        class_id = int(classes_ids[i]);
        score = scores[i]
        if class_id < 0 or class_id >= len(class_names_list_detector):
            label = f"Unknown: {score:.2f}";
            color = (0, 0, 0)
        else:
            label = f"{class_names_list_detector[class_id]}: {score:.2f}"
            color = (0, 0, 255) if class_names_list_detector[class_id] == 'pit' else (0, 255, 0)
        cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_bgr, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_bgr


# --- КОНЕЦ КОПИРОВАНИЯ ФУНКЦИЙ ---


def run_complete_pipeline(image_path_arg, classifier_model_path_arg, detector_model_path_arg,
                          output_path_arg, conf_thresh_arg, iou_thresh_arg, max_dets_arg):
    """
    Выполняет полный пайплайн: классификатор -> детектор -> визуализация.
    Использует пути и параметры, переданные как аргументы (которые могут браться из argparse).
    """
    # 1. Загрузка исходного изображения
    if not os.path.exists(image_path_arg):
        print(f"Ошибка: Исходное изображение не найдено: {image_path_arg}")
        return
    original_bgr_image = cv2.imread(image_path_arg)
    if original_bgr_image is None:
        print(f"Ошибка: Не удалось прочитать изображение: {image_path_arg}")
        return
    original_h, original_w = original_bgr_image.shape[:2]

    # 2. Загрузка моделей
    classifier_model_full_path = os.path.join(_project_root_predict, classifier_model_path_arg)
    detector_model_full_path = os.path.join(_project_root_predict, detector_model_path_arg)

    print(f"Загрузка классификатора из: {classifier_model_full_path}")
    try:
        classifier_model = tf.keras.models.load_model(classifier_model_full_path, compile=False)
    except Exception as e:
        print(f"Ошибка загрузки модели классификатора: {e}");
        return

    print(f"Загрузка детектора из: {detector_model_full_path}")
    try:
        custom_objects_detector = {'compute_detector_loss_v1': compute_detector_loss_v1}
        detector_model = tf.keras.models.load_model(detector_model_full_path, custom_objects=custom_objects_detector,
                                                    compile=False)
    except Exception as e:
        print(f"Ошибка загрузки модели детектора: {e}");
        return
    print("Модели успешно загружены.")

    # 3. Этап Классификации
    # ... (код этапа классификации из предыдущего ответа без изменений) ...
    print("\n--- Этап 1: Классификация 'Дорога / Не дорога' ---")
    classifier_input_batch = preprocess_image_for_model(original_bgr_image, CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH)
    classifier_prediction = classifier_model.predict(classifier_input_batch)
    predicted_class_name_cls = "Unknown";
    confidence_cls = 0.0
    if CLASSIFIER_CONFIG_PREDICT.get('num_classes', 2) == 2 and classifier_prediction.shape[-1] == 1:
        confidence_road = classifier_prediction[0][0]
        if confidence_road > 0.5:
            predicted_class_name_cls = "road";
            confidence_cls = confidence_road
        else:
            predicted_class_name_cls = "not_road";
            confidence_cls = 1.0 - confidence_road
        print(
            f"Предсказание классификатора: '{predicted_class_name_cls}' с уверенностью {confidence_cls:.4f} (для 'road' было {confidence_road:.4f})")
    else:
        predicted_class_index_cls = np.argmax(classifier_prediction[0])
        confidence_cls = classifier_prediction[0][predicted_class_index_cls]
        if CLS_CLASS_NAMES and predicted_class_index_cls < len(CLS_CLASS_NAMES):
            predicted_class_name_cls = CLS_CLASS_NAMES[predicted_class_index_cls]
        print(
            f"Предсказание классификатора: '{predicted_class_name_cls}' (ID: {predicted_class_index_cls}) с уверенностью {confidence_cls:.4f}")

    # 4. Принятие решения и Детекция
    output_image_display = original_bgr_image.copy()  # Начнем с копии для рисования
    if predicted_class_name_cls == "not_road":
        print("\nРЕЗУЛЬТАТ: Дрон сбился с пути! (Обнаружена не дорога)")
        cv2.putText(output_image_display, "DRONE OFF COURSE (NOT A ROAD)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif predicted_class_name_cls == "road":
        # ... (код этапа детекции из предыдущего ответа без изменений) ...
        print("\n--- Этап 2: Детекция дефектов на дороге ---")
        detector_input_batch = preprocess_image_for_model(original_bgr_image, DET_TARGET_IMG_HEIGHT,
                                                          DET_TARGET_IMG_WIDTH)
        start_time_det = time.time();
        raw_detector_predictions = detector_model.predict(detector_input_batch);
        end_time_det = time.time()
        print(f"  Время инференса детектора: {end_time_det - start_time_det:.4f} секунд")
        decoded_boxes_xywh, obj_conf, class_probs = decode_predictions(
            raw_detector_predictions, DET_ANCHORS_WH_NORM, DET_GRID_HEIGHT, DET_GRID_WIDTH, DET_NUM_CLASSES,
            DET_NETWORK_STRIDE)
        final_boxes_norm, final_scores, final_classes_ids, num_valid_dets = apply_nms_and_filter(
            decoded_boxes_xywh, obj_conf, class_probs, DET_GRID_HEIGHT, DET_GRID_WIDTH, DET_NUM_ANCHORS,
            DET_NUM_CLASSES,
            confidence_threshold=conf_thresh_arg, iou_threshold=iou_thresh_arg, max_detections=max_dets_arg)
        num_found_defects = num_valid_dets[0].numpy()
        print(f"  Найдено {num_found_defects} дефектов после NMS.")
        if num_found_defects > 0:
            boxes_to_draw = final_boxes_norm[0][:num_found_defects].numpy()
            scores_to_draw = final_scores[0][:num_found_defects].numpy()
            classes_ids_to_draw = final_classes_ids[0][:num_found_defects].numpy()
            output_image_display = draw_detections(original_bgr_image, boxes_to_draw, scores_to_draw,
                                                   classes_ids_to_draw, DET_CLASSES_LIST, original_w, original_h)
            print("РЕЗУЛЬТАТ: Обнаружены дефекты. См. изображение.")
        else:
            print("РЕЗУЛЬТАТ: Дорога в норме (дефекты не обнаружены детектором).")
            cv2.putText(output_image_display, "ROAD OK - NO DEFECTS DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("РЕЗУЛЬТАТ: Не удалось определить тип поверхности классификатором.")
        cv2.putText(output_image_display, "UNABLE TO CLASSIFY SURFACE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Сохранение результата
    # Используем output_path_template из PREDICT_PARAMS_CONFIG
    final_output_path = ""
    if output_path_arg:  # Если пользователь указал output_path в командной строке, используем его
        final_output_path = output_path_arg
        # Если путь не абсолютный, делаем его относительно корня проекта
        if not os.path.isabs(final_output_path):
            final_output_path = str(_project_root_predict / final_output_path)
    elif PREDICT_PARAMS_CONFIG.get("output_path_template"):
        template = PREDICT_PARAMS_CONFIG["output_path_template"]
        img_path_obj = Path(image_path_arg)
        image_name = img_path_obj.stem
        ext = img_path_obj.suffix[1:]  # убираем точку
        # Заменяем плейсхолдеры
        output_filename = template.format(image_name=image_name, ext=ext)
        final_output_path = str(_project_root_predict / output_filename)  # Относительно корня проекта
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)  # Создаем папку results, если ее нет
    else:  # Фоллбэк, если и output_path_arg нет, и шаблона нет
        base, ext_in = os.path.splitext(image_path_arg)
        final_output_path = base + "_pipeline_result" + ext_in

    cv2.imwrite(final_output_path, output_image_display)
    print(f"\nИтоговое изображение сохранено в: {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный пайплайн: классификация Дорога/Не дорога + Детекция дефектов.")
    parser.add_argument("--image_path", type=str, required=True, help="Путь к входному изображению.")

    # Аргументы теперь НЕОБЯЗАТЕЛЬНЫЕ, значения по умолчанию берутся из predict_config.yaml
    parser.add_argument("--classifier_model_path", type=str,
                        default=PREDICT_PARAMS_CONFIG.get("classifier_model_path"),
                        help="Путь к модели классификатора. По умолчанию из predict_config.yaml.")
    parser.add_argument("--detector_model_path", type=str,
                        default=PREDICT_PARAMS_CONFIG.get("detector_model_path"),
                        help="Путь к модели детектора. По умолчанию из predict_config.yaml.")
    parser.add_argument("--output_path", type=str,
                        default=None,  # Будет обработано с использованием output_path_template
                        help="Путь для сохранения результата. По умолчанию используется output_path_template из predict_config.yaml.")
    parser.add_argument("--conf_thresh", type=float,
                        default=PREDICT_PARAMS_CONFIG.get("default_conf_thresh"),
                        help="Порог уверенности для NMS. По умолчанию из predict_config.yaml.")
    parser.add_argument("--iou_thresh", type=float,
                        default=PREDICT_PARAMS_CONFIG.get("default_iou_thresh"),
                        help="Порог IoU для NMS. По умолчанию из predict_config.yaml.")
    parser.add_argument("--max_dets", type=int,
                        default=PREDICT_PARAMS_CONFIG.get("default_max_dets"),
                        help="Макс. детекций после NMS. По умолчанию из predict_config.yaml.")

    args_pipeline = parser.parse_args()

    # Проверка, что пути к моделям были предоставлены (либо через аргументы, либо есть в конфиге)
    if not args_pipeline.classifier_model_path:
        print("ОШИБКА: Путь к модели классификатора не указан ни в аргументах, ни в predict_config.yaml.")
        exit()
    if not args_pipeline.detector_model_path:
        print("ОШИБКА: Путь к модели детектора не указан ни в аргументах, ни в predict_config.yaml.")
        exit()

    run_complete_pipeline(args_pipeline.image_path,
                          args_pipeline.classifier_model_path,
                          args_pipeline.detector_model_path,
                          args_pipeline.output_path,  # Передаем как есть, логика обработки output_path внутри функции
                          args_pipeline.conf_thresh,
                          args_pipeline.iou_thresh,
                          args_pipeline.max_dets)