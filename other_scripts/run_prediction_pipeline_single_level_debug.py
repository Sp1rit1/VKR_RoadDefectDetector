# run_prediction_pipeline_single_level_debug.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import argparse
from pathlib import Path
import json
import sys

# --- 1. Настройка sys.path ---
_current_script_path = Path(__file__).resolve().parent
_project_root_pipeline_sl = _current_script_path
_src_path_pipeline_sl = _project_root_pipeline_sl / 'src'
if str(_src_path_pipeline_sl) not in sys.path: sys.path.insert(0, str(_src_path_pipeline_sl))
if str(_project_root_pipeline_sl) not in sys.path: sys.path.insert(0, str(_project_root_pipeline_sl))

# --- 2. Импорты из твоих модулей ---
CUSTOM_OBJECTS_DETECTOR_SL_PIPE = {}
try:
    from losses.other_losses.detection_losses_single_level_debug import compute_detector_loss_single_level_debug

    CUSTOM_OBJECTS_DETECTOR_SL_PIPE = {
        'compute_detector_loss_single_level_debug': compute_detector_loss_single_level_debug}
    print("INFO (pipeline_sdl): Кастомная функция потерь детектора ЗАГРУЖЕНА.")
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ (pipeline_sdl): Кастомная функция потерь детектора НЕ НАЙДЕНА.")

SDL_DATA_PARAMS_LOADED_PIPE = False
# Эти переменные будут использоваться для параметров декодирования
# Они должны быть определены в detector_data_loader_single_level_debug.py на основе его конфига
# И здесь мы их импортируем.
try:
    from datasets.other_loaders.detector_data_loader_single_level_debug import (
        ANCHORS_WH_P4_DEBUG_SDL_G as DET_ANCHORS_WH_NORM_PIPE,
        GRID_H_P4_DEBUG_SDL_G as DET_GRID_H_PIPE,
        GRID_W_P4_DEBUG_SDL_G as DET_GRID_W_PIPE,
        CLASSES_LIST_SDL_G as DET_CLASSES_LIST_PIPE,
        NUM_CLASSES_SDL_G as DET_NUM_CLASSES_PIPE,
        TARGET_IMG_HEIGHT_SDL_G as DET_TARGET_IMG_HEIGHT_PIPE,  # Для предобработки входа детектора
        TARGET_IMG_WIDTH_SDL_G as DET_TARGET_IMG_WIDTH_PIPE
    )

    SDL_DATA_PARAMS_LOADED_PIPE = True
    print("INFO (pipeline_sdl): Параметры детектора из detector_data_loader_single_level_debug УСПЕШНО импортированы.")
except ImportError as e_sdl_imp_pipe:
    print(
        f"ОШИБКА (pipeline_sdl): Не импортированы параметры из detector_data_loader_single_level_debug: {e_sdl_imp_pipe}")
    DET_TARGET_IMG_HEIGHT_PIPE, DET_TARGET_IMG_WIDTH_PIPE = 416, 416
    DET_CLASSES_LIST_PIPE = ['pit', 'crack'];
    DET_NUM_CLASSES_PIPE = 2
    DET_ANCHORS_WH_NORM_PIPE = np.array([[0.15, 0.15]] * 7, dtype=np.float32)  # Важно, чтобы было 7 якорей
    DET_GRID_H_PIPE, DET_GRID_W_PIPE = 26, 26
    print("ПРЕДУПРЕЖДЕНИЕ (pipeline_sdl): Используются АВАРИЙНЫЕ ДЕФОЛТЫ для параметров детектора.")


# --- 3. Загрузка Конфигураций ---
def load_config_pipeline(config_path_obj, config_name_str):
    # ... (код load_config_pipeline_sl без изменений)
    default_on_error = {}
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg: print(
            f"ПРЕДУПРЕЖДЕНИЕ: {config_path_obj.name} пуст или неверный формат для {config_name_str}."); return default_on_error
        print(f"INFO: Конфиг '{config_name_str}' успешно загружен из {config_path_obj.name}.")
        return cfg
    except FileNotFoundError:
        print(
            f"ОШИБКА: Файл {config_path_obj.name} не найден: {config_path_obj} ({config_name_str})."); return default_on_error
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении {config_path_obj.name}: {e} ({config_name_str})."); return default_on_error


print("\n--- Загрузка конфигурационных файлов для пайплайна ---")
_classifier_config_path_pipe = _src_path_pipeline_sl / 'configs' / 'classifier_config.yaml'
_predict_config_path_pipe = _src_path_pipeline_sl / 'configs' / 'predict_config.yaml'

CLASSIFIER_CONFIG_PIPE = load_config_pipeline(_classifier_config_path_pipe, "Classifier Config (Pipeline_SDL)")
PREDICT_CONFIG_PIPE = load_config_pipeline(_predict_config_path_pipe, "Predict Config (Pipeline_SDL)")

# --- 4. Параметры из Конфигов (для классификатора и инференса) ---
CLS_MODEL_PATH_PIPE = PREDICT_CONFIG_PIPE.get("classifier_model_path", "weights/classifier_placeholder.keras")
CLS_INPUT_SHAPE_PIPE = tuple(CLASSIFIER_CONFIG_PIPE.get('input_shape', [224, 224, 3]))
CLS_TARGET_IMG_HEIGHT_PIPE, CLS_TARGET_IMG_WIDTH_PIPE = CLS_INPUT_SHAPE_PIPE[0], CLS_INPUT_SHAPE_PIPE[1]
CLS_CLASS_NAMES_PIPE = CLASSIFIER_CONFIG_PIPE.get('class_names_ordered_by_keras', ['not_road', 'road'])
ROAD_CLASS_NAME_FOR_CLS_PIPE = 'road'

DET_MODEL_PATH_FOR_PIPELINE = PREDICT_CONFIG_PIPE.get("single_level_detector_model_path",
                                                      "weights/detector_sl_placeholder.keras")
SL_PREDICT_PARAMS_PIPE = PREDICT_CONFIG_PIPE.get('single_level_predict_params', {})
CONF_THRESH_ARG_DEFAULT = SL_PREDICT_PARAMS_PIPE.get("default_conf_thresh", 0.25)
IOU_THRESH_NMS_ARG_DEFAULT = SL_PREDICT_PARAMS_PIPE.get("default_iou_thresh", 0.45)
MAX_DETS_ARG_DEFAULT = SL_PREDICT_PARAMS_PIPE.get("default_max_dets", 100)
OUTPUT_PATH_TEMPLATE_ARG_DEFAULT = PREDICT_CONFIG_PIPE.get("output_path_template",
                                                           "results/{image_name}_pipeline_sdl_predicted.{ext}")


# --- 5. Вспомогательные Функции ---
def preprocess_image_for_model(image_bgr, target_height, target_width):
    # ... (без изменений)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_single_level_predictions_for_pipeline(raw_predictions_tensor,
                                                 grid_h_decode, grid_w_decode,  # Передаем явно
                                                 anchors_wh_decode, num_classes_decode):  # Передаем явно
    batch_size = tf.shape(raw_predictions_tensor)[0]
    pred_xy_raw = raw_predictions_tensor[..., 0:2];
    pred_wh_raw = raw_predictions_tensor[..., 2:4]
    pred_obj_logit = raw_predictions_tensor[..., 4:5];
    pred_class_logits = raw_predictions_tensor[..., 5:]
    gy_indices = tf.tile(tf.range(grid_h_decode, dtype=tf.float32)[:, tf.newaxis], [1, grid_w_decode])
    gx_indices = tf.tile(tf.range(grid_w_decode, dtype=tf.float32)[tf.newaxis, :], [grid_h_decode, 1])
    grid_coords_xy = tf.stack([gx_indices, gy_indices], axis=-1)
    grid_coords_xy = grid_coords_xy[tf.newaxis, :, :, tf.newaxis, :]
    grid_coords_xy = tf.tile(grid_coords_xy, [batch_size, 1, 1, anchors_wh_decode.shape[0], 1])
    pred_xy_on_grid = (tf.sigmoid(pred_xy_raw) + grid_coords_xy)
    pred_xy_normalized = pred_xy_on_grid / tf.constant([grid_w_decode, grid_h_decode], dtype=tf.float32)
    anchors_tensor = tf.constant(anchors_wh_decode, dtype=tf.float32)
    anchors_reshaped = anchors_tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pred_wh_normalized = (tf.exp(pred_wh_raw) * anchors_reshaped)
    decoded_boxes_xywh_norm = tf.concat([pred_xy_normalized, pred_wh_normalized], axis=-1)
    pred_obj_confidence = tf.sigmoid(pred_obj_logit)
    pred_class_probs = tf.sigmoid(pred_class_logits)
    return decoded_boxes_xywh_norm, pred_obj_confidence, pred_class_probs


def apply_nms_and_filter_for_pipeline(decoded_boxes_xywh_norm, obj_confidence, class_probs,
                                      grid_h_nms, grid_w_nms, num_anchors_nms, num_classes_nms,  # Передаем явно
                                      confidence_threshold_arg, iou_threshold_arg, max_detections_arg):
    batch_size = tf.shape(decoded_boxes_xywh_norm)[0]
    num_total_boxes_per_image = grid_h_nms * grid_w_nms * num_anchors_nms  # Используем переданные
    boxes_flat_xywh = tf.reshape(decoded_boxes_xywh_norm, [batch_size, num_total_boxes_per_image, 4])
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        boxes_flat_xywh[..., 1:2] - boxes_flat_xywh[..., 3:4] / 2.0,
        boxes_flat_xywh[..., 0:1] - boxes_flat_xywh[..., 2:3] / 2.0,
        boxes_flat_xywh[..., 1:2] + boxes_flat_xywh[..., 3:4] / 2.0,
        boxes_flat_xywh[..., 0:1] + boxes_flat_xywh[..., 2:3] / 2.0
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax, 0.0, 1.0)
    obj_conf_flat = tf.reshape(obj_confidence, [batch_size, num_total_boxes_per_image, 1])
    class_probs_flat = tf.reshape(class_probs,
                                  [batch_size, num_total_boxes_per_image, num_classes_nms])  # Используем переданное
    final_scores_per_class = obj_conf_flat * class_probs_flat
    boxes_for_nms = tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2)
    max_output_size_final = max_detections_arg // num_classes_nms if num_classes_nms > 0 else max_detections_arg
    if max_output_size_final == 0: max_output_size_final = 1
    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms, scores=final_scores_per_class,
        max_output_size_per_class=max_output_size_final, max_total_size=max_detections_arg,
        iou_threshold=iou_threshold_arg, score_threshold=confidence_threshold_arg, clip_boxes=False
    )
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def draw_pipeline_detections(image_bgr_input, boxes_norm_yx, scores, classes_ids,
                             class_names_list_detector_arg, original_img_w, original_img_h):
    # ... (код этой функции без изменений, она уже принимала class_names_list_detector_arg)
    image_bgr_output = image_bgr_input.copy()
    num_valid_detections = tf.get_static_value(tf.shape(boxes_norm_yx)[0]) if boxes_norm_yx.shape.rank > 1 else 0
    for i in range(num_valid_detections):
        if scores[i] < 0.001: continue
        ymin_n, xmin_n, ymax_n, xmax_n = boxes_norm_yx[i]
        xmin = int(xmin_n * original_img_w);
        ymin = int(ymin_n * original_img_h)
        xmax = int(xmax_n * original_img_w);
        ymax = int(ymax_n * original_img_h)
        class_id = int(classes_ids[i]);
        score_val = scores[i]
        label_text = f"{class_names_list_detector_arg[class_id]}: {score_val:.2f}" \
            if 0 <= class_id < len(class_names_list_detector_arg) else f"Unknown({class_id}): {score_val:.2f}"
        color = (128, 128, 128)
        if 0 <= class_id < len(class_names_list_detector_arg):
            cn = class_names_list_detector_arg[class_id]
            if cn == DET_CLASSES_LIST_PIPE[0]:
                color = (0, 0, 255)  # pit
            elif len(DET_CLASSES_LIST_PIPE) > 1 and cn == DET_CLASSES_LIST_PIPE[1]:
                color = (0, 255, 0)  # crack
        cv2.rectangle(image_bgr_output, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_bgr_output, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image_bgr_output


# --- 6. Основная функция Пайплайна ---
def run_prediction_pipeline(image_path_arg, classifier_model_path_arg, detector_model_path_arg,
                            output_dir_arg, conf_thresh_arg, iou_thresh_arg, max_dets_arg):
    if not SDL_DATA_PARAMS_LOADED_PIPE:
        print("ОШИБКА: Параметры детектора не были загружены. Пайплайн не может быть запущен.");
        return

    # ... (загрузка изображения и моделей как была) ...
    if not Path(image_path_arg).exists(): print(f"Ошибка: Изображение не найдено: {image_path_arg}"); return
    original_bgr_image = cv2.imread(image_path_arg)
    if original_bgr_image is None: print(f"Ошибка: Не удалось прочитать: {image_path_arg}"); return
    original_h, original_w = original_bgr_image.shape[:2]
    classifier_model_full_path = (_project_root_pipeline_sl / classifier_model_path_arg).resolve()
    detector_model_full_path = (_project_root_pipeline_sl / detector_model_path_arg).resolve()
    try:
        print(f"Загрузка классификатора: {classifier_model_full_path}")
        classifier_model = tf.keras.models.load_model(str(classifier_model_full_path), compile=False)
        print(f"Загрузка детектора: {detector_model_full_path}")
        detector_model = tf.keras.models.load_model(str(detector_model_full_path),
                                                    custom_objects=CUSTOM_OBJECTS_DETECTOR_SL_PIPE, compile=False)
        print("Модели успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}"); return

    # 3. Этап Классификации
    print("\n--- Этап 1: Классификация 'Дорога / Не дорога' ---")
    classifier_input = preprocess_image_for_model(original_bgr_image, CLS_TARGET_IMG_HEIGHT_PIPE,
                                                  CLS_TARGET_IMG_WIDTH_PIPE)
    classifier_prediction = classifier_model.predict(classifier_input, verbose=0)
    pipeline_status = "unknown_surface";
    final_detections_output_list = [];
    output_image_to_display = original_bgr_image.copy()
    predicted_class_name_by_classifier = "Unknown"
    if CLASSIFIER_CONFIG_PIPE.get('num_classes', 2) == 2 and classifier_prediction.shape[-1] == 1:
        confidence_road_score = classifier_prediction[0][0]
        predicted_class_name_by_classifier = ROAD_CLASS_NAME_FOR_CLS_PIPE if confidence_road_score > 0.5 else \
            next(name for name in CLS_CLASS_NAMES_PIPE if name != ROAD_CLASS_NAME_FOR_CLS_PIPE)
        print(
            f"  Предсказание классификатора: '{predicted_class_name_by_classifier}' (уверенность в 'road': {confidence_road_score:.4f})")

    if predicted_class_name_by_classifier != ROAD_CLASS_NAME_FOR_CLS_PIPE:
        print("\nРЕЗУЛЬТАТ: Дрон сбился с пути! (Обнаружена НЕ ДОРОГА)")
        pipeline_status = "off_course_not_road"
        cv2.putText(output_image_to_display, "DRONE OFF COURSE (NOT A ROAD)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    else:
        print("\n--- Этап 2: Детекция дефектов на дороге ---")
        pipeline_status = "on_course_road"
        detector_input = preprocess_image_for_model(original_bgr_image, DET_TARGET_IMG_HEIGHT_PIPE,
                                                    DET_TARGET_IMG_WIDTH_PIPE)
        raw_detector_preds_p = detector_model.predict(detector_input, verbose=0)

        decoded_boxes_xywh_p, obj_conf_p, class_probs_p = decode_single_level_predictions_for_pipeline(
            raw_detector_preds_p,
            DET_GRID_H_PIPE, DET_GRID_W_PIPE,  # Передаем параметры сетки
            DET_ANCHORS_WH_NORM_PIPE, DET_NUM_CLASSES_PIPE  # Передаем якоря и кол-во классов
        )

        nms_boxes_p, nms_scores_p, nms_classes_p, num_valid_dets_p = apply_nms_and_filter_for_pipeline(
            decoded_boxes_xywh_p, obj_conf_p, class_probs_p,
            DET_GRID_H_PIPE, DET_GRID_W_PIPE, DET_ANCHORS_WH_NORM_PIPE.shape[0], DET_NUM_CLASSES_PIPE,
            # Передаем параметры
            conf_thresh_arg, iou_thresh_arg, max_dets_arg
        )

        num_found_p = int(num_valid_dets_p[0].numpy())
        print(f"  Найдено {num_found_p} дефектов после NMS.")

        if num_found_p > 0:
            pipeline_status = "on_course_defects_found"
            boxes_to_draw_p = nms_boxes_p[0][:num_found_p].numpy()
            scores_to_draw_p = nms_scores_p[0][:num_found_p].numpy()
            classes_ids_to_draw_p = nms_classes_p[0][:num_found_p].numpy().astype(int)
            output_image_to_display = draw_pipeline_detections(
                original_bgr_image, boxes_to_draw_p, scores_to_draw_p, classes_ids_to_draw_p,
                DET_CLASSES_LIST_PIPE,  # Используем список классов детектора
                original_w, original_h
            )
            # ... (остальная логика формирования final_detections_output_list)
            for i_det in range(num_found_p):
                final_detections_output_list.append({
                    "class_id": int(classes_ids_to_draw_p[i_det]),
                    "class_name": DET_CLASSES_LIST_PIPE[int(classes_ids_to_draw_p[i_det])],
                    "confidence": float(scores_to_draw_p[i_det]),
                    "bbox_normalized_yxyx": [float(coord) for coord in boxes_to_draw_p[i_det]]
                })
        else:
            pipeline_status = "on_course_road_ok"  # ...
            cv2.putText(output_image_to_display, "ROAD OK - NO DEFECTS", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    # ... (код сохранения изображения и JSON как был) ...
    img_p_obj = Path(image_path_arg);
    output_filename_p = OUTPUT_PATH_TEMPLATE_ARG_DEFAULT.format(image_name=img_p_obj.stem, ext=img_p_obj.suffix[1:])
    output_dir_abs_p = (_project_root_pipeline_sl / output_dir_arg).resolve();
    output_dir_abs_p.mkdir(parents=True, exist_ok=True)
    final_output_path_img_p = str(output_dir_abs_p / output_filename_p)
    cv2.imwrite(final_output_path_img_p, output_image_to_display);
    print(f"\nИтоговое изображение: {final_output_path_img_p}")
    json_output_data = {"image_path": image_path_arg, "pipeline_status": pipeline_status,
                        "classifier_prediction": predicted_class_name_by_classifier,
                        "detections": final_detections_output_list if final_detections_output_list else "No defects or not road"}
    json_output_path_p = str(Path(final_output_path_img_p).with_suffix('.json'))
    with open(json_output_path_p, 'w', encoding='utf-8') as f_json_p:
        json.dump(json_output_data, f_json_p, ensure_ascii=False, indent=4)
    print(f"Результаты JSON: {json_output_path_p}")


# --- 7. Парсер Аргументов и Запуск ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный пайплайн: Классификатор + Одноуровневый Детектор.")
    parser.add_argument("--image_path", type=str, required=True, help="Путь к входному изображению.")
    parser.add_argument("--classifier_model", type=str, default=str(Path(CLS_MODEL_PATH_PIPE)),
                        help=f"Путь к модели классификатора. Дефолт: {CLS_MODEL_PATH_PIPE}")
    parser.add_argument("--detector_model", type=str, default=str(Path(DET_MODEL_PATH_FOR_PIPELINE)),
                        help=f"Путь к модели детектора. Дефолт: {DET_MODEL_PATH_FOR_PIPELINE}")
    parser.add_argument("--output_dir", type=str, default="prediction_results_pipeline",
                        help="Папка для сохранения результатов.")
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH_ARG_DEFAULT,
                        help=f"Порог уверенности NMS. Дефолт: {CONF_THRESH_ARG_DEFAULT}")
    parser.add_argument("--iou_thresh", type=float, default=IOU_THRESH_NMS_ARG_DEFAULT,
                        help=f"Порог IoU NMS. Дефолт: {IOU_THRESH_NMS_ARG_DEFAULT}")
    parser.add_argument("--max_dets", type=int, default=MAX_DETS_ARG_DEFAULT,
                        help=f"Макс. детекций. Дефолт: {MAX_DETS_ARG_DEFAULT}")
    args = parser.parse_args()

    if not SDL_DATA_PARAMS_LOADED_PIPE: print(
        "Выход из пайплайна: ошибка загрузки параметров данных детектора."); exit()
    if not PREDICT_CONFIG_PIPE: print("Выход из пайплайна: ошибка загрузки predict_config.yaml."); exit()
    if not CLASSIFIER_CONFIG_PIPE: print("Выход из пайплайна: ошибка загрузки classifier_config.yaml."); exit()

    run_prediction_pipeline(
        args.image_path, args.classifier_model, args.detector_model,
        args.output_dir, args.conf_thresh, args.iou_thresh, args.max_dets
    )