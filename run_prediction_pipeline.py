# RoadDefectDetector/run_prediction_pipeline.py
import tensorflow as tf
import numpy as np
import cv2  # OpenCV для загрузки/сохранения изображений и рисования
import yaml
import os
import argparse
import time
from pathlib import Path
import json

# --- Добавляем src в sys.path, чтобы скрипт можно было запускать из корня проекта ---
_project_root_pipeline = Path(__file__).resolve().parent
_src_path_pipeline = _project_root_pipeline / 'src'
import sys

if str(_src_path_pipeline) not in sys.path:
    sys.path.insert(0, str(_src_path_pipeline))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_PIPELINE = {}
try:
    # Импортируем ОБЕ функции потерь, так как модель могла быть сохранена с одной из них
    from losses.detection_losses import compute_detector_loss_v2_fpn

    CUSTOM_OBJECTS_PIPELINE['compute_detector_loss_v2_fpn'] = compute_detector_loss_v2_fpn
    print("INFO (run_prediction_pipeline.py): Кастомные функции потерь ЗАГРУЖЕНЫ.")
except ImportError as e_loss:
    print(f"ПРЕДУПРЕЖДЕНИЕ (run_prediction_pipeline.py): Не удалось импортировать функции потерь: {e_loss}.")
except Exception as e_gen_loss:
    print(f"ПРЕДУПРЕЖДЕНИЕ (run_prediction_pipeline.py): Общая ошибка при импорте функций потерь: {e_gen_loss}.")


# --- Загрузка Конфигураций ---
def load_config_pipeline_strict(config_path_obj, config_name_for_log):
    """Загружает конфиг, выходит из программы при ошибке."""
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:  # Проверка, что не пустой
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


print("--- Загрузка конфигурационных файлов для пайплайна ---")
_predict_config_path_obj = _src_path_pipeline / 'configs' / 'predict_config.yaml'
_detector_arch_config_path_obj = _src_path_pipeline / 'configs' / 'detector_config.yaml'  # Для архитектурных деталей детектора

PREDICT_CONFIG = load_config_pipeline_strict(_predict_config_path_obj, "Predict Config")
DETECTOR_ARCH_CONFIG = load_config_pipeline_strict(_detector_arch_config_path_obj, "Detector Architecture Config")

# --- Параметры из PREDICT_CONFIG ---
CLS_MODEL_PATH = PREDICT_CONFIG.get("classifier_model_path")
DET_MODEL_PATH = PREDICT_CONFIG.get("detector_model_path")  # Этот путь должен указывать на твою лучшую FPN модель
DETECTOR_TYPE_FROM_PREDICT_CFG = PREDICT_CONFIG.get("detector_type",
                                                    "fpn").lower()  # Убедимся, что тип детектора указан

CLS_INPUT_SHAPE = tuple(PREDICT_CONFIG.get('classifier_input_shape', [224, 224, 3]))
CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH = CLS_INPUT_SHAPE[0], CLS_INPUT_SHAPE[1]
# Имена классов классификатора, как они были упорядочены Keras при обучении
CLS_CLASS_NAMES_ORDERED = PREDICT_CONFIG.get('classifier_class_names_ordered_by_keras', ['not_road', 'road'])
CLASSIFIER_ROAD_CLASS_NAME = PREDICT_CONFIG.get('classifier_road_class_name', 'road')

# Параметры детектора из PREDICT_CONFIG (для инференса и согласованности)
DET_INPUT_SHAPE_PREDICT = tuple(PREDICT_CONFIG.get('detector_input_shape', [416, 416, 3]))
DET_TARGET_IMG_HEIGHT, DET_TARGET_IMG_WIDTH = DET_INPUT_SHAPE_PREDICT[0], DET_INPUT_SHAPE_PREDICT[1]
DET_CLASSES_LIST_PREDICT = PREDICT_CONFIG.get('detector_class_names', ['pit', 'crack'])  # Имена классов для детектора
DET_NUM_CLASSES_PREDICT = len(DET_CLASSES_LIST_PREDICT)

# --- Загрузка специфичных для детектора архитектурных параметров из DETECTOR_ARCH_CONFIG ---
# Эти параметры нужны для правильного декодирования выходов модели
if DETECTOR_TYPE_FROM_PREDICT_CFG == "fpn":
    _fpn_params_arch = DETECTOR_ARCH_CONFIG.get('fpn_detector_params', {})
    if not _fpn_params_arch: print(f"ОШИБКА: Секция 'fpn_detector_params' не найдена в detector_config.yaml."); exit()

    DET_FPN_LEVELS_ARCH = _fpn_params_arch.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
    DET_FPN_STRIDES_ARCH = _fpn_params_arch.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})
    DET_FPN_ANCHOR_CONFIGS_ARCH = _fpn_params_arch.get('detector_fpn_anchor_configs', {})
    if not DET_FPN_ANCHOR_CONFIGS_ARCH or not all(lvl in DET_FPN_ANCHOR_CONFIGS_ARCH for lvl in DET_FPN_LEVELS_ARCH):
        print(f"ОШИБКА: Конфигурация якорей для FPN не найдена/неполна в detector_config.yaml.");
        exit()

    # Собираем якоря и размеры сетки для каждого уровня FPN
    FPN_LEVEL_DETAILS_FOR_DECODE = {}
    for _lvl_name in DET_FPN_LEVELS_ARCH:
        _lvl_cfg_arch = DET_FPN_ANCHOR_CONFIGS_ARCH.get(_lvl_name, {})
        _lvl_stride_arch = DET_FPN_STRIDES_ARCH.get(_lvl_name)
        if not _lvl_cfg_arch or _lvl_stride_arch is None:
            print(f"ОШИБКА: Неполная конфигурация для FPN уровня '{_lvl_name}' в detector_config.yaml.");
            exit()

        FPN_LEVEL_DETAILS_FOR_DECODE[_lvl_name] = {
            'anchors_wh_normalized': np.array(_lvl_cfg_arch.get('anchors_wh_normalized', [[0.1, 0.1]] * 3),
                                              dtype=np.float32),
            'num_anchors': _lvl_cfg_arch.get('num_anchors_this_level', 3),
            'grid_h': DET_TARGET_IMG_HEIGHT // _lvl_stride_arch,
            'grid_w': DET_TARGET_IMG_WIDTH // _lvl_stride_arch,
            'stride': _lvl_stride_arch
        }
elif DETECTOR_TYPE_FROM_PREDICT_CFG == "single_level":  # Если вдруг будешь тестировать старую модель
    # ... (здесь должна быть аналогичная загрузка параметров для single_level из DETECTOR_ARCH_CONFIG) ...
    print("ПРЕДУПРЕЖДЕНИЕ: Логика для 'single_level' детектора в этом скрипте может быть неполной.")
    # Заглушки, чтобы скрипт не падал, но это нужно будет дописать, если реально используется
    SINGLE_ANCHORS_WH_NORM = np.array([[0.1, 0.1]] * 3, dtype=np.float32)
    SINGLE_GRID_HEIGHT = DET_TARGET_IMG_HEIGHT // 16
    SINGLE_GRID_WIDTH = DET_TARGET_IMG_WIDTH // 16
else:
    print(f"ОШИБКА: Неизвестный detector_type: '{DETECTOR_TYPE_FROM_PREDICT_CFG}'.");
    exit()


# --- Вспомогательные Функции (как в evaluate_detector.py) ---
def preprocess_image_for_model_tf(image_bgr_cv2, target_height, target_width):
    # Конвертируем BGR (OpenCV) в RGB
    image_rgb_cv2 = cv2.cvtColor(image_bgr_cv2, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb_cv2, dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_single_level_predictions_generic(raw_level_preds_tensor,
                                            level_anchors_wh_norm_np,  # Numpy array of anchors for this level
                                            level_grid_h, level_grid_w,
                                            num_model_classes):  # Количество классов, которое модель предсказывает
    """Декодирует сырые предсказания для одного уровня FPN."""
    batch_size = tf.shape(raw_level_preds_tensor)[0]
    num_anchors_this_level = level_anchors_wh_norm_np.shape[0]

    # raw_level_preds_tensor имеет форму (batch, grid_h, grid_w, num_anchors, 4+1+num_classes)
    pred_xy_raw = raw_level_preds_tensor[..., 0:2]  # (tx, ty) - смещения центра относительно ячейки
    pred_wh_raw = raw_level_preds_tensor[..., 2:4]  # (tw, th) - логарифмы масштаба относительно якоря
    pred_obj_logit = raw_level_preds_tensor[..., 4:5]  # Логит objectness
    pred_class_logits = raw_level_preds_tensor[..., 5:]  # Логиты классов

    # Создание сетки координат центров ячеек
    gy_indices = tf.tile(tf.range(level_grid_h, dtype=tf.float32)[:, tf.newaxis], [1, level_grid_w])
    gx_indices = tf.tile(tf.range(level_grid_w, dtype=tf.float32)[tf.newaxis, :], [level_grid_h, 1])
    grid_xy_centers = tf.stack([gx_indices, gy_indices], axis=-1)  # Форма (grid_h, grid_w, 2) -> (cx, cy) ячеек

    # Расширяем для батча и якорей
    grid_xy_expanded = grid_xy_centers[tf.newaxis, :, :, tf.newaxis, :]  # (1, grid_h, grid_w, 1, 2)
    grid_xy_tiled = tf.tile(grid_xy_expanded, [batch_size, 1, 1, num_anchors_this_level, 1])  # (B, Gh, Gw, A, 2)

    # Декодируем координаты центра (tx, ty -> bx_norm, by_norm)
    pred_box_center_norm = (tf.sigmoid(pred_xy_raw) + grid_xy_tiled) / tf.constant([level_grid_w, level_grid_h],
                                                                                   dtype=tf.float32)

    # Декодируем ширину и высоту (tw, th -> bw_norm, bh_norm)
    anchors_tf_tensor = tf.constant(level_anchors_wh_norm_np, dtype=tf.float32)  # (A, 2) -> (w, h)
    anchors_reshaped_for_broadcast = anchors_tf_tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # (1,1,1,A,2)
    pred_box_wh_norm = tf.exp(pred_wh_raw) * anchors_reshaped_for_broadcast

    # Собираем декодированные рамки: [x_center_norm, y_center_norm, width_norm, height_norm]
    decoded_boxes_xywh_normalized = tf.concat([pred_box_center_norm, pred_box_wh_norm], axis=-1)

    # Уверенность в объекте и вероятности классов
    obj_confidence_scores = tf.sigmoid(pred_obj_logit)
    class_probabilities = tf.sigmoid(pred_class_logits)  # Если sigmoid для каждого класса

    return decoded_boxes_xywh_normalized, obj_confidence_scores, class_probabilities


def apply_nms_and_filter_generic(all_decoded_boxes_xywh_norm_list,  # Список тензоров (B,Gh,Gw,A,4) для каждого уровня
                                 all_obj_confidence_list,  # Список тензоров (B,Gh,Gw,A,1)
                                 all_class_probs_list,  # Список тензоров (B,Gh,Gw,A,NumClasses)
                                 num_output_classes,  # Количество классов, на которые обучен детектор
                                 confidence_threshold_nms,
                                 iou_threshold_nms,
                                 max_total_detections_nms):
    batch_size = tf.shape(all_decoded_boxes_xywh_norm_list[0])[0]  # Предполагаем, что batch_size одинаковый

    flat_boxes_list = []
    flat_obj_conf_list = []
    flat_class_probs_list = []

    # Решейпим выходы с каждого уровня FPN в плоский список предсказаний
    for i in range(len(all_decoded_boxes_xywh_norm_list)):
        num_total_boxes_this_level = tf.reduce_prod(tf.shape(all_decoded_boxes_xywh_norm_list[i])[1:-1])  # Gh*Gw*A
        flat_boxes_list.append(
            tf.reshape(all_decoded_boxes_xywh_norm_list[i], [batch_size, num_total_boxes_this_level, 4]))
        flat_obj_conf_list.append(tf.reshape(all_obj_confidence_list[i], [batch_size, num_total_boxes_this_level, 1]))
        flat_class_probs_list.append(
            tf.reshape(all_class_probs_list[i], [batch_size, num_total_boxes_this_level, num_output_classes]))

    # Объединяем предсказания со всех уровней
    combined_boxes_xywh = tf.concat(flat_boxes_list, axis=1)  # (B, Total_Num_All_Boxes, 4)
    combined_obj_conf = tf.concat(flat_obj_conf_list, axis=1)  # (B, Total_Num_All_Boxes, 1)
    combined_class_probs = tf.concat(flat_class_probs_list, axis=1)  # (B, Total_Num_All_Boxes, NumClasses)

    # Преобразуем [xc, yc, w, h] в [ymin, xmin, ymax, xmax] для NMS
    boxes_ymin_xmin_ymax_xmax_norm = tf.concat([
        combined_boxes_xywh[..., 1:2] - combined_boxes_xywh[..., 3:4] / 2.0,  # y_center - height/2 (ymin)
        combined_boxes_xywh[..., 0:1] - combined_boxes_xywh[..., 2:3] / 2.0,  # x_center - width/2  (xmin)
        combined_boxes_xywh[..., 1:2] + combined_boxes_xywh[..., 3:4] / 2.0,  # y_center + height/2 (ymax)
        combined_boxes_xywh[..., 0:1] + combined_boxes_xywh[..., 2:3] / 2.0  # x_center + width/2  (xmax)
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax_norm = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax_norm, 0.0, 1.0)

    # Финальные скоры для каждого класса = objectness_confidence * class_probability
    final_class_scores = combined_obj_conf * combined_class_probs  # (B, Total_Num_All_Boxes, NumClasses)

    # tf.image.combined_non_max_suppression ожидает boxes формы [batch_size, num_boxes, q, 4]
    # и scores формы [batch_size, num_boxes, num_classes]
    # q=1 означает, что одни и те же рамки используются для всех классов.
    boxes_for_nms_tf = tf.expand_dims(boxes_ymin_xmin_ymax_xmax_norm, axis=2)  # -> (B, Total_Num_All_Boxes, 1, 4)

    max_output_per_class_nms = max_total_detections_nms // num_output_classes if num_output_classes > 0 else max_total_detections_nms
    if max_output_per_class_nms == 0: max_output_per_class_nms = 1  # Защита от нуля

    return tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms_tf,
        scores=final_class_scores,
        max_output_size_per_class=max_output_per_class_nms,
        max_total_size=max_total_detections_nms,
        iou_threshold=iou_threshold_nms,
        score_threshold=confidence_threshold_nms,
        clip_boxes=False  # Координаты уже должны быть в [0,1]
    )


def draw_detections_on_image(image_bgr_input, boxes_norm_yminxminymaxxmax_draw, scores_draw, classes_ids_draw,
                             class_names_list_draw, original_img_w, original_img_h):
    # ... (твой код draw_detections_on_image, но убедись, что class_names_list_draw это DET_CLASSES_LIST_PREDICT) ...
    # Код из твоего evaluate_detector.py здесь подходит
    image_bgr_output = image_bgr_input.copy()
    num_valid_detections_to_draw = tf.shape(boxes_norm_yminxminymaxxmax_draw)[0]
    for i in range(num_valid_detections_to_draw):
        if scores_draw[i] < 0.001: continue
        ymin_n, xmin_n, ymax_n, xmax_n = boxes_norm_yminxminymaxxmax_draw[i]
        xmin, ymin, xmax, ymax = int(xmin_n * original_img_w), int(ymin_n * original_img_h), int(
            xmax_n * original_img_w), int(ymax_n * original_img_h)
        class_id = int(classes_ids_draw[i]);
        score_val = scores_draw[i]
        label_text = f"Unknown({class_id}): {score_val:.2f}";
        color = (128, 128, 128)
        if 0 <= class_id < len(class_names_list_draw):
            label_text = f"{class_names_list_draw[class_id]}: {score_val:.2f}"
            color = (0, 0, 255) if class_names_list_draw[class_id] == DET_CLASSES_LIST_PREDICT[0] else (
            0, 255, 0)  # pit=red, crack=green (предполагая такой порядок в DET_CLASSES_LIST_PREDICT)
        cv2.rectangle(image_bgr_output, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_bgr_output, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_bgr_output


# --- КОНЕЦ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---


# --- Основной Пайплайн ---
def run_complete_pipeline(image_path_arg, classifier_model_path_arg, detector_model_path_arg,
                          output_path_arg, conf_thresh_arg, iou_thresh_arg, max_dets_arg):
    # ... (Загрузка изображения и моделей как была) ...
    # ... (Этап Классификации как был) ...
    # ... (Логика выбора DETECTOR_TYPE и загрузки параметров для FPN или Single Level) ...
    # ... (Этап Детекции с использованием новых функций) ...
    # Скопирую и адаптирую весь этот блок из твоего последнего предоставленного run_prediction_pipeline.py
    # Он уже был хорошо структурирован.
    if not os.path.exists(image_path_arg): print(f"Ошибка: Изображение не найдено: {image_path_arg}"); return
    original_bgr_image = cv2.imread(image_path_arg);
    if original_bgr_image is None: print(f"Ошибка: Не удалось прочитать: {image_path_arg}"); return
    original_h, original_w = original_bgr_image.shape[:2];
    print(f"Обработка: {image_path_arg} ({original_w}x{original_h})")
    classifier_model_full_path_abs = (_project_root_pipeline / classifier_model_path_arg).resolve()
    detector_model_full_path_abs = (_project_root_pipeline / detector_model_path_arg).resolve()
    try:
        print(f"Загрузка классификатора: {classifier_model_full_path_abs}")
        classifier_model = tf.keras.models.load_model(str(classifier_model_full_path_abs), compile=False)
        print(f"Загрузка детектора: {detector_model_full_path_abs}")
        detector_model = tf.keras.models.load_model(str(detector_model_full_path_abs),
                                                    custom_objects=CUSTOM_OBJECTS_PIPELINE, compile=False)
        print("Модели успешно загружены.")
    except Exception as e_load_model:
        print(f"Ошибка загрузки моделей: {e_load_model}"); return

    pipeline_result_json = {"image_path": image_path_arg, "status_message": "Processing...", "is_road": None,
                            "classifier_confidence": None, "defects": []}
    print("\n--- Этап 1: Классификация ---");
    cls_input_batch_tf = preprocess_image_for_model_tf(original_bgr_image, CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH)
    cls_pred_raw_tf = classifier_model.predict(cls_input_batch_tf, verbose=0)

    predicted_class_name_for_pipeline = "Unknown_Classifier_Issue";
    confidence_for_pipeline = 0.0
    if cls_pred_raw_tf.shape[-1] == 1:  # Бинарный выход (sigmoid)
        confidence_road_score = float(cls_pred_raw_tf[0][0])
        is_road_pipeline = confidence_road_score > 0.5
        predicted_class_name_for_pipeline = CLASSIFIER_ROAD_CLASS_NAME if is_road_pipeline else \
        [name for name in CLS_CLASS_NAMES_ORDERED if name != CLASSIFIER_ROAD_CLASS_NAME][0]
        confidence_for_pipeline = confidence_road_score if is_road_pipeline else 1.0 - confidence_road_score
    else:  # Мультиклассовый выход (softmax)
        predicted_class_index_pipeline = int(np.argmax(cls_pred_raw_tf[0]))
        confidence_for_pipeline = float(cls_pred_raw_tf[0][predicted_class_index_pipeline])
        if 0 <= predicted_class_index_pipeline < len(CLS_CLASS_NAMES_ORDERED):
            predicted_class_name_for_pipeline = CLS_CLASS_NAMES_ORDERED[predicted_class_index_pipeline]

    pipeline_result_json["is_road"] = (predicted_class_name_for_pipeline == CLASSIFIER_ROAD_CLASS_NAME)
    pipeline_result_json["classifier_confidence"] = confidence_for_pipeline
    print(f"Классификатор: '{predicted_class_name_for_pipeline}' (уверенность: {confidence_for_pipeline:.3f})")

    output_image_to_display = original_bgr_image.copy()
    if not pipeline_result_json["is_road"]:
        pipeline_result_json["status_message"] = "Дрон сбился с пути (НЕ дорога)";
        print(f"\nРЕЗУЛЬТАТ: {pipeline_result_json['status_message']}")
        cv2.putText(output_image_to_display, "DRONE OFF COURSE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)
    else:
        print("\n--- Этап 2: Детекция дефектов ---")
        detector_input_batch_tf = preprocess_image_for_model_tf(original_bgr_image, DET_TARGET_IMG_HEIGHT,
                                                                DET_TARGET_IMG_WIDTH)

        print("  Предсказание детектора...")
        start_time_detector_inference = time.time()
        raw_detector_outputs_from_model = detector_model.predict(detector_input_batch_tf, verbose=0)
        end_time_detector_inference = time.time()
        print(f"  Время инференса детектора: {end_time_detector_inference - start_time_detector_inference:.4f} секунд")

        all_level_decoded_boxes_list, all_level_obj_conf_list, all_level_class_probs_list = [], [], []

        if DETECTOR_TYPE_FROM_PREDICT_CFG == "fpn":
            if not isinstance(raw_detector_outputs_from_model, list) or len(raw_detector_outputs_from_model) != len(
                    DET_FPN_LEVELS_ARCH):
                print(f"ОШИБКА: Выход FPN модели не является списком из {len(DET_FPN_LEVELS_ARCH)} тензоров!");
                return

            print(f"  Обработка выхода FPN модели ({len(raw_detector_outputs_from_model)} уровней).")
            for i_level_decode, fpn_level_name_decode in enumerate(DET_FPN_LEVELS_ARCH):
                raw_preds_this_level = raw_detector_outputs_from_model[i_level_decode]
                level_config_for_decode = FPN_LEVEL_DETAILS_FOR_DECODE.get(fpn_level_name_decode)
                if not level_config_for_decode: print(
                    f"ОШИБКА: Нет конфига для FPN уровня {fpn_level_name_decode} в FPN_LEVEL_DETAILS_FOR_DECODE"); continue

                decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                    raw_preds_this_level,
                    level_config_for_decode['anchors_wh_normalized'],
                    level_config_for_decode['grid_h'],
                    level_config_for_decode['grid_w'],
                    DET_NUM_CLASSES_PREDICT
                )
                all_level_decoded_boxes_list.append(decoded_boxes)
                all_level_obj_conf_list.append(obj_conf)
                all_level_class_probs_list.append(class_probs)

        # elif DETECTOR_TYPE_FROM_PREDICT_CFG == "single_level":
        # ... (твоя логика для single_level, если она будет отличаться)

        if not all_level_decoded_boxes_list:
            num_final_defects = 0
        else:
            final_boxes_nms_norm, final_scores_nms, final_classes_ids_nms, num_valid_detections_nms = apply_nms_and_filter_generic(
                all_level_decoded_boxes_list, all_level_obj_conf_list, all_level_class_probs_list,
                DET_NUM_CLASSES_PREDICT, conf_thresh_arg, iou_thresh_arg, max_dets_arg
            )
            num_final_defects = int(num_valid_detections_nms[0].numpy())  # num_valid_detections_nms это (B,)

        print(f"  Найдено {num_final_defects} дефектов после NMS.")
        if num_final_defects > 0:
            pipeline_result_json["status_message"] = "Обнаружены дефекты"
            boxes_to_draw_nms_norm = final_boxes_nms_norm[0][:num_final_defects].numpy()
            scores_to_draw_nms = final_scores_nms[0][:num_final_defects].numpy()
            classes_ids_to_draw_nms = final_classes_ids_nms[0][:num_final_defects].numpy()

            for k_det in range(num_final_defects):
                class_id_int = int(classes_ids_to_draw_nms[k_det])
                pipeline_result_json["defects"].append({
                    "class_id": class_id_int,
                    "class_name": DET_CLASSES_LIST_PREDICT[class_id_int] if 0 <= class_id_int < len(
                        DET_CLASSES_LIST_PREDICT) else "Unknown",
                    "confidence": float(scores_to_draw_nms[k_det]),
                    "bbox_normalized_ymin_xmin_ymax_xmax": [float(coord) for coord in boxes_to_draw_nms_norm[k_det]]
                })
            output_image_to_display = draw_detections_on_image(original_bgr_image, boxes_to_draw_nms_norm,
                                                               scores_to_draw_nms,
                                                               classes_ids_to_draw_nms, DET_CLASSES_LIST_PREDICT,
                                                               original_w, original_h)
            print("РЕЗУЛЬТАТ: Обнаружены дефекты.")
        else:
            pipeline_result_json["status_message"] = "Дорога в норме (дефекты не обнаружены)"
            print(f"РЕЗУЛЬТАТ: {pipeline_result_json['status_message']}");
            cv2.putText(output_image_to_display, "ROAD OK", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Сохранение Результата ---
    # ... (логика сохранения как была)
    final_output_image_path = "";
    output_dir_for_results_abs = _project_root_pipeline / PREDICT_CONFIG.get("output_dir",
                                                                             "prediction_results_pipeline_default")
    output_dir_for_results_abs.mkdir(parents=True, exist_ok=True)
    if output_path_arg:
        final_output_image_path = output_path_arg
        if not os.path.isabs(final_output_image_path): final_output_image_path = str(
            output_dir_for_results_abs / Path(final_output_image_path).name)
    else:
        img_path_obj_for_name = Path(image_path_arg);
        image_name_base = img_path_obj_for_name.stem;
        current_ext = img_path_obj_for_name.suffix
        final_output_image_path = str(output_dir_for_results_abs / f"{image_name_base}_predicted_pipeline{current_ext}")
    try:
        cv2.imwrite(final_output_image_path, output_image_to_display)
        print(f"\nИтоговое изображение сохранено в: {final_output_image_path}")
        pipeline_result_json["processed_image_path"] = final_output_image_path
    except Exception as e_img_write:
        print(f"ОШИБКА сохранения изображения в {final_output_image_path}: {e_img_write}")

    print("\n--- JSON Ответ Пайплайна ---");
    print(json.dumps(pipeline_result_json, indent=4, ensure_ascii=False))
    # Можно также сохранить JSON в файл
    # json_output_path = os.path.splitext(final_output_image_path)[0] + ".json"
    # with open(json_output_path, 'w', encoding='utf-8') as f_json:
    #    json.dump(pipeline_result_json, f_json, indent=4, ensure_ascii=False)
    # print(f"JSON результат сохранен в: {json_output_path}")


if __name__ == "__main__":
    # ... (argparse как был, с дефолтами из PREDICT_CONFIG) ...
    parser = argparse.ArgumentParser(description="Пайплайн: Классификатор + Детектор.")
    parser.add_argument("--image_path", type=str, required=True, help="Путь к входному изображению.")
    parser.add_argument("--classifier_model_path", type=str, default=PREDICT_CONFIG.get("classifier_model_path"),
                        help="Путь к модели классификатора.")
    parser.add_argument("--detector_model_path", type=str, default=PREDICT_CONFIG.get("detector_model_path"),
                        help="Путь к модели детектора.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Путь для сохранения результата (если не указан, генерируется автоматически).")
    parser.add_argument("--conf_thresh", type=float, default=PREDICT_CONFIG.get("default_conf_thresh"),
                        help="Порог уверенности для NMS.")
    parser.add_argument("--iou_thresh", type=float, default=PREDICT_CONFIG.get("default_iou_thresh"),
                        help="Порог IoU для NMS.")
    parser.add_argument("--max_dets", type=int, default=PREDICT_CONFIG.get("default_max_dets"),
                        help="Макс. детекций после NMS.")
    args_pipeline = parser.parse_args()
    if not args_pipeline.classifier_model_path or not (
            _project_root_pipeline / args_pipeline.classifier_model_path).exists():
        print(f"ОШИБКА: Модель классификатора не найдена или путь не указан: {args_pipeline.classifier_model_path}");
        exit()
    if not args_pipeline.detector_model_path or not (
            _project_root_pipeline / args_pipeline.detector_model_path).exists():
        print(f"ОШИБКА: Модель детектора не найдена или путь не указан: {args_pipeline.detector_model_path}");
        exit()
    if DETECTOR_TYPE_FROM_PREDICT_CFG not in ["fpn", "single_level"]:
        print(f"ОШИБКА: Некорректный detector_type ('{DETECTOR_TYPE_FROM_PREDICT_CFG}') в predict_config.yaml.");
        exit()

    run_complete_pipeline(args_pipeline.image_path, args_pipeline.classifier_model_path,
                          args_pipeline.detector_model_path,
                          args_pipeline.output_path, args_pipeline.conf_thresh, args_pipeline.iou_thresh,
                          args_pipeline.max_dets)