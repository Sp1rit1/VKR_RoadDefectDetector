# evaluate_detector_single_level_debug.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import glob
from pathlib import Path
import argparse
import sys
import time

# --- Настройка sys.path для импорта из src ---
_current_script_path = Path(__file__).resolve()  # Путь к текущему скрипту
_project_root_eval_sl = _current_script_path.parent  # Корень проекта, где лежит этот скрипт
_src_path_eval_sl = _project_root_eval_sl / 'src'  # Путь к папке src
if str(_src_path_eval_sl) not in sys.path:
    sys.path.insert(0, str(_src_path_eval_sl))
if str(_project_root_eval_sl) not in sys.path:  # Добавим и корень проекта, если вдруг понадобится для других импортов
    sys.path.insert(0, str(_project_root_eval_sl))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_EVAL_SL = {}
try:
    from losses.detection_losses_single_level_debug import compute_detector_loss_single_level_debug

    CUSTOM_OBJECTS_EVAL_SL = {
        'compute_detector_loss_single_level_debug': compute_detector_loss_single_level_debug}
    print("INFO (evaluate_sdl): Кастомная функция потерь УСПЕШНО импортирована.")
except ImportError as e_loss_imp:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (evaluate_sdl): Не удалось импортировать compute_detector_loss_single_level_P4_debug: {e_loss_imp}")
    print(
        "                   Модель будет загружаться с compile=False или без custom_objects, если потеря не стандартная.")

SDL_DATA_PARAMS_LOADED_FOR_EVAL = False
try:
    from datasets.detector_data_loader_single_level_debug import (
        parse_xml_annotation,
        ANCHORS_WH_P4_DEBUG_SDL_G as ANCHORS_FOR_DECODING,
        GRID_H_P4_DEBUG_SDL_G as GRID_H_FOR_DECODING,
        GRID_W_P4_DEBUG_SDL_G as GRID_W_FOR_DECODING,
        CLASSES_LIST_SDL_G as DET_CLASSES_LIST_EVAL,
        NUM_CLASSES_SDL_G as DET_NUM_CLASSES_EVAL,
        TARGET_IMG_HEIGHT_SDL_G as DET_TARGET_IMG_HEIGHT_EVAL,
        TARGET_IMG_WIDTH_SDL_G as DET_TARGET_IMG_WIDTH_EVAL
        # NETWORK_STRIDE_SDL_P4 - если он там есть и нужен
    )

    SDL_DATA_PARAMS_LOADED_FOR_EVAL = True
    print("INFO (evaluate_sdl): Параметры из detector_data_loader_single_level_debug УСПЕШНО импортированы.")
except ImportError as e_sdl_imp_eval:
    print(
        f"ОШИБКА (evaluate_sdl): Не удалось импортировать параметры из detector_data_loader_single_level_debug: {e_sdl_imp_eval}")
    DET_TARGET_IMG_HEIGHT_EVAL, DET_TARGET_IMG_WIDTH_EVAL = 416, 416
    DET_CLASSES_LIST_EVAL = ['pit', 'crack'];
    DET_NUM_CLASSES_EVAL = 2
    ANCHORS_FOR_DECODING = np.array([[0.15, 0.15]] * 3, dtype=np.float32)
    GRID_H_FOR_DECODING, GRID_W_FOR_DECODING = 26, 26


# --- Загрузка Конфигураций ---
def load_config_eval_sl(config_path_obj, config_name_str):
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(f"ОШИБКА: Файл {config_path_obj.name} пуст или имеет неверный формат для {config_name_str}.")
            # Возвращаем пустой словарь, чтобы основной код мог использовать .get() с дефолтами
            return {}  # ИЗМЕНЕНО: не выходим, а возвращаем пустой словарь
        print(f"INFO: Конфиг '{config_name_str}' успешно загружен из {config_path_obj.name}.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден по пути: {config_path_obj} (для {config_name_str}).")
        return {}  # ИЗМЕНЕНО
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении {config_path_obj.name}: {e} (для {config_name_str}).")
        return {}  # ИЗМЕНЕНО


print("\n--- Загрузка конфигурационных файлов для evaluate_detector_single_level_debug.py ---")
_base_config_path_obj_eval_sl = _src_path_eval_sl / 'configs' / 'base_config.yaml'
_detector_config_single_level_path_obj_eval_sl = _src_path_eval_sl / 'configs' / 'detector_config_single_level_debug.yaml'

BASE_CONFIG_EVAL_SL = load_config_eval_sl(_base_config_path_obj_eval_sl, "Base Config (Eval SDL)")
DETECTOR_SINGLE_LEVEL_CONFIG_EVAL = load_config_eval_sl(_detector_config_single_level_path_obj_eval_sl,
                                                        "Detector Single Level Debug Config (Eval SDL)")

# --- Параметры из Конфигов для Путей и Инференса ---
_images_subdir_name_eval_sl = BASE_CONFIG_EVAL_SL.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_eval_sl = BASE_CONFIG_EVAL_SL.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel_eval_sl = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_EVAL_SL = (_project_root_eval_sl / _detector_dataset_ready_path_rel_eval_sl).resolve()
VAL_IMAGE_DIR_EVAL_SL = str(DETECTOR_DATASET_READY_ABS_EVAL_SL / "validation" / _images_subdir_name_eval_sl)
VAL_ANNOT_DIR_EVAL_SL = str(DETECTOR_DATASET_READY_ABS_EVAL_SL / "validation" / _annotations_subdir_name_eval_sl)

PREDICT_PARAMS_EVAL_SL = DETECTOR_SINGLE_LEVEL_CONFIG_EVAL.get('predict_params', {})  # Берем из отладочного конфига
DEFAULT_CONF_THRESH_EVAL_SL = PREDICT_PARAMS_EVAL_SL.get("confidence_threshold", 0.25)
DEFAULT_IOU_THRESH_NMS_EVAL_SL = PREDICT_PARAMS_EVAL_SL.get("iou_threshold", 0.45)
DEFAULT_MAX_DETS_EVAL_SL = PREDICT_PARAMS_EVAL_SL.get("max_detections", 100)  # ИСПРАВЛЕНО ИМЯ
DEFAULT_MODEL_PATH_EVAL_SL = DETECTOR_SINGLE_LEVEL_CONFIG_EVAL.get("detector_model_path",
                                                                   "weights/model_placeholder_sdl.keras")

IOU_THRESH_FOR_MATCHING_EVAL_SL = 0.5


# --- Вспомогательные Функции ---
# ИСПРАВЛЕНО: Копируем сюда определение collect_split_data_paths
def collect_split_data_paths(split_dir_abs_path_str, images_subdir_name, annotations_subdir_name):
    image_paths = []
    xml_paths = []
    # Преобразуем в Path, если это строка
    split_dir_abs_path = Path(split_dir_abs_path_str)
    current_images_dir = split_dir_abs_path / images_subdir_name
    current_annotations_dir = split_dir_abs_path / annotations_subdir_name

    if not current_images_dir.is_dir() or not current_annotations_dir.is_dir():
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ (evaluate_sdl): Директория {current_images_dir} или {current_annotations_dir} не найдена. Не удастся собрать пути.")
        return image_paths, xml_paths

    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files_in_split = []
    for ext in valid_extensions:
        image_files_in_split.extend(list(current_images_dir.glob(f"*{ext.lower()}")))
        image_files_in_split.extend(list(current_images_dir.glob(f"*{ext.upper()}")))

    image_files_in_split = sorted(list(set(image_files_in_split)))  # Убираем дубликаты и сортируем

    for img_path_obj in image_files_in_split:
        base_name = img_path_obj.stem
        xml_path_obj = current_annotations_dir / (base_name + ".xml")
        if xml_path_obj.exists():
            image_paths.append(str(img_path_obj))
            xml_paths.append(str(xml_path_obj))
    return image_paths, xml_paths


def preprocess_image_for_eval(image_bgr, target_height, target_width):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_single_level_predictions(raw_predictions_tensor):
    # Используем глобальные переменные, импортированные из detector_data_loader_single_level_debug
    # или их дефолты, если импорт не удался
    batch_size = tf.shape(raw_predictions_tensor)[0]
    pred_xy_raw = raw_predictions_tensor[..., 0:2]
    pred_wh_raw = raw_predictions_tensor[..., 2:4]
    pred_obj_logit = raw_predictions_tensor[..., 4:5]
    pred_class_logits = raw_predictions_tensor[..., 5:]
    gy_indices = tf.tile(tf.range(GRID_H_FOR_DECODING, dtype=tf.float32)[:, tf.newaxis], [1, GRID_W_FOR_DECODING])
    gx_indices = tf.tile(tf.range(GRID_W_FOR_DECODING, dtype=tf.float32)[tf.newaxis, :], [GRID_H_FOR_DECODING, 1])
    grid_coords_xy = tf.stack([gx_indices, gy_indices], axis=-1)
    grid_coords_xy = grid_coords_xy[tf.newaxis, :, :, tf.newaxis, :]
    grid_coords_xy = tf.tile(grid_coords_xy, [batch_size, 1, 1, ANCHORS_FOR_DECODING.shape[0], 1])
    pred_xy_on_grid = (tf.sigmoid(pred_xy_raw) + grid_coords_xy)
    pred_xy_normalized = pred_xy_on_grid / tf.constant([GRID_W_FOR_DECODING, GRID_H_FOR_DECODING], dtype=tf.float32)
    anchors_tensor = tf.constant(ANCHORS_FOR_DECODING, dtype=tf.float32)
    anchors_reshaped = anchors_tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pred_wh_normalized = (tf.exp(pred_wh_raw) * anchors_reshaped)
    decoded_boxes_xywh_norm = tf.concat([pred_xy_normalized, pred_wh_normalized], axis=-1)
    pred_obj_confidence = tf.sigmoid(pred_obj_logit)
    pred_class_probs = tf.sigmoid(pred_class_logits)
    return decoded_boxes_xywh_norm, pred_obj_confidence, pred_class_probs


def apply_nms_and_filter_single_level(decoded_boxes_xywh_norm, obj_confidence, class_probs,
                                      confidence_threshold_arg, iou_threshold_arg, max_detections_arg):
    batch_size = tf.shape(decoded_boxes_xywh_norm)[0]
    num_total_boxes_per_image = GRID_H_FOR_DECODING * GRID_W_FOR_DECODING * ANCHORS_FOR_DECODING.shape[0]
    boxes_flat_xywh = tf.reshape(decoded_boxes_xywh_norm, [batch_size, num_total_boxes_per_image, 4])
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        boxes_flat_xywh[..., 1:2] - boxes_flat_xywh[..., 3:4] / 2.0,
        boxes_flat_xywh[..., 0:1] - boxes_flat_xywh[..., 2:3] / 2.0,
        boxes_flat_xywh[..., 1:2] + boxes_flat_xywh[..., 3:4] / 2.0,
        boxes_flat_xywh[..., 0:1] + boxes_flat_xywh[..., 2:3] / 2.0
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax, 0.0, 1.0)
    obj_conf_flat = tf.reshape(obj_confidence, [batch_size, num_total_boxes_per_image, 1])
    class_probs_flat = tf.reshape(class_probs, [batch_size, num_total_boxes_per_image, DET_NUM_CLASSES_EVAL])
    final_scores_per_class = obj_conf_flat * class_probs_flat
    boxes_for_nms = tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2)

    # ИСПРАВЛЕНО: Используем max_detections_arg вместо глобальной переменной
    max_output_size_final = max_detections_arg // DET_NUM_CLASSES_EVAL if DET_NUM_CLASSES_EVAL > 0 else max_detections_arg
    if max_output_size_final == 0: max_output_size_final = 1  # Защита от нуля

    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms, scores=final_scores_per_class,
        max_output_size_per_class=max_output_size_final,
        max_total_size=max_detections_arg,  # ИСПРАВЛЕНО
        iou_threshold=iou_threshold_arg, score_threshold=confidence_threshold_arg,
        clip_boxes=False
    )
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def calculate_iou_for_eval(box1_xyxy, box2_xyxy):
    x1_i = max(box1_xyxy[0], box2_xyxy[0]);
    y1_i = max(box1_xyxy[1], box2_xyxy[1])
    x2_i = min(box1_xyxy[2], box2_xyxy[2]);
    y2_i = min(box1_xyxy[3], box2_xyxy[3])
    iw = max(0, x2_i - x1_i);
    ih = max(0, y2_i - y1_i);
    ia = iw * ih
    b1a = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    b2a = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    ua = b1a + b2a - ia;
    return ia / (ua + 1e-6)


# --- Основная функция Оценки ---
def evaluate_single_level_detector(model_path_arg_fn, conf_thresh_fn_arg, iou_thresh_nms_fn_arg,
                                   iou_thresh_match_fn_arg, max_dets_fn_arg):
    if not SDL_DATA_PARAMS_LOADED_FOR_EVAL:
        print("ОШИБКА: Параметры из detector_data_loader_single_level_debug не были загружены. Оценка невозможна.")
        return

    model_full_path_fn = (_project_root_eval_sl / model_path_arg_fn).resolve()
    print(f"\n--- Оценка Одноуровневой Модели Детектора ---")
    print(f"Загрузка модели из: {model_full_path_fn}")
    if not model_full_path_fn.exists():
        print(f"ОШИБКА: Файл модели не найден: {model_full_path_fn}")
        return
    try:
        model = tf.keras.models.load_model(str(model_full_path_fn), custom_objects=CUSTOM_OBJECTS_EVAL_SL,
                                           compile=False)
        print("Модель успешно загружена.")
        # Проверка формы выхода (опционально, но полезно)
        # ...
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}");
        return

    # ИСПРАВЛЕНО: Используем Path объекты для VAL_IMAGE_DIR_EVAL_SL и VAL_ANNOT_DIR_EVAL_SL
    val_image_paths, val_xml_paths = collect_split_data_paths(
        Path(VAL_IMAGE_DIR_EVAL_SL).parent,  # Передаем data/Detector_Dataset_Ready/validation
        _images_subdir_name_eval_sl,
        _annotations_subdir_name_eval_sl
    )
    if not val_image_paths: print(f"Изображения не найдены в {VAL_IMAGE_DIR_EVAL_SL}"); return

    print(f"\nНайдено {len(val_image_paths)} изображений в валидационной выборке для оценки.")
    print(f"  Порог уверенности (conf_thresh) для детекций: {conf_thresh_fn_arg}")
    print(f"  Порог IoU для NMS: {iou_thresh_nms_fn_arg}")
    print(f"  Порог IoU для сопоставления TP/FP: {iou_thresh_match_fn_arg}")
    print(f"  Максимальное количество детекций: {max_dets_fn_arg}")

    true_positives = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)
    false_positives = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)
    num_gt_objects_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)
    processing_times = []

    for img_idx, image_path_str in enumerate(val_image_paths):
        xml_path_str = val_xml_paths[img_idx]
        if (img_idx + 1) % 50 == 0 or img_idx == 0 or img_idx == len(val_image_paths) - 1:
            print(f"  Обработка изображения {img_idx + 1}/{len(val_image_paths)}: {os.path.basename(image_path_str)}")

        original_bgr_image = cv2.imread(image_path_str)
        if original_bgr_image is None: print(f"    Пропуск: не удалось загрузить {image_path_str}"); continue
        original_h, original_w = original_bgr_image.shape[:2]

        gt_objects, _, _, _ = parse_xml_annotation(xml_path_str)
        if gt_objects is None: print(f"    Пропуск: ошибка парсинга XML {xml_path_str}"); continue

        gt_boxes_for_eval_current_img = []
        for gt_obj in gt_objects:
            num_gt_objects_per_class[int(gt_obj['class_id'])] += 1
            gt_boxes_for_eval_current_img.append([
                int(gt_obj['xmin']), int(gt_obj['ymin']), int(gt_obj['xmax']), int(gt_obj['ymax']),
                int(gt_obj['class_id']), False
            ])

        start_time_inf = time.time()
        detector_input_batch = preprocess_image_for_eval(original_bgr_image, DET_TARGET_IMG_HEIGHT_EVAL,
                                                         DET_TARGET_IMG_WIDTH_EVAL)
        raw_preds_eval = model.predict(detector_input_batch, verbose=0)

        decoded_boxes_xywh_n_eval, obj_conf_n_eval, class_probs_n_eval = decode_single_level_predictions(raw_preds_eval)

        nms_boxes_n_eval, nms_scores_n_eval, nms_classes_n_eval, num_valid_dets_n_eval = apply_nms_and_filter_single_level(
            decoded_boxes_xywh_n_eval, obj_conf_n_eval, class_probs_n_eval,
            conf_thresh_fn_arg, iou_thresh_nms_fn_arg, max_dets_fn_arg  # ИСПРАВЛЕНО: передаем max_dets_fn_arg
        )
        processing_times.append(time.time() - start_time_inf)

        num_detections_eval = int(num_valid_dets_n_eval[0].numpy())
        pred_boxes_norm_yx_eval = nms_boxes_n_eval[0][:num_detections_eval].numpy()
        pred_scores_eval = nms_scores_n_eval[0][:num_detections_eval].numpy()
        pred_class_ids_eval = nms_classes_n_eval[0][:num_detections_eval].numpy().astype(int)

        if num_detections_eval > 0:
            sorted_indices_eval = np.argsort(pred_scores_eval)[::-1]
            pred_boxes_norm_yx_eval = pred_boxes_norm_yx_eval[sorted_indices_eval]
            # pred_scores_eval = pred_scores_eval[sorted_indices_eval] # Уже отсортированы по NMS
            pred_class_ids_eval = pred_class_ids_eval[sorted_indices_eval]

        for i_pred_eval in range(num_detections_eval):
            pred_box_norm_yx = pred_boxes_norm_yx_eval[i_pred_eval]
            pred_box_px_xyxy = [
                int(pred_box_norm_yx[1] * original_w), int(pred_box_norm_yx[0] * original_h),
                int(pred_box_norm_yx[3] * original_w), int(pred_box_norm_yx[2] * original_h)]
            pred_cls_id_eval = pred_class_ids_eval[i_pred_eval]

            best_iou_val = 0.0;
            best_gt_idx_val = -1
            # ИСПРАВЛЕНО: переменная цикла для GT
            for i_gt_current, gt_data_eval in enumerate(gt_boxes_for_eval_current_img):
                gt_box_px_eval = gt_data_eval[0:4];
                gt_cls_id_eval = gt_data_eval[4];
                gt_matched_eval = gt_data_eval[5]
                if gt_cls_id_eval == pred_cls_id_eval and not gt_matched_eval:
                    iou_val = calculate_iou_for_eval(pred_box_px_xyxy, gt_box_px_eval)
                    if iou_val > best_iou_val: best_iou_val = iou_val; best_gt_idx_val = i_gt_current  # ИСПРАВЛЕНО

            if best_iou_val >= iou_thresh_match_fn_arg:
                if best_gt_idx_val != -1:  # Убедимся, что совпадение найдено
                    true_positives[pred_cls_id_eval] += 1
                    gt_boxes_for_eval_current_img[best_gt_idx_val][5] = True
                else:  # Это не должно происходить, если best_iou_val > 0, но для безопасности
                    false_positives[pred_cls_id_eval] += 1
            else:
                false_positives[pred_cls_id_eval] += 1

    # Расчет финальных метрик
    print("\n--- Результаты Оценки Одноуровневой Модели ---")
    # ... (остальной код расчета и вывода метрик как был) ...
    precision_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.float32)  # Указал dtype
    recall_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.float32)
    f1_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.float32)

    for i_cls_eval_final in range(DET_NUM_CLASSES_EVAL):
        class_name_final = DET_CLASSES_LIST_EVAL[i_cls_eval_final]
        tp_final = true_positives[i_cls_eval_final]
        fp_final = false_positives[i_cls_eval_final]
        fn_final = num_gt_objects_per_class[i_cls_eval_final] - tp_final

        precision_final = tp_final / (tp_final + fp_final + 1e-6)
        recall_final = tp_final / (num_gt_objects_per_class[i_cls_eval_final] + 1e-6)
        f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final + 1e-6)

        precision_per_class[i_cls_eval_final] = precision_final
        recall_per_class[i_cls_eval_final] = recall_final
        f1_per_class[i_cls_eval_final] = f1_final

        print(f"\nКласс: {class_name_final}")
        print(f"  Всего Ground Truth объектов: {int(num_gt_objects_per_class[i_cls_eval_final])}")
        print(f"  True Positives (TP): {int(tp_final)}")
        print(f"  False Positives (FP): {int(fp_final)}")
        print(f"  False Negatives (FN): {int(fn_final)}")
        print(f"  Precision: {precision_final:.4f}")
        print(f"  Recall: {recall_final:.4f}")
        print(f"  F1-score: {f1_final:.4f}")

    if processing_times:
        avg_time = np.mean(processing_times);
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nСреднее время обработки одного изображения: {avg_time:.4f} сек ({fps:.2f} FPS)")

    macro_precision_final = np.mean(precision_per_class);
    macro_recall_final = np.mean(recall_per_class);
    macro_f1_final = np.mean(f1_per_class)
    print("\n--- Макро-усредненные метрики ---")
    print(f"  Macro Precision: {macro_precision_final:.4f}");
    print(f"  Macro Recall: {macro_recall_final:.4f}");
    print(f"  Macro F1-score: {macro_f1_final:.4f}")

    total_tp_all_final = np.sum(true_positives);
    total_fp_all_final = np.sum(false_positives);
    total_gt_all_final = np.sum(num_gt_objects_per_class)
    micro_precision_final = total_tp_all_final / (total_tp_all_final + total_fp_all_final + 1e-6)
    micro_recall_final = total_tp_all_final / (total_gt_all_final + 1e-6)
    micro_f1_final = 2 * (micro_precision_final * micro_recall_final) / (
                micro_precision_final + micro_recall_final + 1e-6)
    print("\n--- Микро-усредненные метрики ---")
    print(f"  Micro Precision: {micro_precision_final:.4f}");
    print(f"  Micro Recall: {micro_recall_final:.4f}");
    print(f"  Micro F1-score: {micro_f1_final:.4f}")
    print("\nОценка одноуровневой модели завершена.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка одноуровневой модели детектора.")
    parser.add_argument("--model_path", type=str, default=str(Path(DEFAULT_MODEL_PATH_EVAL_SL)),  # Конвертируем в str
                        help=f"Путь к обученной модели (.keras). Дефолт: {DEFAULT_MODEL_PATH_EVAL_SL}")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONF_THRESH_EVAL_SL,
                        help=f"Порог уверенности для NMS. Дефолт: {DEFAULT_CONF_THRESH_EVAL_SL}")
    parser.add_argument("--iou_thresh_nms", type=float, default=DEFAULT_IOU_THRESH_NMS_EVAL_SL,
                        help=f"Порог IoU для NMS. Дефолт: {DEFAULT_IOU_THRESH_NMS_EVAL_SL}")
    parser.add_argument("--iou_thresh_match", type=float, default=IOU_THRESH_FOR_MATCHING_EVAL_SL,
                        help=f"Порог IoU для сопоставления предсказаний с GT. Дефолт: {IOU_THRESH_FOR_MATCHING_EVAL_SL}")
    parser.add_argument("--max_dets", type=int, default=DEFAULT_MAX_DETS_EVAL_SL,  # ИСПРАВЛЕНО ИМЯ АРГУМЕНТА
                        help=f"Макс. детекций после NMS. Дефолт: {DEFAULT_MAX_DETS_EVAL_SL}")

    args_eval_sl_main = parser.parse_args()

    if not SDL_DATA_PARAMS_LOADED_FOR_EVAL:
        print("Выход из evaluate_detector_single_level_debug.py из-за ошибки загрузки параметров данных.")
    else:
        evaluate_single_level_detector(
            args_eval_sl_main.model_path,
            args_eval_sl_main.conf_thresh,
            args_eval_sl_main.iou_thresh_nms,
            args_eval_sl_main.iou_thresh_match,
            args_eval_sl_main.max_dets  # ИСПРАВЛЕНО: передаем правильный аргумент
        )