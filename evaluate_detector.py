# RoadDefectDetector/evaluate_detector.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import glob
from pathlib import Path
import time  # Хотя в этом скрипте время инференса не так критично, как в predict

# --- Добавляем src в sys.path ---
_project_root_eval = Path(__file__).parent.resolve()
_src_path_eval = _project_root_eval / 'src'
import sys

if str(_src_path_eval) not in sys.path:
    sys.path.insert(0, str(_src_path_eval))

# --- Импорты из твоих модулей ---
try:
    from losses.detection_losses import compute_detector_loss_v1

    CUSTOM_OBJECTS_EVAL = {'compute_detector_loss_v1': compute_detector_loss_v1}
    print("INFO (evaluate_detector.py): Кастомная функция потерь для детектора загружена.")
except ImportError:
    CUSTOM_OBJECTS_EVAL = {}
    print("ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Кастомная функция потерь не найдена.")

# Загружаем функции из predict_detector.py, так как они нам нужны для инференса
# Предполагаем, что predict_detector.py находится в КОРНЕ проекта
_predict_detector_script_path = _project_root_eval / "predict_detector.py"  # ИЛИ predict_pipeline.py
# Чтобы импортировать из него, нужно временно добавить корень проекта в sys.path, если еще не там
if str(_project_root_eval) not in sys.path:
    sys.path.insert(0, str(_project_root_eval))

try:
    # Если predict_detector.py - это имя твоего скрипта инференса
    from predict_detector import (
        preprocess_image_for_model,
        decode_predictions,
        apply_nms_and_filter
    )

    # Если он называется predict_pipeline.py, измени имя выше
    PREDICT_FUNCS_LOADED = True
    print("INFO (evaluate_detector.py): Функции инференса успешно импортированы.")
except ImportError as e_imp_pred:
    print(f"ОШИБКА: Не удалось импортировать функции из predict_detector.py: {e_imp_pred}")
    PREDICT_FUNCS_LOADED = False


    # Заглушки, чтобы скрипт не падал сразу
    def preprocess_image_for_model(i, h, w):
        return None


    def decode_predictions(r, a, gh, gw, nc, s):
        return None, None, None


    def apply_nms_and_filter(b, o, c, gh, gw, na, nc, ct, it, md):
        return None, None, None, tf.constant([0])

from datasets.detector_data_loader import parse_xml_annotation


# --- Загрузка Конфигураций ---
def load_config_eval_strict(config_path_obj, config_name):
    """Загружает конфиг, выходит из программы при ошибке."""
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:  # Проверка, что не пустой
            print(f"ОШИБКА: {config_path_obj.name} пуст или имеет неверный формат.")
            exit()
        print(f"INFO: Конфиг {config_name} успешно загружен.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден по пути: {config_path_obj}.")
        exit()
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении {config_path_obj.name}: {e}.")
        exit()


_base_config_path_obj_eval = _src_path_eval / 'configs' / 'base_config.yaml'
_detector_config_path_obj_eval = _src_path_eval / 'configs' / 'detector_config.yaml'
_predict_config_path_obj_eval = _src_path_eval / 'configs' / 'predict_config.yaml'

print("--- Загрузка конфигурационных файлов ---")
BASE_CONFIG_EVAL = load_config_eval_strict(_base_config_path_obj_eval, "Base Config")
DETECTOR_CONFIG_EVAL = load_config_eval_strict(_detector_config_path_obj_eval, "Detector Config")
PREDICT_CONFIG_EVAL = load_config_eval_strict(_predict_config_path_obj_eval, "Predict Config")

# --- Параметры из Конфигов ---
# Для модели детектора и данных
DET_INPUT_SHAPE_EVAL = tuple(DETECTOR_CONFIG_EVAL['input_shape'])
DET_TARGET_IMG_HEIGHT_EVAL, DET_TARGET_IMG_WIDTH_EVAL = DET_INPUT_SHAPE_EVAL[0], DET_INPUT_SHAPE_EVAL[1]
DET_CLASSES_LIST_EVAL = DETECTOR_CONFIG_EVAL['classes']
DET_NUM_CLASSES_EVAL = len(DET_CLASSES_LIST_EVAL)
DET_ANCHORS_WH_NORM_EVAL = np.array(DETECTOR_CONFIG_EVAL['anchors_wh_normalized'], dtype=np.float32)
DET_NUM_ANCHORS_EVAL = DETECTOR_CONFIG_EVAL['num_anchors_per_location']
DET_NETWORK_STRIDE_EVAL = 16
DET_GRID_HEIGHT_EVAL = DET_TARGET_IMG_HEIGHT_EVAL // DET_NETWORK_STRIDE_EVAL
DET_GRID_WIDTH_EVAL = DET_TARGET_IMG_WIDTH_EVAL // DET_NETWORK_STRIDE_EVAL

# Пути к данным из base_config
_images_subdir_name_eval = BASE_CONFIG_EVAL['dataset']['images_dir']
_annotations_subdir_name_eval = BASE_CONFIG_EVAL['dataset']['annotations_dir']

# Пути к разделенному датасету для ВАЛИДАЦИИ
_detector_dataset_ready_path_rel_eval = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_EVAL = (_project_root_eval / _detector_dataset_ready_path_rel_eval).resolve()
VAL_IMAGE_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _images_subdir_name_eval)
VAL_ANNOT_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _annotations_subdir_name_eval)

# Параметры для инференса из predict_config.yaml
DETECTOR_MODEL_PATH_EVAL = PREDICT_CONFIG_EVAL.get("detector_model_path",
                                                   "weights/detector_v1_best_val_loss.keras")  # Дефолт, если вдруг нет
CONF_THRESH_EVAL = PREDICT_CONFIG_EVAL.get("default_conf_thresh", 0.25)
IOU_THRESH_NMS_EVAL = PREDICT_CONFIG_EVAL.get("default_iou_thresh", 0.45)  # Это для NMS при предсказании
MAX_DETS_EVAL = PREDICT_CONFIG_EVAL.get("default_max_dets", 100)

# Порог IoU для сопоставления предсказания с GT при оценке (TP/FP)
IOU_THRESH_FOR_MATCHING_EVAL = 0.5  # Стандартный порог для оценки


def calculate_iou_for_eval(box1_xyxy, box2_xyxy):
    # ... (Код этой функции остается таким же) ...
    x1_inter = max(box1_xyxy[0], box2_xyxy[0]);
    y1_inter = max(box1_xyxy[1], box2_xyxy[1])
    x2_inter = min(box1_xyxy[2], box2_xyxy[2]);
    y2_inter = min(box1_xyxy[3], box2_xyxy[3])
    inter_width = max(0, x2_inter - x1_inter);
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / (union_area + 1e-6)


def evaluate_detector_on_validation_set():
    """
    Оценивает модель детектора на валидационной выборке, используя параметры из конфигов.
    """
    if not PREDICT_FUNCS_LOADED:
        print("ОШИБКА: Не удалось загрузить необходимые функции для инференса. Оценка невозможна.")
        return

    # 1. Загрузка модели
    model_full_path_eval = (_project_root_eval / DETECTOR_MODEL_PATH_EVAL).resolve()
    print(f"\n--- Оценка Модели Детектора ---")
    print(f"Загрузка модели из: {model_full_path_eval}")
    if not model_full_path_eval.exists():
        print(f"ОШИБКА: Файл модели не найден: {model_full_path_eval}")
        print(f"Проверьте путь 'detector_model_path' в 'src/configs/predict_config.yaml'.")
        return
    try:
        model = tf.keras.models.load_model(str(model_full_path_eval), custom_objects=CUSTOM_OBJECTS_EVAL, compile=False)
        print("Модель детектора успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели детектора: {e}");
        return

    # 2. Сбор путей к валидационным данным
    if not Path(VAL_IMAGE_DIR_EVAL).is_dir() or not Path(VAL_ANNOT_DIR_EVAL).is_dir():
        print(f"ОШИБКА: Директории валидационных данных не найдены:")
        print(f"  Изображения: {VAL_IMAGE_DIR_EVAL}")
        print(f"  Аннотации: {VAL_ANNOT_DIR_EVAL}")
        print(
            "Убедитесь, что 'create_data_splits.py' был запущен и пути к 'data/Detector_Dataset_Ready/validation/' корректны.")
        return

    val_image_paths = sorted(list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpg")) + \
                             list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpeg")) + \
                             list(Path(VAL_IMAGE_DIR_EVAL).glob("*.png")))

    if not val_image_paths:
        print(f"Изображения не найдены в валидационной директории: {VAL_IMAGE_DIR_EVAL}")
        return
    print(f"Найдено {len(val_image_paths)} изображений в валидационной выборке для оценки.")
    print(f"Используемый порог уверенности (conf_thresh) для детекций: {CONF_THRESH_EVAL}")
    print(f"Используемый порог IoU для NMS: {IOU_THRESH_NMS_EVAL}")
    print(f"Используемый порог IoU для сопоставления TP/FP: {IOU_THRESH_FOR_MATCHING_EVAL}")

    true_positives = np.zeros(DET_NUM_CLASSES_EVAL)
    false_positives = np.zeros(DET_NUM_CLASSES_EVAL)
    num_gt_objects_per_class = np.zeros(DET_NUM_CLASSES_EVAL)  # Общее число GT объектов каждого класса

    # 3. Итерация по валидационным изображениям
    for img_idx, image_path_obj in enumerate(val_image_paths):
        image_path_str = str(image_path_obj)
        # print(f"\nОбработка изображения {img_idx+1}/{len(val_image_paths)}: {image_path_obj.name}") # Можно раскомментировать для детального лога

        original_bgr_image = cv2.imread(image_path_str)
        if original_bgr_image is None: continue
        original_h, original_w = original_bgr_image.shape[:2]

        xml_filename = image_path_obj.stem + ".xml"
        xml_path_str = str(Path(VAL_ANNOT_DIR_EVAL) / xml_filename)

        gt_objects_data, _, _, _ = parse_xml_annotation(xml_path_str, DET_CLASSES_LIST_EVAL)
        if gt_objects_data is None: continue  # Пропускаем, если ошибка парсинга XML

        gt_boxes_for_eval = []  # Список [[xmin, ymin, xmax, ymax, class_id, matched_flag]]
        for gt_obj in gt_objects_data:
            num_gt_objects_per_class[int(gt_obj['class_id'])] += 1
            gt_boxes_for_eval.append([
                int(gt_obj['xmin']), int(gt_obj['ymin']), int(gt_obj['xmax']), int(gt_obj['ymax']),
                int(gt_obj['class_id']), False
            ])

        detector_input_batch = preprocess_image_for_model(original_bgr_image, DET_TARGET_IMG_HEIGHT_EVAL,
                                                          DET_TARGET_IMG_WIDTH_EVAL)
        raw_preds = model.predict(detector_input_batch, verbose=0)

        decoded_boxes_xywh_n, obj_conf_n, class_probs_n = decode_predictions(
            raw_preds, DET_ANCHORS_WH_NORM_EVAL, DET_GRID_HEIGHT_EVAL, DET_GRID_WIDTH_EVAL,
            DET_NUM_CLASSES_EVAL, DET_NETWORK_STRIDE_EVAL)

        nms_boxes_n, nms_scores_n, nms_classes_n, num_valid_dets_n = apply_nms_and_filter(
            decoded_boxes_xywh_n, obj_conf_n, class_probs_n,
            DET_GRID_HEIGHT_EVAL, DET_GRID_WIDTH_EVAL, DET_NUM_ANCHORS_EVAL, DET_NUM_CLASSES_EVAL,
            confidence_threshold=CONF_THRESH_EVAL, iou_threshold=IOU_THRESH_NMS_EVAL, max_detections=MAX_DETS_EVAL
        )

        num_detections = int(num_valid_dets_n[0].numpy())

        pred_boxes_norm_yminxminymaxxmax = nms_boxes_n[0][:num_detections].numpy()
        pred_scores = nms_scores_n[0][:num_detections].numpy()
        pred_class_ids = nms_classes_n[0][:num_detections].numpy().astype(int)

        # Сопоставление
        # Сортируем предсказания по убыванию уверенности (важно для правильного расчета AP, если будем его делать)
        # Но для простого TP/FP/FN это не строго обязательно, но хорошая практика
        if num_detections > 0:
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes_norm_yminxminymaxxmax = pred_boxes_norm_yminxminymaxxmax[sorted_indices]
            pred_scores = pred_scores[sorted_indices]
            pred_class_ids = pred_class_ids[sorted_indices]

        for i_pred in range(num_detections):
            pred_box_norm = pred_boxes_norm_yminxminymaxxmax[i_pred]
            pred_box_px = [  # Конвертируем в пиксельный [xmin, ymin, xmax, ymax]
                int(pred_box_norm[1] * original_w), int(pred_box_norm[0] * original_h),
                int(pred_box_norm[3] * original_w), int(pred_box_norm[2] * original_h)]
            pred_cls_id = pred_class_ids[i_pred]

            best_iou_for_pred = 0.0
            best_gt_match_idx = -1

            for i_gt, gt_data in enumerate(gt_boxes_for_eval):
                gt_box_px = gt_data[0:4]
                gt_cls_id = gt_data[4]
                gt_matched_flag = gt_data[5]

                if gt_cls_id == pred_cls_id and not gt_matched_flag:
                    iou = calculate_iou_for_eval(pred_box_px, gt_box_px)
                    if iou > best_iou_for_pred:
                        best_iou_for_pred = iou
                        best_gt_match_idx = i_gt

            if best_iou_for_pred >= IOU_THRESH_FOR_MATCHING_EVAL:
                true_positives[pred_cls_id] += 1
                gt_boxes_for_eval[best_gt_match_idx][5] = True  # Помечаем GT как сопоставленный
            else:
                false_positives[pred_cls_id] += 1

    # 4. Расчет финальных метрик
    print("\n--- Результаты Оценки ---")
    precision_per_class = np.zeros(DET_NUM_CLASSES_EVAL)
    recall_per_class = np.zeros(DET_NUM_CLASSES_EVAL)
    f1_per_class = np.zeros(DET_NUM_CLASSES_EVAL)

    for i in range(DET_NUM_CLASSES_EVAL):
        class_name = DET_CLASSES_LIST_EVAL[i]
        tp = true_positives[i]
        fp = false_positives[i]
        # FN = общее количество GT этого класса - TP для этого класса
        fn = num_gt_objects_per_class[i] - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (num_gt_objects_per_class[i] + 1e-6)  # Используем общее число GT для знаменателя recall
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        precision_per_class[i] = precision;
        recall_per_class[i] = recall;
        f1_per_class[i] = f1

        print(f"\nКласс: {class_name}")
        print(f"  Всего Ground Truth объектов: {int(num_gt_objects_per_class[i])}")
        print(f"  True Positives (TP): {int(tp)}")
        print(f"  False Positives (FP): {int(fp)}")
        print(f"  False Negatives (FN): {int(fn)}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

    macro_precision = np.mean(precision_per_class);
    macro_recall = np.mean(recall_per_class);
    macro_f1 = np.mean(f1_per_class)
    print("\n--- Макро-усредненные метрики ---")
    print(f"  Macro Precision: {macro_precision:.4f}");
    print(f"  Macro Recall: {macro_recall:.4f}");
    print(f"  Macro F1-score: {macro_f1:.4f}")

    total_tp_all = np.sum(true_positives);
    total_fp_all = np.sum(false_positives);
    total_gt_all = np.sum(num_gt_objects_per_class)
    micro_precision = total_tp_all / (total_tp_all + total_fp_all + 1e-6)
    micro_recall = total_tp_all / (total_gt_all + 1e-6)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-6)
    print("\n--- Микро-усредненные метрики ---")
    print(f"  Micro Precision: {micro_precision:.4f}");
    print(f"  Micro Recall: {micro_recall:.4f}");
    print(f"  Micro F1-score: {micro_f1:.4f}")
    print("\nОценка завершена.")


if __name__ == "__main__":
    if not PREDICT_FUNCS_LOADED:
        print("Выход из evaluate_detector.py из-за ошибки импорта функций предсказания.")
    else:
        evaluate_detector_on_validation_set()