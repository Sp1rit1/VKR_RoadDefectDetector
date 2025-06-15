# RoadDefectDetector/evaluate_detector.py
import argparse

import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import glob
from pathlib import Path
import time
import sys

# --- Добавляем src и корень проекта в sys.path для корректных импортов ---
_project_root_eval = Path(__file__).parent.resolve()
_src_path_eval = _project_root_eval / 'src'

if str(_src_path_eval) not in sys.path:
    sys.path.insert(0, str(_src_path_eval))
if str(_project_root_eval) not in sys.path:  # Для импорта predict_detector
    sys.path.insert(0, str(_project_root_eval))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_EVAL = {}
try:
    # Эта функция потерь нужна только если модель сохранялась с compile=True и этой функцией
    # Если модель сохранена с compile=False или как SavedModel, она может не понадобиться для загрузки
    from losses.detection_losses import compute_detector_loss_v1

    CUSTOM_OBJECTS_EVAL = {'compute_detector_loss_v1': compute_detector_loss_v1}
    print("INFO (evaluate_detector.py): Кастомная функция потерь для детектора ЗАГРУЖЕНА (для custom_objects).")
except ImportError:
    print(
        "ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Кастомная функция потерь не найдена. Модель будет загружаться без нее.")
except Exception as e_loss_imp:
    print(f"ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Ошибка импорта detection_losses: {e_loss_imp}")

PREDICT_FUNCS_LOADED = False
try:
    from predict_detector import (  # ИЛИ predict_pipeline, если имя файла другое
        preprocess_image_for_model,
        decode_predictions,
        apply_nms_and_filter
    )

    PREDICT_FUNCS_LOADED = True
    print("INFO (evaluate_detector.py): Функции инференса успешно импортированы из predict_detector.py.")
except ImportError as e_imp_pred:
    print(f"ОШИБКА: Не удалось импортировать функции из predict_detector.py (или predict_pipeline.py): {e_imp_pred}")
    print("         Убедитесь, что файл predict_detector.py находится в корне проекта и не содержит ошибок импорта.")


    # Заглушки, чтобы скрипт не падал сразу при импорте, но оценка не будет работать
    def preprocess_image_for_model(i, h, w):
        return None


    def decode_predictions(r, a, gh, gw, nc, s):
        return None, None, None


    def apply_nms_and_filter(b, o, c, gh, gw, na, nc, ct, it, md):
        return None, None, None, tf.constant([0])
except Exception as e_pred_other:
    print(f"ОШИБКА: Другая ошибка при импорте из predict_detector.py: {e_pred_other}")
    PREDICT_FUNCS_LOADED = False
    # ... (те же заглушки) ...

try:
    from datasets.detector_data_loader import parse_xml_annotation, \
        CLASSES_LIST_GLOBAL_FOR_DETECTOR as DET_CLASSES_FROM_LOADER

    print(
        f"INFO (evaluate_detector.py): parse_xml_annotation и классы ({DET_CLASSES_FROM_LOADER}) импортированы из detector_data_loader.")
except ImportError as e_imp_data:
    print(f"ОШИБКА: Не удалось импортировать parse_xml_annotation из detector_data_loader: {e_imp_data}")


    def parse_xml_annotation(x, c):
        return None, 0, 0, ""  # Заглушка


    DET_CLASSES_FROM_LOADER = ['class0_fallback', 'class1_fallback']  # Заглушка
except Exception as e_data_other:
    print(f"ОШИБКА: Другая ошибка при импорте из detector_data_loader: {e_data_other}")


    def parse_xml_annotation(x, c):
        return None, 0, 0, ""


    DET_CLASSES_FROM_LOADER = ['class0_fallback', 'class1_fallback']


# --- Загрузка Конфигураций ---
def load_config_eval_strict(config_path_obj, config_name_str):
    """Загружает конфиг, выходит из программы при ошибке."""
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(f"ОШИБКА: Конфиг '{config_name_str}' ({config_path_obj}) пуст или имеет неверный формат. Выход.")
            exit(1)
        print(f"INFO: Конфиг '{config_name_str}' успешно загружен из {config_path_obj.name}.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл конфига '{config_name_str}' не найден по пути: {config_path_obj}. Выход.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении '{config_name_str}' ({config_path_obj.name}): {e}. Выход.")
        exit(1)
    except Exception as e_gen:
        print(f"ОШИБКА при загрузке '{config_name_str}' ({config_path_obj.name}): {e_gen}. Выход.")
        exit(1)


print("\n--- Загрузка конфигурационных файлов для evaluate_detector.py ---")
_base_config_path_obj_eval = _src_path_eval / 'configs' / 'base_config.yaml'
_detector_config_path_obj_eval = _src_path_eval / 'configs' / 'detector_config.yaml'
_predict_config_path_obj_eval = _src_path_eval / 'configs' / 'predict_config.yaml'

BASE_CONFIG_EVAL = load_config_eval_strict(_base_config_path_obj_eval, "Base Config")
DETECTOR_CONFIG_EVAL = load_config_eval_strict(_detector_config_path_obj_eval, "Detector Config")
PREDICT_CONFIG_EVAL = load_config_eval_strict(_predict_config_path_obj_eval, "Predict Config")

# --- Параметры из Конфигов (с проверками) ---
try:
    DET_INPUT_SHAPE_EVAL = tuple(DETECTOR_CONFIG_EVAL['input_shape'])
    DET_TARGET_IMG_HEIGHT_EVAL, DET_TARGET_IMG_WIDTH_EVAL = DET_INPUT_SHAPE_EVAL[0], DET_INPUT_SHAPE_EVAL[1]

    # Используем классы из detector_config, так как они определяют выход модели
    DET_CLASSES_LIST_EVAL = DETECTOR_CONFIG_EVAL['classes']
    if not DET_CLASSES_LIST_EVAL or not isinstance(DET_CLASSES_LIST_EVAL, list):
        print("ОШИБКА: 'classes' в detector_config.yaml не определен или не является списком. Выход.")
        exit(1)
    DET_NUM_CLASSES_EVAL = len(DET_CLASSES_LIST_EVAL)
    if DET_NUM_CLASSES_EVAL == 0:
        print("ОШИБКА: Список классов в detector_config.yaml пуст. Выход.")
        exit(1)

    DET_ANCHORS_WH_NORM_EVAL = np.array(DETECTOR_CONFIG_EVAL['anchors_wh_normalized'], dtype=np.float32)
    DET_NUM_ANCHORS_EVAL = DETECTOR_CONFIG_EVAL['num_anchors_per_location']
    if DET_NUM_ANCHORS_EVAL != DET_ANCHORS_WH_NORM_EVAL.shape[0]:
        print(
            "ОШИБКА: 'num_anchors_per_location' не соответствует количеству якорей в 'anchors_wh_normalized' в detector_config.yaml. Выход.")
        exit(1)

    DET_NETWORK_STRIDE_EVAL = 16  # Предположение, должно быть консистентно с моделью
    DET_GRID_HEIGHT_EVAL = DET_TARGET_IMG_HEIGHT_EVAL // DET_NETWORK_STRIDE_EVAL
    DET_GRID_WIDTH_EVAL = DET_TARGET_IMG_WIDTH_EVAL // DET_NETWORK_STRIDE_EVAL

    _images_subdir_name_eval = BASE_CONFIG_EVAL['dataset']['images_dir']
    _annotations_subdir_name_eval = BASE_CONFIG_EVAL['dataset']['annotations_dir']

    _detector_dataset_ready_path_rel_eval = "data/Detector_Dataset_Ready"
    DETECTOR_DATASET_READY_ABS_EVAL = (_project_root_eval / _detector_dataset_ready_path_rel_eval).resolve()
    VAL_IMAGE_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _images_subdir_name_eval)
    VAL_ANNOT_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _annotations_subdir_name_eval)

    DETECTOR_MODEL_PATH_EVAL = PREDICT_CONFIG_EVAL["detector_model_path"]  # Должен быть в конфиге
    CONF_THRESH_EVAL = float(PREDICT_CONFIG_EVAL.get("default_conf_thresh", 0.25))
    IOU_THRESH_NMS_EVAL = float(PREDICT_CONFIG_EVAL.get("default_iou_thresh", 0.45))
    MAX_DETS_EVAL = int(PREDICT_CONFIG_EVAL.get("default_max_dets", 100))
    IOU_THRESH_FOR_MATCHING_EVAL = 0.5  # Стандартный порог для TP/FP

except KeyError as e_key:
    print(f"ОШИБКА: Отсутствует необходимый ключ в конфигурационном файле: {e_key}. Выход.")
    exit(1)
except Exception as e_conf:
    print(f"ОШИБКА при извлечении параметров из конфигов: {e_conf}. Выход.")
    exit(1)


# --- Вспомогательная Функция для IoU ---
def calculate_iou_for_eval(box1_xyxy, box2_xyxy):
    """box_format: [xmin, ymin, xmax, ymax]"""
    x1_inter = max(box1_xyxy[0], box2_xyxy[0])
    y1_inter = max(box1_xyxy[1], box2_xyxy[1])
    x2_inter = min(box1_xyxy[2], box2_xyxy[2])
    y2_inter = min(box1_xyxy[3], box2_xyxy[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / (union_area + 1e-9)  # Добавил epsilon для стабильности
    return iou


# --- Основная Функция Оценки ---
def evaluate_detector_on_validation_set(cli_args=None):
    """
    Оценивает модель детектора на валидационной выборке, используя параметры из конфигов
    и опционально переопределяемые из аргументов командной строки.
    """
    if not PREDICT_FUNCS_LOADED:
        print("ОШИБКА (evaluate): Не удалось загрузить необходимые функции для инференса (из predict_detector.py).")
        print("                  Пожалуйста, проверьте импорты и наличие файла predict_detector.py в корне проекта.")
        return

    # Получаем параметры из CLI или конфигов
    model_path_to_load = cli_args.model_path if cli_args and cli_args.model_path else DETECTOR_MODEL_PATH_EVAL
    conf_thresh_to_use = cli_args.conf_thresh if cli_args and cli_args.conf_thresh is not None else CONF_THRESH_EVAL
    iou_thresh_nms_to_use = cli_args.iou_thresh if cli_args and cli_args.iou_thresh is not None else IOU_THRESH_NMS_EVAL
    iou_thresh_match_to_use = cli_args.iou_thresh_match if cli_args and cli_args.iou_thresh_match is not None else IOU_THRESH_FOR_MATCHING_EVAL

    # 1. Загрузка модели
    model_full_path_eval = (_project_root_eval / model_path_to_load).resolve()
    print(f"\n--- Оценка Модели Детектора ---")
    print(f"Загрузка модели из: {model_full_path_eval}")

    if not model_full_path_eval.exists():
        print(f"ОШИБКА: Файл модели не найден: {model_full_path_eval}")
        print(f"  Проверьте путь 'detector_model_path' в 'src/configs/predict_config.yaml' или аргумент --model_path.")
        return
    try:
        model = tf.keras.models.load_model(str(model_full_path_eval), custom_objects=CUSTOM_OBJECTS_EVAL, compile=False)
        print("Модель детектора успешно загружена.")

        # Проверка формы выхода загруженной модели
        actual_model_output_shape_raw = model.output.shape
        print(f"  Форма выхода загруженной модели (model.output.shape): {actual_model_output_shape_raw}")

        # Ожидаемая форма (None заменяется на фактическое значение None из модели при сравнении)
        # Важно, чтобы остальные измерения совпадали с конфигом
        expected_num_anchors = DETECTOR_CONFIG_EVAL.get('num_anchors_per_location')
        expected_num_classes = len(DETECTOR_CONFIG_EVAL.get('classes', []))
        expected_features_per_anchor = 5 + expected_num_classes  # 4 (box) + 1 (obj) + C (classes)

        # Сравниваем измерения, которые должны быть фиксированы
        model_grid_h = actual_model_output_shape_raw[1]
        model_grid_w = actual_model_output_shape_raw[2]
        model_num_anchors = actual_model_output_shape_raw[3]
        model_features_per_anchor = actual_model_output_shape_raw[4]

        config_grid_h = DET_TARGET_IMG_HEIGHT_EVAL // DET_NETWORK_STRIDE_EVAL
        config_grid_w = DET_TARGET_IMG_WIDTH_EVAL // DET_NETWORK_STRIDE_EVAL

        valid_shape = True
        if model_grid_h != config_grid_h:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Высота сетки модели ({model_grid_h}) не совпадает с конфигом ({config_grid_h})!")
            valid_shape = False
        if model_grid_w != config_grid_w:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Ширина сетки модели ({model_grid_w}) не совпадает с конфигом ({config_grid_w})!")
            valid_shape = False
        if model_num_anchors != expected_num_anchors:
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Количество якорей в модели ({model_num_anchors}) не совпадает с конфигом ({expected_num_anchors})!")
            valid_shape = False
        if model_features_per_anchor != expected_features_per_anchor:
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Количество признаков на якорь в модели ({model_features_per_anchor}) не совпадает с ожидаемым ({expected_features_per_anchor})!")
            valid_shape = False

        if not valid_shape:
            print(
                "  КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Форма выхода загруженной модели не соответствует конфигурации детектора.")
            print("                    Это, скорее всего, приведет к ошибкам в decode_predictions.")
            print(
                "                    Убедитесь, что загружена правильная модель, обученная с текущими параметрами detector_config.yaml (особенно num_anchors_per_location).")
            # Можно добавить exit(), если это критично, или позволить продолжиться для отладки
            # exit(1)

    except Exception as e:
        print(f"Критическая ошибка загрузки модели детектора: {e}");
        import traceback
        traceback.print_exc()
        return

    # 2. Сбор путей к валидационным данным
    # (Код сбора val_image_paths остается таким же)
    if not Path(VAL_IMAGE_DIR_EVAL).is_dir() or not Path(VAL_ANNOT_DIR_EVAL).is_dir():
        print(f"ОШИБКА: Директории валидационных данных не найдены.");
        return
    val_image_paths = []
    for ext_pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        val_image_paths.extend(list(Path(VAL_IMAGE_DIR_EVAL).glob(ext_pattern)))
    val_image_paths = sorted(list(set(val_image_paths)))
    if not val_image_paths:
        print(f"Изображения не найдены в валидационной директории: {VAL_IMAGE_DIR_EVAL}");
        return

    print(f"Найдено {len(val_image_paths)} изображений в валидационной выборке для оценки.")
    print(f"Используемый порог уверенности (conf_thresh) для детекций: {conf_thresh_to_use:.2f}")
    print(f"Используемый порог IoU для NMS: {iou_thresh_nms_to_use:.2f}")
    print(f"Используемый порог IoU для сопоставления TP/FP: {iou_thresh_match_to_use:.2f}")

    # Инициализация счетчиков (используем словари для большей ясности, если классы могут меняться)
    true_positives = {cls_id: 0 for cls_id in range(DET_NUM_CLASSES_EVAL)}
    false_positives = {cls_id: 0 for cls_id in range(DET_NUM_CLASSES_EVAL)}
    num_gt_objects_per_class = {cls_id: 0 for cls_id in range(DET_NUM_CLASSES_EVAL)}

    # 3. Итерация по валидационным изображениям
    for img_idx, image_path_obj in enumerate(val_image_paths):
        image_path_str = str(image_path_obj)
        if (img_idx + 1) % 50 == 0 or img_idx == 0 or img_idx == len(val_image_paths) - 1:  # Логируем чаще
            print(f"  Обработка изображения {img_idx + 1}/{len(val_image_paths)}: {image_path_obj.name}")

        original_bgr_image = cv2.imread(image_path_str)
        if original_bgr_image is None:
            print(f"    Предупреждение: Не удалось прочитать {image_path_str}, пропускаем.");
            continue
        original_h, original_w = original_bgr_image.shape[:2]

        xml_filename = image_path_obj.stem + ".xml"
        xml_path_str = str(Path(VAL_ANNOT_DIR_EVAL) / xml_filename)

        # Используем DET_CLASSES_LIST_EVAL для parse_xml_annotation, так как он соответствует модели
        gt_objects_data, _, _, _ = parse_xml_annotation(xml_path_str, DET_CLASSES_LIST_EVAL)
        if gt_objects_data is None:
            print(f"    Предупреждение: Ошибка парсинга XML для {xml_filename} или файл не найден, пропускаем.");
            continue

        gt_boxes_for_eval_img = []
        for gt_obj in gt_objects_data:
            class_id_int = int(gt_obj['class_id'])
            if 0 <= class_id_int < DET_NUM_CLASSES_EVAL:  # Проверка, что class_id валиден
                num_gt_objects_per_class[class_id_int] += 1
                gt_boxes_for_eval_img.append([
                    int(gt_obj['xmin']), int(gt_obj['ymin']), int(gt_obj['xmax']), int(gt_obj['ymax']),
                    class_id_int,
                    False  # matched_flag
                ])
            else:
                print(f"    Предупреждение: Невалидный class_id={class_id_int} в {xml_filename}. Пропускаем объект.")

        detector_input_batch = preprocess_image_for_model(original_bgr_image, DET_TARGET_IMG_HEIGHT_EVAL,
                                                          DET_TARGET_IMG_WIDTH_EVAL)
        raw_preds = model.predict(detector_input_batch,
                                  verbose=0)  # verbose=0 чтобы не было прогресс-бара на каждом predict

        # Передаем правильные параметры, соответствующие модели и конфигу
        decoded_boxes_xywh_n, obj_conf_n, class_probs_n = decode_predictions(
            raw_preds,
            DET_ANCHORS_WH_NORM_EVAL,  # Якоря из конфига
            DET_GRID_HEIGHT_EVAL,  # Размеры сетки из конфига
            DET_GRID_WIDTH_EVAL,
            DET_NUM_CLASSES_EVAL,  # Количество классов из конфига
            DET_NETWORK_STRIDE_EVAL
        )

        nms_boxes_n, nms_scores_n, nms_classes_n, num_valid_dets_n = apply_nms_and_filter(
            decoded_boxes_xywh_n, obj_conf_n, class_probs_n,
            DET_GRID_HEIGHT_EVAL, DET_GRID_WIDTH_EVAL,
            DET_NUM_ANCHORS_EVAL,  # Количество якорей из конфига
            DET_NUM_CLASSES_EVAL,
            confidence_threshold=conf_thresh_to_use,
            iou_threshold=iou_thresh_nms_to_use,
            max_detections=MAX_DETS_EVAL
        )
        num_detections = int(num_valid_dets_n[0].numpy())

        pred_boxes_norm_yxYX = nms_boxes_n[0][:num_detections].numpy()  # ymin, xmin, ymax, xmax
        pred_scores = nms_scores_n[0][:num_detections].numpy()
        pred_class_ids = nms_classes_n[0][:num_detections].numpy().astype(int)

        if num_detections > 0:
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes_norm_yxYX_sorted = pred_boxes_norm_yxYX[sorted_indices]
            pred_scores_sorted = pred_scores[sorted_indices]  # Используем отсортированные скоры тоже
            pred_class_ids_sorted = pred_class_ids[sorted_indices]

            for i_pred in range(num_detections):
                pred_box_norm = pred_boxes_norm_yxYX_sorted[i_pred]
                pred_score_val = pred_scores_sorted[i_pred]  # Используем score предсказания
                pred_cls_id_int = pred_class_ids_sorted[i_pred]

                # Пропускаем предсказания с очень низкой уверенностью, даже если NMS их пропустил (на всякий случай)
                # NMS уже должен был это сделать по score_threshold, но дополнительная проверка не помешает.
                if pred_score_val < conf_thresh_to_use:  # Используем тот же порог, что и для NMS
                    continue

                pred_box_px = [
                    int(pred_box_norm[1] * original_w), int(pred_box_norm[0] * original_h),
                    int(pred_box_norm[3] * original_w), int(pred_box_norm[2] * original_h)]

                best_iou_for_pred = 0.0
                best_gt_match_idx = -1

                for i_gt, gt_data_item in enumerate(gt_boxes_for_eval_img):
                    gt_box_px_item = gt_data_item[0:4]
                    gt_cls_id_item = gt_data_item[4]
                    gt_matched_flag_item = gt_data_item[5]

                    if gt_cls_id_item == pred_cls_id_int and not gt_matched_flag_item:
                        iou = calculate_iou_for_eval(pred_box_px, gt_box_px_item)
                        if iou > best_iou_for_pred:
                            best_iou_for_pred = iou
                            best_gt_match_idx = i_gt

                if 0 <= pred_cls_id_int < DET_NUM_CLASSES_EVAL:  # Проверка валидности ID класса
                    if best_iou_for_pred >= iou_thresh_match_to_use:
                        true_positives[pred_cls_id_int] += 1
                        gt_boxes_for_eval_img[best_gt_match_idx][5] = True
                    else:
                        false_positives[pred_cls_id_int] += 1
                else:
                    print(
                        f"    ПРЕДУПРЕЖДЕНИЕ: Невалидный pred_cls_id_int={pred_cls_id_int} при сопоставлении в {image_path_obj.name}. FP засчитан для абстрактного класса, если такой есть, или пропущен.")
                    # Можно либо игнорировать, либо засчитывать FP для какого-то общего "неизвестного" класса, если это нужно.
                    # Пока просто выведем предупреждение.

    # 4. Расчет финальных метрик
    # (Код расчета метрик остается таким же, как в твоей последней версии)
    print("\n--- Результаты Оценки ---")
    precision_per_class_list = []
    recall_per_class_list = []
    f1_per_class_list = []

    for i_cls in range(DET_NUM_CLASSES_EVAL):
        class_name = DET_CLASSES_LIST_EVAL[i_cls]
        tp = true_positives.get(i_cls, 0)  # Используем .get для безопасности
        fp = false_positives.get(i_cls, 0)
        num_gt = num_gt_objects_per_class.get(i_cls, 0)
        fn = num_gt - tp

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (num_gt + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        precision_per_class_list.append(precision)
        recall_per_class_list.append(recall)
        f1_per_class_list.append(f1)

        print(f"\nКласс: {class_name} (ID: {i_cls})")
        print(f"  Всего Ground Truth объектов: {int(num_gt)}")
        print(f"  True Positives (TP): {int(tp)}")
        print(f"  False Positives (FP): {int(fp)}")
        print(f"  False Negatives (FN): {int(fn)}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")

    # Макро-усреднение (только по классам, у которых были GT объекты)
    valid_precisions = [p for i, p in enumerate(precision_per_class_list) if num_gt_objects_per_class.get(i, 0) > 0]
    valid_recalls = [r for i, r in enumerate(recall_per_class_list) if num_gt_objects_per_class.get(i, 0) > 0]
    valid_f1s = [f for i, f in enumerate(f1_per_class_list) if num_gt_objects_per_class.get(i, 0) > 0]

    macro_precision = np.mean(valid_precisions) if valid_precisions else 0.0
    macro_recall = np.mean(valid_recalls) if valid_recalls else 0.0
    macro_f1 = np.mean(valid_f1s) if valid_f1s else 0.0

    print("\n--- Макро-усредненные метрики (по классам с GT объектами) ---")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall: {macro_recall:.4f}")
    print(f"  Macro F1-score: {macro_f1:.4f}")

    # Микро-усреднение
    total_tp_all = sum(true_positives.values())
    total_fp_all = sum(false_positives.values())
    total_gt_all = sum(num_gt_objects_per_class.values())

    micro_precision = total_tp_all / (total_tp_all + total_fp_all + 1e-9)
    micro_recall = total_tp_all / (total_gt_all + 1e-9)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)
    print("\n--- Микро-усредненные метрики (по всем объектам) ---")
    print(f"  Micro Precision: {micro_precision:.4f}")
    print(f"  Micro Recall: {micro_recall:.4f}")
    print(f"  Micro F1-score: {micro_f1:.4f}")

    print("\nОценка завершена.")


if __name__ == "__main__":
    if not PREDICT_FUNCS_LOADED:
        print("Выход из evaluate_detector.py из-за ошибки импорта функций предсказания.")
        exit(1)

    parser_eval = argparse.ArgumentParser(description="Оценка модели детектора на валидационной выборке.")
    # Аргументы теперь не обязательны, будут браться из PREDICT_CONFIG_EVAL
    parser_eval.add_argument("--model_path", type=str, default=None, help="Путь к .keras файлу модели детектора.")
    parser_eval.add_argument("--conf_thresh", type=float, default=None, help="Порог уверенности для NMS.")
    parser_eval.add_argument("--iou_thresh", type=float, default=None, help="Порог IoU для NMS.")
    parser_eval.add_argument("--iou_thresh_match", type=float, default=None, help="Порог IoU для сопоставления TP/FP.")

    args_eval = parser_eval.parse_args()

    evaluate_detector_on_validation_set(args_eval)