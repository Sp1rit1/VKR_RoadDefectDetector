# RoadDefectDetector/evaluate_detector.py
import tensorflow as tf
import numpy as np
import cv2
import yaml
import os
import glob
from pathlib import Path
import argparse
import json  # Для возможного сохранения результатов в JSON

# --- Добавляем src в sys.path, чтобы скрипт можно было запускать из корня проекта ---
_project_root_eval = Path(__file__).resolve().parent
_src_path_eval = _project_root_eval / 'src'
import sys

if str(_src_path_eval) not in sys.path:
    sys.path.insert(0, str(_src_path_eval))

# --- Импорты из твоих модулей ---
CUSTOM_OBJECTS_EVAL = {}
try:
    from losses.detection_losses import compute_detector_loss_v2_fpn

    CUSTOM_OBJECTS_EVAL['compute_detector_loss_v2_fpn'] = compute_detector_loss_v2_fpn
    print("INFO (evaluate_detector.py): Кастомная функция потерь compute_detector_loss_v2_fpn ЗАГРУЖЕНА.")
except ImportError as e_loss_eval:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Не удалось импортировать compute_detector_loss_v2_fpn: {e_loss_eval}.")
except Exception as e_gen_loss_eval:
    print(f"ПРЕДУПРЕЖДЕНИЕ (evaluate_detector.py): Общая ошибка при импорте функции потерь: {e_gen_loss_eval}.")

try:
    from datasets.detector_data_loader import parse_xml_annotation  # Нужна для чтения GT аннотаций

    print("INFO (evaluate_detector.py): Функция parse_xml_annotation успешно импортирована.")
except ImportError as e_parse_imp:
    print(f"ОШИБКА (evaluate_detector.py): Не удалось импортировать parse_xml_annotation: {e_parse_imp}. Выход.")
    exit()

# --- Загрузка функций из predict_detector.py (или predict_pipeline.py) ---
PREDICT_FUNCS_LOADED = False
# Предполагаем, что predict_pipeline.py находится в КОРНЕ проекта
_predict_script_name = "run_prediction_pipeline.py"  # ИЛИ "predict_detector.py" если функции там
_predict_script_path = _project_root_eval / _predict_script_name
# Чтобы импортировать из него, нужно временно добавить корень проекта в sys.path, если еще не там
if str(_project_root_eval) not in sys.path:
    sys.path.insert(0, str(_project_root_eval))

try:
    if _predict_script_name == "run_prediction_pipeline.py":
        from run_prediction_pipeline import (  # Импортируем из правильного файла
            preprocess_image_for_model_tf,
            decode_single_level_predictions_generic,
            apply_nms_and_filter_generic
            # draw_detections_on_image # Эта функция здесь не нужна, так как мы только считаем метрики
        )
    # elif _predict_script_name == "predict_detector.py": # Если бы они были в другом файле
    #     from predict_detector import (...)
    PREDICT_FUNCS_LOADED = True
    print(f"INFO (evaluate_detector.py): Функции инференса успешно импортированы из {_predict_script_name}.")
except ImportError as e_imp_pred:
    print(f"ОШИБКА: Не удалось импортировать функции из {_predict_script_name}: {e_imp_pred}")


    # Заглушки, чтобы скрипт не падал сразу, но это будет нерабочий вариант
    def preprocess_image_for_model_tf(i, h, w):
        return None


    def decode_single_level_predictions_generic(r, a, gh, gw, nc):
        return None, None, None


    def apply_nms_and_filter_generic(b, o, c, nc, ct, it, md):
        return None, None, None, tf.constant([0])
except Exception as e_gen_pred_imp:
    print(f"ОШИБКА: Общая ошибка при импорте функций из {_predict_script_name}: {e_gen_pred_imp}")


# --- Загрузка Конфигураций ---
def load_config_eval(config_path_obj, config_name_for_log, exit_on_error=True, default_on_error=None):
    if default_on_error is None: default_on_error = {}
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(f"ОШИБКА: {config_path_obj.name} пуст или неверный формат для '{config_name_for_log}'.")
            if exit_on_error: exit()
            return default_on_error
        print(f"INFO: Конфиг '{config_name_for_log}' ({config_path_obj.name}) успешно загружен.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден: {config_path_obj}.");
        if exit_on_error: exit()
        return default_on_error
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML в {config_path_obj.name}: {e}.");
        if exit_on_error: exit()
        return default_on_error


print("\n--- Загрузка конфигурационных файлов для evaluate_detector.py ---")
_base_config_path_obj_eval = _src_path_eval / 'configs' / 'base_config.yaml'
_detector_arch_config_path_obj_eval = _src_path_eval / 'configs' / 'detector_config.yaml'
_predict_config_path_obj_eval = _src_path_eval / 'configs' / 'predict_config.yaml'

BASE_CONFIG_EVAL = load_config_eval(_base_config_path_obj_eval, "Base Config")
DETECTOR_ARCH_CONFIG_EVAL = load_config_eval(_detector_arch_config_path_obj_eval, "Detector Arch Config")
PREDICT_CONFIG_EVAL = load_config_eval(_predict_config_path_obj_eval, "Predict Config")

# --- Параметры из Конфигов ---
_fpn_params_eval = DETECTOR_ARCH_CONFIG_EVAL.get('fpn_detector_params', {})
DET_INPUT_SHAPE_EVAL = tuple(_fpn_params_eval.get('input_shape', [416, 416, 3]))
DET_TARGET_IMG_HEIGHT_EVAL, DET_TARGET_IMG_WIDTH_EVAL = DET_INPUT_SHAPE_EVAL[0], DET_INPUT_SHAPE_EVAL[1]
DET_CLASSES_LIST_EVAL = _fpn_params_eval.get('classes', ['pit', 'crack'])
DET_NUM_CLASSES_EVAL = len(DET_CLASSES_LIST_EVAL)

DET_FPN_LEVELS_EVAL = _fpn_params_eval.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
DET_FPN_STRIDES_EVAL = _fpn_params_eval.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})
DET_FPN_ANCHOR_CONFIGS_EVAL = _fpn_params_eval.get('detector_fpn_anchor_configs', {})

FPN_LEVEL_DETAILS_FOR_DECODE_EVAL = {}
for _lvl_name_eval in DET_FPN_LEVELS_EVAL:
    _lvl_cfg_eval = DET_FPN_ANCHOR_CONFIGS_EVAL.get(_lvl_name_eval, {})
    _lvl_stride_eval = DET_FPN_STRIDES_EVAL.get(_lvl_name_eval)
    if not _lvl_cfg_eval or _lvl_stride_eval is None:
        print(f"ОШИБКА: Неполная конфигурация для FPN уровня '{_lvl_name_eval}' в detector_config.yaml. Выход.");
        exit()
    _num_anchors_lvl = _lvl_cfg_eval.get('num_anchors_this_level', 3)
    _anchors_wh_lvl = _lvl_cfg_eval.get('anchors_wh_normalized', [[0.1, 0.1]] * _num_anchors_lvl)
    FPN_LEVEL_DETAILS_FOR_DECODE_EVAL[_lvl_name_eval] = {
        'anchors_wh_normalized': np.array(_anchors_wh_lvl, dtype=np.float32),
        'num_anchors': _num_anchors_lvl,
        'grid_h': DET_TARGET_IMG_HEIGHT_EVAL // _lvl_stride_eval,
        'grid_w': DET_TARGET_IMG_WIDTH_EVAL // _lvl_stride_eval,
        'stride': _lvl_stride_eval
    }

_images_subdir_name_eval = BASE_CONFIG_EVAL.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_eval = BASE_CONFIG_EVAL.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel_eval = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_EVAL = (_project_root_eval / _detector_dataset_ready_path_rel_eval).resolve()
VAL_IMAGE_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _images_subdir_name_eval)
VAL_ANNOT_DIR_EVAL = str(DETECTOR_DATASET_READY_ABS_EVAL / "validation" / _annotations_subdir_name_eval)


# --- Вспомогательная Функция для Расчета IoU ---
def calculate_iou_for_eval(box1_xyxy, box2_xyxy):  # [xmin, ymin, xmax, ymax]
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


# --- КОНЕЦ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

def evaluate_detector(model_path_arg, conf_thresh_arg, iou_thresh_nms_arg, iou_thresh_matching_arg, max_dets_arg):
    """Основная функция оценки детектора."""
    if not PREDICT_FUNCS_LOADED:  # Проверяем флаг здесь
        print("ОШИБКА: Не удалось загрузить необходимые функции для инференса. Оценка невозможна.")
        return

    model_full_path_abs_eval = (_project_root_eval / model_path_arg).resolve()
    print(f"\n--- Оценка Модели Детектора ---")
    print(f"Загрузка модели из: {model_full_path_abs_eval}")
    if not model_full_path_abs_eval.exists():
        print(f"ОШИБКА: Файл модели не найден: {model_full_path_abs_eval}");
        return
    try:
        model = tf.keras.models.load_model(str(model_full_path_abs_eval), custom_objects=CUSTOM_OBJECTS_EVAL,
                                           compile=False)
        print("Модель детектора успешно загружена.")
    except Exception as e_load:
        print(f"Ошибка загрузки модели детектора: {e_load}"); return

    if not Path(VAL_IMAGE_DIR_EVAL).is_dir() or not Path(VAL_ANNOT_DIR_EVAL).is_dir():
        print(f"ОШИБКА: Директории валидационных данных не найдены.");
        return

    val_image_paths_list = sorted(list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpg")) + \
                                  list(Path(VAL_IMAGE_DIR_EVAL).glob("*.jpeg")) + \
                                  list(Path(VAL_IMAGE_DIR_EVAL).glob("*.png")))
    if not val_image_paths_list: print(f"Изображения не найдены в: {VAL_IMAGE_DIR_EVAL}"); return

    print(f"\nНайдено {len(val_image_paths_list)} изображений в валидационной выборке для оценки.")
    print(f"Используемые параметры для оценки:")
    print(f"  Порог уверенности (для NMS): {conf_thresh_arg:.3f}")
    print(f"  Порог IoU для NMS: {iou_thresh_nms_arg:.3f}")
    print(f"  Порог IoU для сопоставления TP/FP: {iou_thresh_matching_arg:.3f}")
    print(f"  Максимальное количество детекций: {max_dets_arg}")

    true_positives_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)
    false_positives_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)
    num_gt_objects_per_class = np.zeros(DET_NUM_CLASSES_EVAL, dtype=np.int32)

    for img_idx, image_path_obj_eval in enumerate(val_image_paths_list):
        image_path_str_eval = str(image_path_obj_eval)
        if (img_idx + 1) % 50 == 0 or img_idx == 0:  # Логирование прогресса
            print(f"  Обработка изображения {img_idx + 1}/{len(val_image_paths_list)}: {image_path_obj_eval.name}")

        original_bgr_image_eval = cv2.imread(image_path_str_eval)
        if original_bgr_image_eval is None: continue
        original_h_eval, original_w_eval = original_bgr_image_eval.shape[:2]

        xml_filename_eval = image_path_obj_eval.stem + ".xml"
        xml_path_str_eval = str(Path(VAL_ANNOT_DIR_EVAL) / xml_filename_eval)

        gt_objects_from_xml, _, _, _ = parse_xml_annotation(xml_path_str_eval, DET_CLASSES_LIST_EVAL)
        if gt_objects_from_xml is None: continue

        current_gt_boxes_for_matching = []
        for gt_obj_item in gt_objects_from_xml:
            num_gt_objects_per_class[int(gt_obj_item['class_id'])] += 1
            current_gt_boxes_for_matching.append([
                int(gt_obj_item['xmin']), int(gt_obj_item['ymin']),
                int(gt_obj_item['xmax']), int(gt_obj_item['ymax']),
                int(gt_obj_item['class_id']), False])

        detector_input_batch_eval = preprocess_image_for_model_tf(original_bgr_image_eval, DET_TARGET_IMG_HEIGHT_EVAL,
                                                                  DET_TARGET_IMG_WIDTH_EVAL)
        raw_model_outputs = model.predict(detector_input_batch_eval, verbose=0)

        all_level_decoded_boxes_list_eval, all_level_obj_conf_list_eval, all_level_class_probs_list_eval = [], [], []

        if isinstance(raw_model_outputs, list):  # FPN case
            for i_lvl_eval, level_key_eval in enumerate(DET_FPN_LEVELS_EVAL):
                raw_preds_this_level_eval = raw_model_outputs[i_lvl_eval]
                level_config_eval = FPN_LEVEL_DETAILS_FOR_DECODE_EVAL.get(level_key_eval)

                decoded_boxes, obj_conf, class_probs = decode_single_level_predictions_generic(
                    raw_preds_this_level_eval,
                    level_config_eval['anchors_wh_normalized'],
                    level_config_eval['grid_h'], level_config_eval['grid_w'],
                    DET_NUM_CLASSES_EVAL)
                all_level_decoded_boxes_list_eval.append(decoded_boxes)
                all_level_obj_conf_list_eval.append(obj_conf)
                all_level_class_probs_list_eval.append(class_probs)
        else:  # Single-level
            # ... (Если будешь реализовывать, то здесь логика для одного выхода)
            print("ОШИБКА: evaluate_detector.py (в текущей версии) ожидает FPN выход (список тензоров).")
            continue

        if not all_level_decoded_boxes_list_eval:
            # Этого не должно быть, если FPN отработал, списки будут, но могут быть пустыми тензорами
            continue

        final_boxes_norm_eval, final_scores_eval, final_classes_ids_eval, num_valid_dets_eval = apply_nms_and_filter_generic(
            all_level_decoded_boxes_list_eval, all_level_obj_conf_list_eval, all_level_class_probs_list_eval,
            DET_NUM_CLASSES_EVAL, conf_thresh_arg, iou_thresh_nms_arg, max_dets_arg  # Передаем max_dets_arg
        )

        num_predictions_after_nms = int(num_valid_dets_eval[0].numpy())

        if num_predictions_after_nms > 0:
            pred_boxes_norm_nms = final_boxes_norm_eval[0][:num_predictions_after_nms].numpy()
            pred_scores_nms = final_scores_eval[0][:num_predictions_after_nms].numpy()
            pred_class_ids_nms = final_classes_ids_eval[0][:num_predictions_after_nms].numpy().astype(int)

            sorted_pred_indices = np.argsort(pred_scores_nms)[::-1]

            for pred_idx_sorted in sorted_pred_indices:
                pred_box_norm_current = pred_boxes_norm_nms[pred_idx_sorted]
                pred_class_id_current = pred_class_ids_nms[pred_idx_sorted]

                pred_box_pixels_current = [
                    int(pred_box_norm_current[1] * original_w_eval), int(pred_box_norm_current[0] * original_h_eval),
                    int(pred_box_norm_current[3] * original_w_eval), int(pred_box_norm_current[2] * original_h_eval)]

                best_iou_for_this_pred = 0.0
                best_gt_match_index_for_this_pred = -1

                for i_gt_loop_idx, gt_data_eval in enumerate(current_gt_boxes_for_matching):  # ИЗМЕНЕНО: i_gt_loop_idx
                    gt_box_pixels_current = gt_data_eval[0:4]
                    gt_class_id_current = gt_data_eval[4]
                    is_gt_already_matched = gt_data_eval[5]

                    if gt_class_id_current == pred_class_id_current and not is_gt_already_matched:
                        iou = calculate_iou_for_eval(pred_box_pixels_current, gt_box_pixels_current)
                        if iou > best_iou_for_this_pred:
                            best_iou_for_this_pred = iou
                            best_gt_match_index_for_this_pred = i_gt_loop_idx  # ИЗМЕНЕНО: i_gt_loop_idx

                if best_iou_for_this_pred >= iou_thresh_matching_arg:
                    true_positives_per_class[pred_class_id_current] += 1
                    if best_gt_match_index_for_this_pred != -1:  # Проверка, что совпадение было найдено
                        current_gt_boxes_for_matching[best_gt_match_index_for_this_pred][5] = True
                else:
                    false_positives_per_class[pred_class_id_current] += 1

    # Расчет финальных метрик (как было)
    # ... (вставь сюда свой блок расчета и вывода Precision, Recall, F1)
    print("\n--- Результаты Оценки ---")
    precision_per_class = np.zeros(DET_NUM_CLASSES_EVAL);
    recall_per_class = np.zeros(DET_NUM_CLASSES_EVAL);
    f1_per_class = np.zeros(DET_NUM_CLASSES_EVAL)
    for i_cls_metric in range(DET_NUM_CLASSES_EVAL):
        class_name_metric = DET_CLASSES_LIST_EVAL[i_cls_metric];
        tp = true_positives_per_class[i_cls_metric];
        fp = false_positives_per_class[i_cls_metric]
        fn = num_gt_objects_per_class[i_cls_metric] - tp
        precision = tp / (tp + fp + 1e-6);
        recall = tp / (num_gt_objects_per_class[i_cls_metric] + 1e-6);
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        precision_per_class[i_cls_metric] = precision;
        recall_per_class[i_cls_metric] = recall;
        f1_per_class[i_cls_metric] = f1
        print(f"\nКласс: {class_name_metric} (ID: {i_cls_metric})")
        print(f"  Всего Ground Truth объектов: {int(num_gt_objects_per_class[i_cls_metric])}")
        print(f"  True Positives (TP): {int(tp)}");
        print(f"  False Positives (FP): {int(fp)}");
        print(f"  False Negatives (FN): {int(fn)}")
        print(f"  Precision: {precision:.4f}");
        print(f"  Recall: {recall:.4f}");
        print(f"  F1-score: {f1:.4f}")
    macro_precision = np.mean(precision_per_class);
    macro_recall = np.mean(recall_per_class);
    macro_f1 = np.mean(f1_per_class)
    print("\n--- Макро-усредненные метрики (усреднение по классам) ---")
    print(f"  Macro Precision: {macro_precision:.4f}");
    print(f"  Macro Recall: {macro_recall:.4f}");
    print(f"  Macro F1-score: {macro_f1:.4f}")
    total_tp_all_classes = np.sum(true_positives_per_class);
    total_fp_all_classes = np.sum(false_positives_per_class);
    total_gt_all_classes = np.sum(num_gt_objects_per_class)
    micro_precision = total_tp_all_classes / (total_tp_all_classes + total_fp_all_classes + 1e-6)
    micro_recall = total_tp_all_classes / (total_gt_all_classes + 1e-6)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-6)
    print("\n--- Микро-усредненные метрики (общие по всем объектам) ---")
    print(f"  Micro Precision: {micro_precision:.4f}");
    print(f"  Micro Recall: {micro_recall:.4f}");
    print(f"  Micro F1-score: {micro_f1:.4f}")
    print("\nОценка завершена.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели детектора на валидационной выборке.")
    parser.add_argument("--model_path", type=str,
                        default=PREDICT_CONFIG_EVAL.get("detector_model_path", "weights/detector_default_best.keras"),
                        help="Путь к обученной модели детектора (.keras).")
    parser.add_argument("--conf_thresh", type=float,
                        default=PREDICT_CONFIG_EVAL.get("default_conf_thresh", 0.25),
                        help="Порог уверенности для отсеивания предсказаний перед NMS и оценкой.")
    parser.add_argument("--iou_thresh_nms", type=float,
                        default=PREDICT_CONFIG_EVAL.get("default_iou_thresh", 0.45),
                        help="Порог IoU для Non-Maximum Suppression.")
    parser.add_argument("--iou_thresh_matching", type=float, default=0.5,
                        help="Порог IoU для сопоставления предсказания с Ground Truth (для TP/FP). Стандартно 0.5.")
    # Добавляем аргумент для max_dets, который был пропущен
    parser.add_argument("--max_dets", type=int,
                        default=PREDICT_CONFIG_EVAL.get("default_max_dets", 100),
                        help="Максимальное количество детекций после NMS.")

    args_eval = parser.parse_args()

    if not PREDICT_FUNCS_LOADED:  # Проверяем флаг здесь
        print("Выход из evaluate_detector.py из-за ошибки импорта функций предсказания.")
    else:
        evaluate_detector(args_eval.model_path,
                          args_eval.conf_thresh,
                          args_eval.iou_thresh_nms,
                          args_eval.iou_thresh_matching,
                          args_eval.max_dets)  # Передаем max_dets