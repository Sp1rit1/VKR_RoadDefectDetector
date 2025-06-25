# RoadDefectDetector/evaluate_detector_v3_standard.py

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
import time  # Для измерения времени оценки

import tensorflow as tf
import numpy as np
from tqdm import tqdm  # Для индикатора прогресса

# --- Настройка путей для импорта ---
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    _added_to_sys_path_eval = True
else:
    _added_to_sys_path_eval = False

# --- Настройка логирования ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --- Импорт наших модулей ---
try:
    from src.datasets.data_loader_v3_standard import DataGenerator, generate_all_anchors
    from src.models.detector_v3_standard import build_detector_v3_standard
    from src.utils.postprocessing import decode_predictions, perform_nms
except ImportError as e:
    logger.error(f"Ошибка импорта модулей проекта: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    if _added_to_sys_path_eval: sys.path.pop(0)
    sys.exit(1)


# --- Функции для расчета метрик ---

def calculate_iou(box1, box2):
    """Вычисляет IoU между двумя боксами. Формат: [xmin, ymin, xmax, ymax]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        intersection_area = 0.0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / (union_area + 1e-7)


def calculate_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return precision, recall, f1


# --- Основная функция оценки ---
def evaluate_detector(main_config_path, predict_config_path, weights_filename="best_model.weights.h5"):
    logger.info("--- Начало оценки модели детектора (Precision, Recall, F1) ---")

    # 1. Загрузка конфигов
    try:
        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        with open(predict_config_path, 'r', encoding='utf-8') as f:
            predict_config = yaml.safe_load(f)
        logger.info("Конфигурационные файлы успешно загружены.")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигов: {e}");
        sys.exit(1)

    logger.info(f"Параметры для построения модели из main_config:")
    logger.info(f"  freeze_backbone: {main_config.get('freeze_backbone', 'Не указано, default True в модели')}")
    logger.info(f"  l2_regularization: {main_config.get('l2_regularization', 'Не указано, default None в модели')}")

    # 2. Подготовка путей
    weights_base_dir = PROJECT_ROOT / main_config.get('weights_base_dir', 'weights')
    saved_model_dir_base = weights_base_dir / main_config['saved_model_dir']
    weights_path = saved_model_dir_base / weights_filename

    if not weights_path.exists():
        logger.error(f"Файл с весами не найден: {weights_path}")
        sys.exit(1)
    logger.info(f"Используются веса из: {weights_path}")

    # 3. Создание модели
    logger.info("Создание модели...")
    model = build_detector_v3_standard(main_config)  # Убедитесь, что это tf.keras.Model
    model.load_weights(str(weights_path))
    logger.info("Модель создана и веса загружены.")

    # 4. Создание валидационного датасета
    batch_size_eval = 1
    logger.info(f"Создание валидационного датасета (batch_size={batch_size_eval}, debug_mode=True)...")
    fpn_strides_eval = main_config.get('fpn_strides', [8, 16, 32])
    all_anchors_for_eval = generate_all_anchors(
        main_config['input_shape'], fpn_strides_eval,
        main_config['anchor_scales'], main_config['anchor_ratios']
    )
    all_anchors_for_eval_tf = tf.constant(all_anchors_for_eval, dtype=tf.float32)

    val_generator_instance = DataGenerator(
        config=main_config, all_anchors=all_anchors_for_eval,
        is_training=False, debug_mode=True
    )
    num_val_images = len(val_generator_instance)
    if num_val_images == 0: logger.error("Валидационный датасет пуст."); sys.exit(1)
    logger.info(f"Количество изображений в валидационном датасете: {num_val_images}")

    # 5. Итерация по датасету и сбор предсказаний/GT
    all_predictions = []  # Список словарей для предсказаний [{'image_id', 'bbox', 'score', 'class_id'}]
    all_ground_truths = []  # Список словарей для GT [{'image_id', 'bbox', 'class_id', 'used'}]
    image_id_counter = 0
    class_id_map = {name: i for i, name in enumerate(main_config['class_names'])}
    h_target, w_target = main_config['input_shape'][:2]

    logger.info("Начало итерации по валидационному датасету...")
    start_time = time.time()

    for image_norm, (y_true_reg_flat, y_true_cls_flat), debug_info in tqdm(val_generator_instance(),
                                                                           total=num_val_images, desc="Оценка"):
        image_batch = tf.expand_dims(image_norm, axis=0)
        raw_predictions_list = model.predict(image_batch, verbose=0)
        decoded_boxes_norm, decoded_scores = decode_predictions(raw_predictions_list, all_anchors_for_eval_tf,
                                                                main_config)
        nms_boxes_yxYX, nms_scores, nms_classes, valid_detections = perform_nms(
            decoded_boxes_norm, decoded_scores, predict_config
        )
        num_dets = valid_detections[0].numpy()

        for i in range(num_dets):
            ymin, xmin, ymax, xmax = nms_boxes_yxYX[0, i].numpy()
            all_predictions.append({
                'image_id': image_id_counter,
                'bbox': [xmin, ymin, xmax, ymax],
                'score': nms_scores[0, i].numpy(),
                'class_id': int(nms_classes[0, i].numpy())
            })

        gt_boxes_pixels_from_debug = debug_info.get('gt_boxes_augmented', np.empty((0, 4)))
        gt_class_names_from_debug = debug_info.get('gt_class_names_augmented', np.empty((0,)))

        for box_pixels, class_name_str in zip(gt_boxes_pixels_from_debug, gt_class_names_from_debug):
            class_id = class_id_map.get(class_name_str)
            if class_id is not None:
                x1_p, y1_p, x2_p, y2_p = box_pixels
                gt_box_xyxy_norm = [
                    x1_p / w_target, y1_p / h_target,
                    x2_p / w_target, y2_p / h_target
                ]
                all_ground_truths.append({
                    'image_id': image_id_counter,
                    'bbox': gt_box_xyxy_norm,
                    'class_id': class_id,
                    'used': False  # 'used' флаг для сопоставления
                })
        image_id_counter += 1

    end_time = time.time();
    logger.info(f"Итерация завершена. Время: {end_time - start_time:.2f} сек.")

    # 6. Расчет метрик (только Precision, Recall, F1)
    logger.info("--- Расчет метрик (Precision, Recall, F1) ---")
    num_classes = main_config['num_classes']
    class_names = main_config['class_names']
    # Используем порог IoU и порог уверенности из predict_config для определения TP/FP
    iou_threshold_for_metrics = predict_config.get('eval_iou_threshold', 0.2)


    print("\n" + "=" * 30 + " Результаты Оценки " + "=" * 30)
    print(f"Порог IoU для TP/FP: {iou_threshold_for_metrics}")
    # print(f"Порог уверенности для предсказаний: {score_threshold_for_metrics}") # Если бы применяли

    # Общие TP, FP, FN для расчета Micro-F1 (если нужно)
    total_tp, total_fp, total_fn = 0, 0, 0

    for class_id_eval in range(num_classes):
        class_name_eval = class_names[class_id_eval]
        # Фильтруем предсказания по классу И по порогу уверенности (если бы он был)
        preds_for_class_metrics = [
            p for p in all_predictions
            if p['class_id'] == class_id_eval  # and p['score'] >= score_threshold_for_metrics
        ]
        gts_for_class_metrics = [g for g in all_ground_truths if g['class_id'] == class_id_eval]

        # Сортируем предсказания по убыванию уверенности (важно для правильного сопоставления)
        preds_for_class_metrics.sort(key=lambda x: x['score'], reverse=True)

        tp_class, fp_class = 0, 0
        # Сбрасываем 'used' флаги для GT этого класса перед сопоставлением
        for gt in gts_for_class_metrics: gt['used'] = False

        for pred in preds_for_class_metrics:
            best_iou = 0.0
            best_gt_match_idx = -1  # Индекс в gts_for_class_metrics

            # Ищем лучший GT на том же изображении
            gts_on_image_for_pred = [
                (idx, gt) for idx, gt in enumerate(gts_for_class_metrics)
                if gt['image_id'] == pred['image_id'] and not gt['used']  # Только неиспользованные GT
            ]

            for gt_original_idx, gt_on_img in gts_on_image_for_pred:
                iou = calculate_iou(pred['bbox'], gt_on_img['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_match_idx = gt_original_idx  # Сохраняем индекс из gts_for_class_metrics

            if best_iou >= iou_threshold_for_metrics and best_gt_match_idx != -1:
                # Условие `and not gts_for_class_metrics[best_gt_match_idx]['used']` уже учтено в gts_on_image_for_pred
                tp_class += 1
                gts_for_class_metrics[best_gt_match_idx]['used'] = True  # Отмечаем GT как использованный
            else:
                fp_class += 1  # Либо IoU низкий, либо нет подходящего GT (или он уже использован)

        fn_class = len(gts_for_class_metrics) - tp_class  # Все GT, которые не были сопоставлены с TP

        precision, recall, f1 = calculate_precision_recall_f1(tp_class, fp_class, fn_class)

        total_tp += tp_class
        total_fp += fp_class
        total_fn += fn_class

        print(f"\n--- Класс: {class_name_eval} (ID: {class_id_eval}) ---")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  (TP: {tp_class}, FP: {fp_class}, FN: {fn_class}, Total GT: {len(gts_for_class_metrics)})")

    # Расчет общих (Micro-averaged) Precision, Recall, F1
    micro_precision, micro_recall, micro_f1 = calculate_precision_recall_f1(total_tp, total_fp, total_fn)

    print("\n" + "=" * 30 + " Итоговые Метрики (Micro-Averaged) " + "=" * 30)
    print(f"Micro-Precision: {micro_precision:.4f}")
    print(f"Micro-Recall:    {micro_recall:.4f}")
    print(f"Micro-F1-score:  {micro_f1:.4f}")
    print(
        f"(Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}, Total GT Objects: {len(all_ground_truths)})")
    print("=" * 78)

    logger.info(f"Оценка завершена. Micro-F1 = {micro_f1:.4f}")
    return micro_f1  # Возвращаем Micro-F1


# --- Точка входа скрипта ---
if __name__ == '__main__':
    # Настройка логирования в консоль для запуска как main скрипт
    # Проверяем, не настроено ли уже (например, при запуске из другого скрипта)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__) # Получим логгер еще раз после настройки

    logger.info("Запуск evaluate_detector_v3_standard.py как основного скрипта.")

    # --- Пути к конфигам ---
    # Предполагаем, что конфиги находятся в src/configs относительно PROJECT_ROOT
    # и что PROJECT_ROOT был правильно определен выше.
    try:
        MAIN_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        PREDICT_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "predict_config.yaml"

        # Загружаем main_config здесь, чтобы проверить параметр best_model_filename, если он влияет на имя файла
        with open(MAIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
            temp_main_config_for_weights = yaml.safe_load(f)
            if not isinstance(temp_main_config_for_weights, dict):
                logger.error("Содержимое основного конфигурационного файла не является словарем.")
                sys.exit(1)
        logger.info(f"Основной конфигурационный файл для проверки параметров загружен: {MAIN_CONFIG_PATH}")

        # === [ИЗМЕНЕНО] Явно указываем имя файла с лучшими весами ===
        BEST_WEIGHTS_FILENAME = "best_model.keras.weights.h5"
        # Вы также можете взять его из конфига, если он там указан точно так:
        # BEST_WEIGHTS_FILENAME = temp_main_config_for_weights.get('best_model_filename', "best_model.keras.weights.h5")
        # Но если вы уверены, что файл называется именно "best_model.keras.weights.h5",
        # то явное указание надежнее.

        logger.info(f"Используемое имя файла с лучшими весами: {BEST_WEIGHTS_FILENAME}")

    except FileNotFoundError:
        logger.error(f"Один из конфигурационных файлов не найден. Проверьте пути.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при загрузке или обработке конфигурационных файлов: {e}")
        sys.exit(1)

    # Запускаем оценку
    evaluate_detector(
        main_config_path=MAIN_CONFIG_PATH,
        predict_config_path=PREDICT_CONFIG_PATH,
        weights_filename=BEST_WEIGHTS_FILENAME
    )

    # Очищаем sys.path, если мы его изменили в начале скрипта
    if _added_to_sys_path_eval:
         if str(PROJECT_ROOT) in sys.path: # Проверяем, что путь все еще там
             sys.path.pop(sys.path.index(str(PROJECT_ROOT))) # Удаляем по значению
             logger.debug(f"Путь {PROJECT_ROOT} удален из sys.path.")