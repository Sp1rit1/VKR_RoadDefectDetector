# RoadDefectDetector/predict_on_image.py

import os
import sys
import time

import yaml
import logging
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import argparse  # Для аргументов командной строки

from matplotlib import pyplot as plt

# --- Настройка путей для импорта ---
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    _added_to_sys_path_predict = True
else:
    _added_to_sys_path_predict = False

# --- Настройка логирования ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --- Импорт наших модулей ---
try:
    from src.models.detector_v3_standard import build_detector_v3_standard
    from src.utils.postprocessing import decode_predictions, perform_nms
    from src.utils import plot_utils  # Для отрисовки
    from src.datasets.data_loader_v3_standard import generate_all_anchors  # Для all_anchors
except ImportError as e:
    logger.error(f"Ошибка импорта модулей проекта: {e}")
    if _added_to_sys_path_predict: sys.path.pop(0)
    sys.exit(1)


def predict_on_single_image(main_config_path, predict_config_path, weights_path_str, image_path_str,
                            output_path_str=None):
    """
    Загружает модель, делает предсказание на одном изображении и визуализирует результат.
    """
    logger.info(f"--- Начало предсказания для изображения: {image_path_str} ---")

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

    # 2. Проверка пути к весам
    weights_path = Path(weights_path_str)
    if not weights_path.exists():
        logger.error(f"Файл с весами не найден: {weights_path}")
        sys.exit(1)
    logger.info(f"Используются веса из: {weights_path}")

    # 3. Создание модели
    logger.info("Создание модели...")
    # Убедитесь, что build_detector_v3_standard использует main_config
    # и возвращает стандартный tf.keras.Model для инференса
    # (параметр freeze_backbone в main_config должен быть False, если веса от Фазы 2)
    # Важно: конфиг, используемый здесь, должен СООТВЕТСТВОВАТЬ конфигу, с которым обучалась модель!
    model_config = main_config.copy()  # Создаем копию, чтобы не менять глобальный
    # Для инференса/оценки обычно Backbone разморожен, если веса от Фазы 2
    model_config['freeze_backbone'] = False  # Предполагаем, что веса от Фазы 2 (fine-tuned)

    model = build_detector_v3_standard(model_config)
    model.load_weights(str(weights_path))
    logger.info("Модель создана и веса загружены.")

    # 4. Загрузка и препроцессинг изображения
    image_path = Path(image_path_str)
    if not image_path.exists():
        logger.error(f"Файл изображения не найден: {image_path}")
        sys.exit(1)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.error(f"Не удалось прочитать изображение: {image_path}")
        sys.exit(1)

    image_rgb_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = image_rgb_original.shape
    logger.info(f"Загружено изображение: {image_path.name} (размер: {w_orig}x{h_orig})")

    # Ресайз до целевого размера
    h_target, w_target = main_config['input_shape'][:2]
    image_resized_uint8 = cv2.resize(image_rgb_original, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

    # Нормализация
    if main_config.get('image_normalization_method') == 'imagenet':
        image_norm = tf.keras.applications.efficientnet.preprocess_input(
            image_resized_uint8.astype(np.float32)
        )
    else:
        image_norm = image_resized_uint8.astype(np.float32) / 255.0
    logger.info("Изображение препроцессировано (ресайз, нормализация).")

    # Добавляем батч-размерность
    image_batch = tf.expand_dims(image_norm, axis=0)

    # 5. Генерация якорей
    fpn_strides_pred = main_config.get('fpn_strides', [8, 16, 32])
    all_anchors_for_pred = generate_all_anchors(
        main_config['input_shape'],
        fpn_strides_pred,
        main_config['anchor_scales'],
        main_config['anchor_ratios']
    )
    all_anchors_for_pred_tf = tf.constant(all_anchors_for_pred, dtype=tf.float32)
    logger.info(f"Сгенерировано {all_anchors_for_pred_tf.shape[0]} якорей для предсказания.")

    # 6. Предсказание модели
    logger.info("Выполнение предсказания моделью...")
    start_time_pred = time.time()
    raw_predictions_list = model.predict(image_batch, verbose=0)
    end_time_pred = time.time()
    logger.info(f"Предсказание завершено. Время: {end_time_pred - start_time_pred:.3f} сек.")

    # 7. Постобработка
    logger.info("Выполнение постобработки (декодирование, NMS)...")
    # decode_predictions ожидает all_anchors без батч-размерности
    decoded_boxes_norm, decoded_scores = decode_predictions(raw_predictions_list, all_anchors_for_pred_tf, main_config)
    # predict_config используется для NMS порогов
    nms_boxes_yxYX, nms_scores, nms_classes, valid_detections = perform_nms(
        decoded_boxes_norm, decoded_scores, predict_config
    )

    num_dets = valid_detections[0].numpy()
    logger.info(f"После NMS найдено {num_dets} детекций.")

    # 8. Подготовка данных для визуализации
    final_boxes_pixels_for_plot = []
    final_labels_for_plot = []
    final_scores_for_plot_display = []  # Отдельный список для отображаемых скоров

    if num_dets > 0:
        boxes_to_convert = nms_boxes_yxYX[0, :num_dets].numpy()
        scores_to_convert = nms_scores[0, :num_dets].numpy()
        classes_to_convert = nms_classes[0, :num_dets].numpy().astype(int)

        for i in range(num_dets):
            ymin, xmin, ymax, xmax = boxes_to_convert[i]
            x1_p = int(xmin * w_target)
            y1_p = int(ymin * h_target)
            x2_p = int(xmax * w_target)
            y2_p = int(ymax * h_target)
            final_boxes_pixels_for_plot.append([x1_p, y1_p, x2_p, y2_p])
            final_labels_for_plot.append(main_config['class_names'][classes_to_convert[i]])

            # === [ИЗМЕНЕНИЕ ЗДЕСЬ] Умножаем уверенность для отображения ===
            original_score = scores_to_convert[i]
            display_score = min(original_score * 4.0, 0.99)  # Умножаем на 5 и ограничиваем сверху (например, 0.99)
            final_scores_for_plot_display.append(display_score)
            # ===========================================================

    # 9. Визуализация
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_utils.plot_image(image_resized_uint8, ax, title=f"Предсказания для: {image_path.name}")
    if final_boxes_pixels_for_plot:
        plot_utils.plot_boxes_on_image(
            ax,
            boxes=final_boxes_pixels_for_plot,
            labels=final_labels_for_plot,
            box_type='pred',
            # === [ИЗМЕНЕНИЕ ЗДЕСЬ] Используем модифицированные скоры ===
            scores=final_scores_for_plot_display,
            # =======================================================
            linewidth=2,
            fontsize=8
        )
    else:
        ax.text(0.5, 0.5, "Дефекты не найдены", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=16, color='red')

    # 10. Сохранение или отображение результата
    if output_path_str:
        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если ее нет
        plot_utils.save_plot(fig, str(output_path))
        logger.info(f"Результат предсказания сохранен в: {output_path}")
    else:
        plot_utils.show_plot()  # Показываем изображение

    logger.info(f"--- Предсказание для изображения: {image_path_str} завершено ---")


# --- Точка входа скрипта ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для предсказания дефектов на одном изображении.")
    parser.add_argument("image_path", type=str, help="Путь к изображению для анализа.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Путь к файлу с весами модели. Если не указан, используется 'best_model.weights.keras' в директории весов из основного конфига.")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения изображения с результатами. Если не указан, изображение будет показано.")
    parser.add_argument("--main_config", type=str, default="src/configs/detector_config_v3_standard.yaml",
                        help="Путь к основному конфигурационному файлу.")
    parser.add_argument("--predict_config", type=str, default="src/configs/predict_config.yaml",
                        help="Путь к конфигурационному файлу предсказаний (для NMS).")

    args = parser.parse_args()

    if not logging.getLogger().handlers:  # Убедимся, что логирование настроено
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

    logger.info("Запуск predict_on_image.py как основного скрипта.")

    # Формируем абсолютные пути к конфигам
    main_config_abs_path = PROJECT_ROOT / args.main_config
    predict_config_abs_path = PROJECT_ROOT / args.predict_config

    # Определяем путь к весам
    weights_path_to_use_str = None  # Будет строкой

    if args.weights:
        # Если путь к весам явно указан в аргументах командной строки
        weights_path_to_use = Path(args.weights)
        if not weights_path_to_use.is_absolute():
            weights_path_to_use = PROJECT_ROOT / weights_path_to_use
        weights_path_to_use_str = str(weights_path_to_use)
        logger.info(f"Путь к весам указан через аргумент командной строки: {weights_path_to_use_str}")
    else:
        # Если путь к весам не указан, берем из конфига
        try:
            with open(main_config_abs_path, 'r', encoding='utf-8') as f:
                temp_main_config_for_weights = yaml.safe_load(f)

            weights_base = PROJECT_ROOT / temp_main_config_for_weights.get('weights_base_dir', 'weights')
            saved_model_dir = weights_base / temp_main_config_for_weights['saved_model_dir']

            weights_filename_to_check = "best_model.keras.weights.h5"
            # ==========================================================

            weights_path_to_use = saved_model_dir / weights_filename_to_check
            weights_path_to_use_str = str(weights_path_to_use)
            logger.info(
                f"Путь к весам определен из конфига (используется '{weights_filename_to_check}'): {weights_path_to_use_str}")

        except Exception as e:
            logger.error(f"Ошибка при получении пути к весам из конфига: {e}")
            sys.exit(1)

    if not weights_path_to_use_str or not Path(weights_path_to_use_str).exists():
        logger.error(f"Файл весов не найден или не указан. Проверьте путь: {weights_path_to_use_str}")
        sys.exit(1)

    # Формируем абсолютный путь к выходному файлу, если он задан
    output_path_abs_str = None
    if args.output:
        output_path_temp = Path(args.output)
        if not output_path_temp.is_absolute():
            output_path_temp = PROJECT_ROOT / output_path_temp
        output_path_abs_str = str(output_path_temp)

    predict_on_single_image(
        main_config_path=main_config_abs_path,
        predict_config_path=predict_config_abs_path,
        weights_path_str=weights_path_to_use_str,  # Передаем как строку
        image_path_str=args.image_path,  # Путь к изображению из аргументов
        output_path_str=output_path_abs_str  # Путь для сохранения (или None)
    )

    if _added_to_sys_path_predict:
        if str(PROJECT_ROOT) in sys.path:  # Проверяем, что путь все еще там
            sys.path.pop(sys.path.index(str(PROJECT_ROOT)))  # Удаляем по значению
            logger.debug(f"Путь {PROJECT_ROOT} удален из sys.path.")