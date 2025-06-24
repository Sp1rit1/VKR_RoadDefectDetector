# RoadDefectDetector/debug_single_image.py

import sys
import cv2
import yaml
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import random # Добавлено для установки сида random
import tensorflow as tf # Добавлено для установки сида tensorflow
import albumentations as A # Добавлено для установки сида Albumentations






# --- Настройка логирования ---
# Перенастроим логирование, чтобы сообщения не дублировались и были более читаемыми
# Имя логгера будет соответствовать имени файла ('debug_single_image')
logger = logging.getLogger(__name__)
# Установим уровень только если он не был установлен ранее (для избежания дублирования при запуске из main скрипта)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Отключим логирование из Albumentations если оно мешает
logging.getLogger('albumentations').setLevel(logging.WARNING)
# --- Настройка путей и импортов ---
# Убедимся, что корень проекта добавлен в sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
# Если скрипт запускается не из корня, нужно подняться на один уровень выше.
# Проверим, находится ли скрипт в подпапке src/utils, src/datasets и т.д.
# Если да, то корень проекта - на один уровень выше.
# Если скрипт находится непосредственно в корне, PROJECT_ROOT уже является корнем.
if PROJECT_ROOT.name in ['src', 'data', 'graphs', 'logs', 'weights', 'outputs', 'prediction_results_pipeline', 'runs']:
     # Значит, мы уже в корневой директории
     pass
elif PROJECT_ROOT.parent.name in ['src', 'data', 'graphs', 'logs', 'weights', 'outputs', 'prediction_results_pipeline', 'runs']:
     # Значит, мы в подпапке src/utils, src/datasets, src/models и т.д.
     PROJECT_ROOT = PROJECT_ROOT.parent
elif PROJECT_ROOT.parent.parent.name in ['src', 'data', 'graphs', 'logs', 'weights', 'outputs', 'prediction_results_pipeline', 'runs']:
     # Значит, мы в подпапке типа src/utils/main_utils
     PROJECT_ROOT = PROJECT_ROOT.parent.parent

# Добавляем корень проекта в sys.path, если его там нет
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    logger.info(f"Добавлен корень проекта в sys.path: {PROJECT_ROOT}")


try:
    # Импортируем нужные функции из data_loader_v3_standard
    from src.datasets.data_loader_v3_standard import (
        generate_all_anchors, parse_voc_xml, assign_gt_to_anchors,
        encode_box_targets # Возможно пригодится для тестов кодирования/декодирования
    )
    # Импортируем plot_utils (теперь находится в src/utils)
    from src.utils import plot_utils
    # Импортируем augmentations
    from src.datasets import augmentations
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}\nУбедитесь, что скрипт запускается из корня проекта и все зависимости установлены.")
    logger.error(f"Current sys.path: {sys.path}")
    logger.error(f"Attempted PROJECT_ROOT: {PROJECT_ROOT}")
    sys.exit(1)


# --- Установка сида для воспроизводимости (очень важно для отладки аугментации) ---
def set_debug_seed(seed):
    """Устанавливает сид для random, numpy, tensorflow и albumentations."""
    if seed is not None:
        logger.info(f"Установка отладочного сида: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Albumentations использует сид numpy.random.
        # Мы также можем явно установить сид Albumentations для надежности
        if hasattr(A, 'set_seed'):
             A.set_seed(seed)
        else:
             logger.warning("Albumentations.set_seed не найдена. Воспроизводимость аугментации может быть неполной.")
    else:
        logger.info("Отладочный сид НЕ установлен (аугментация будет случайной).")


# --- Основная функция отладки ---
def debug_one_image(config, image_name, use_augmentation=True, aug_seed=None, top_k_positive=10, top_k_ignored=10):
    logger.info(f"--- Запуск детального анализа для: {image_name} ---")

    # Устанавливаем сид для воспроизведения конкретного сценария аугментации
    set_debug_seed(aug_seed) # Используем aug_seed для фиксации всего

    dataset_path = Path(config['dataset_path'])
    # Проверяем в обеих папках (train/val)
    image_path_train = dataset_path / config['train_images_subdir'] / image_name
    annot_path_train = dataset_path / config['train_annotations_subdir'] / (Path(image_name).stem + ".xml")
    image_path_val = dataset_path / config['val_images_subdir'] / image_name
    annot_path_val = dataset_path / config['val_annotations_subdir'] / (Path(image_name).stem + ".xml")

    image_path = None
    annot_path = None

    if image_path_train.exists():
        image_path = image_path_train
        annot_path = annot_path_train
        logger.info(f"Изображение найдено в тренировочной выборке: {image_path}")
    elif image_path_val.exists():
        image_path = image_path_val
        annot_path = annot_path_val
        logger.info(f"Изображение найдено в валидационной выборке: {image_path}")
    else:
        logger.error(f"Изображение '{image_name}' не найдено ни в train, ни в validation.")
        return

    try:
        image_original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image_original.shape
        gt_boxes_original_pixels, gt_class_names_original = parse_voc_xml(annot_path)
        logger.info(f"Загружено изображение {image_name} ({w_orig}x{h_orig}) и {len(gt_boxes_original_pixels)} GT объектов.")

    except Exception as e:
        logger.error(f"Ошибка при загрузке или парсинге аннотаций для {image_name}: {e}")
        return


    h_target, w_target = config['input_shape'][:2]
    image_resized = cv2.resize(image_original, (w_target, h_target))
    logger.info(f"Изображение изменено до целевого размера {w_target}x{h_target}.")


    # Преобразуем оригинальные пиксельные рамки в рамки на ресайзнутом изображении (тоже пиксели)
    gt_boxes_resized_pixels = []
    if gt_boxes_original_pixels:
        for box in gt_boxes_original_pixels:
            x1, y1, x2, y2 = box
            # Используем float деление перед умножением для точности
            x1_r = (x1 / w_orig) * w_target
            y1_r = (y1 / h_orig) * h_target
            x2_r = (x2 / w_orig) * w_target
            y2_r = (y2 / h_orig) * h_target
            gt_boxes_resized_pixels.append([x1_r, y1_r, x2_r, y2_r])
    logger.info(f"GT рамки масштабированы до размера {w_target}x{h_target}.")


    # --- Применение Аугментации ---
    image_to_augment = image_resized
    boxes_to_augment = gt_boxes_resized_pixels # Ramki v pikseljah posle resize
    class_labels_for_aug = gt_class_names_original # Imena klassov

    image_augmented_uint8 = image_to_augment # Default, if no augmentation
    gt_boxes_augmented_pixels = boxes_to_augment
    gt_class_names_augmented = class_labels_for_aug


    if use_augmentation:
        # Albumentations ожидает uint8 RGB и рамки в пикселях в формате, указанном в BboxParams
        # Мы используем 'pascal_voc' [x_min, y_min, x_max, y_max]
        # Важно: сид Albumentations уже установлен функцией set_debug_seed
        try:
             augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
             if augmenter:
                 augmented = augmenter(image=image_to_augment, bboxes=boxes_to_augment,
                                       class_labels_for_albumentations=class_labels_for_aug)
                 image_augmented_uint8 = augmented['image']
                 gt_boxes_augmented_pixels = augmented['bboxes'] # Ramki v pikseljah posle augmentacii
                 gt_class_names_augmented = augmented['class_labels_for_albumentations'] # Imena klassov posle augmentacii (nekotorye mogut propast' iz-za min_visibility/min_area)
                 logger.info(f"Аугментация применена. Осталось {len(gt_boxes_augmented_pixels)} GT объектов.")
             else:
                logger.warning("Функция get_detector_train_augmentations вернула None. Аугментация не применена.")
                use_augmentation = False # Отключаем флаг, если аугментатор не загружен
        except Exception as e:
            logger.error(f"Ошибка при применении аугментации: {e}")
            # Продолжим без аугментации, если произошла ошибка
            use_augmentation = False
            image_augmented_uint8 = image_to_augment
            gt_boxes_augmented_pixels = boxes_to_augment
            gt_class_names_augmented = class_labels_for_aug
            logger.warning("Продолжаем без аугментации из-за ошибки.")
    # else: use_augmentation is False, no change needed to vars

    # --- Генерация якорей и Назначение GT ---

    # Генерируем все якоря (в нормализованных координатах)
    # Используем параметры из конфига (Scales+Ratios)
    fpn_strides = [8, 16, 32] # Стандартные шаги для P3, P4, P5
    # generate_all_anchors теперь использует Scales и Ratios из конфига
    all_anchors_norm = generate_all_anchors(
        config['input_shape'],
        fpn_strides,
        config['anchor_scales'], # Скалярные scales
        config['anchor_ratios']  # Ratios
    )
    logger.info(f"Сгенерировано {len(all_anchors_norm)} нормализованных якорей ({config['num_anchors_per_level']} на ячейку).")


    # Преобразуем аугментированные пиксельные рамки GT в нормализованные
    gt_boxes_aug_norm = np.array(gt_boxes_augmented_pixels, dtype=np.float32) / np.array(
        [w_target, h_target, w_target, h_target]) if gt_boxes_augmented_pixels else np.empty((0, 4), dtype=np.float32)

    # Преобразуем имена классов аугментированных GT в ID
    class_name_to_id = {name: i for i, name in enumerate(config['class_names'])}
    # Обработаем случай, когда после аугментации остались только объекты,
    # или когда gt_class_names_augmented пустой
    gt_class_ids_augmented = np.array([class_name_to_id[name] for name in gt_class_names_augmented],
                                      dtype=np.int32) if gt_class_names_augmented else np.empty((0,), dtype=np.int32)

    # ===> КОРРЕКТНЫЙ ВЫЗОВ assign_gt_to_anchors <===
    # Передаем все необходимые 5 аргументов
    # Распаковываем все 4 возвращаемых значения
    anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor = assign_gt_to_anchors(
        gt_boxes_aug_norm,           # 1. Нормализованные рамки GT после аугментации
        gt_class_ids_augmented,      # 2. ID классов GT после аугментации (ИСПРАВЛЕНО)
        all_anchors_norm,            # 3. Все нормализованные якоря
        config['anchor_positive_iou_threshold'], # 4. Порог для позитивных
        config['anchor_ignore_iou_threshold']  # 5. Порог для игнорируемых (ИСПРАВЛЕНО)
    )
    logger.info(f"Назначение GT якорям завершено.")


    # --- Анализ и подготовка данных для визуализации ---

    # Преобразуем нормализованные якоря обратно в пиксельные координаты для отрисовки
    all_anchors_pixels = all_anchors_norm * np.array([w_target, h_target, w_target, h_target], dtype=np.float32)

    # Находим индексы якорей по меткам
    positive_indices = np.where(anchor_labels == 1)[0]
    ignored_indices = np.where(anchor_labels == 0)[0]
    negative_indices = np.where(anchor_labels == -1)[0] # Добавим негативные для статистики

    logger.info(f"Количество назначенных якорей:")
    logger.info(f"  - Позитивных: {len(positive_indices)}")
    logger.info(f"  - Игнорируемых: {len(ignored_indices)}")
    logger.info(f"  - Негативных: {len(negative_indices)}")


    # Сортируем индексы позитивных и игнорируемых якорей по убыванию IoU для выбора "топ-K"
    # Убедимся, что массив max_iou_per_anchor имеет ту же длину, что и anchor_labels и all_anchors_norm
    if len(max_iou_per_anchor) != len(all_anchors_norm):
         logger.error("Длина max_iou_per_anchor не соответствует количеству якорей! Ошибка в assign_gt_to_anchors?")
         # Продолжим, но результаты могут быть некорректными
         # Создадим массив нулей правильной длины, чтобы избежать IndexError далее
         max_iou_per_anchor = np.zeros_like(anchor_labels, dtype=np.float32)


    # Убедимся, что индексы не выходят за границы max_iou_per_anchor (хотя по идее не должны)
    positive_indices_valid = positive_indices[positive_indices < len(max_iou_per_anchor)]
    ignored_indices_valid = ignored_indices[ignored_indices < len(max_iou_per_anchor)]

    # Сортируем индексы по убыванию IoU
    sorted_pos_indices = positive_indices_valid[np.argsort(-max_iou_per_anchor[positive_indices_valid])]
    sorted_ign_indices = ignored_indices_valid[np.argsort(-max_iou_per_anchor[ignored_indices_valid])]


    # Выбираем топ-K якорей для отрисовки и собираем информацию о них для plot_specific_anchors_on_image
    # plot_specific_anchors_on_image ожидает список словарей с ключами 'bbox', 'type', 'iou' (опц.), 'level' (опц.)
    top_pos_info = []
    for i in sorted_pos_indices[:top_k_positive]:
        top_pos_info.append({
            'bbox': all_anchors_pixels[i], # Пиксельные координаты для отрисовки
            'type': 'positive', # Тип якоря для выбора цвета
            'iou': max_iou_per_anchor[i], # IoU для отображения в тексте
            # 'level': 'N/A' # Пока не можем легко определить уровень якоря здесь
        })


    top_ign_info = []
    for i in sorted_ign_indices[:top_k_ignored]:
         top_ign_info.append({
             'bbox': all_anchors_pixels[i], # Пиксельные координаты
             'type': 'ignored', # Тип якоря
             'iou': max_iou_per_anchor[i], # IoU
             # 'level': 'N/A'
         })

    # Объединяем списки для отрисовки
    anchors_to_plot = top_pos_info + top_ign_info


    # --- Визуализация ---
    logger.info("Подготовка визуализации...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # 1 строка, 2 столбца для двух графиков
    # Общий заголовок для всей фигуры
    fig.suptitle(f"Детальный анализ назначения якорей для: {image_name}\n"
                 f"Аугментация: {'ВКЛ' if use_augmentation else 'ВЫКЛ'} | Сид: {aug_seed if use_augmentation and aug_seed is not None else 'N/A'}",
                 fontsize=16) # Покажем сид только если аугментация включена и сид задан


    # График 1: Исходное изображение с оригинальным GT
    # plot_utils.plot_original_gt ожидает image_original (uint8 RGB) и список объектов GT
    plot_utils.plot_original_gt(axes[0], image_original,
                                [{'bbox': b, 'class': l} for b, l in zip(gt_boxes_original_pixels, gt_class_names_original)])
    # plot_utils.plot_original_gt уже устанавливает заголовок "Исходное изображение + GT"


    # График 2: Аугментированное изображение с аугментированным GT и топ-якорями
    # image_augmented_uint8 - это uint8 RGB после аугментации (как ожидает plot_utils)
    # plot_utils.plot_augmented_gt ожидает image_augmented_uint8 и список объектов GT
    # plot_specific_anchors_on_image ожидает image_augmented_uint8 и список информации о якорях

    # Начинаем с отрисовки аугментированного изображения и аугментированного GT с помощью plot_augmented_gt
    plot_utils.plot_augmented_gt(axes[1], image_augmented_uint8,
                                [{'bbox': b, 'class': l} for b, l in zip(gt_boxes_augmented_pixels, gt_class_names_augmented)])
    # plot_utils.plot_augmented_gt уже устанавливает заголовок "Аугментированное изображение + GT"

    # Теперь отрисовываем топ-якоря поверх аугментированного изображения и GT
    # plot_specific_anchors_on_image сам отрисовывает изображение, но если ax уже содержит изображение, он просто добавит поверх.
    # Чтобы избежать двойной отрисовки изображения, можно передать None вместо image_augmented_uint8,
    # НО plot_specific_anchors_on_image использует image_np для настройки размеров осей.
    # Лучше явно не задавать title в plot_specific_anchors_on_image, если он уже задан plot_augmented_gt.
    # В вашей версии plot_specific_anchors_on_image заголовок по умолчанию "Якоря",
    # а в вызове plot_specific_anchors_on_image(..., title="") он пустой, так что это нормально.
    plot_utils.plot_specific_anchors_on_image(axes[1], image_augmented_uint8, anchors_to_plot, title="")


    # --- Вывод детальной информации в консоль ---
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ЯКОРЯХ С IoU >= {} (отсортировано по IoU)".format(config['anchor_ignore_iou_threshold']).center(80))
    print("=" * 80)
    # Фильтруем якоря с IoU >= neg_iou_thresh для более осмысленного вывода
    meaningful_iou_indices = np.where(max_iou_per_anchor >= config['anchor_ignore_iou_threshold'])[0]
    meaningful_anchors_info = []
    class_id_to_name = {i: name for i, name in enumerate(config['class_names'])} # Создаем mapping ID -> Name


    for idx in meaningful_iou_indices:
        anchor_type = {1: 'positive', 0: 'ignored', -1: 'negative'}.get(anchor_labels[idx], 'unknown')
        iou_val = max_iou_per_anchor[idx]
        bbox_pixels = [f"{c:.1f}" for c in all_anchors_pixels[idx]]
        bbox_string = f"[{', '.join(bbox_pixels)}]"
        info_str = ""

        if anchor_type == 'positive':
             # Для позитивных якорей покажем, с каким GT они сопоставлены и его класс
             matched_gt_class_id = matched_gt_class_ids_for_anchors[idx]
             matched_gt_class_name = class_id_to_name.get(matched_gt_class_id, f'Unknown Class ID:{matched_gt_class_id}')
             info_str = f" | Matched GT Class: {matched_gt_class_name}"
             # Можно также добавить информацию о сопоставленном GT боксе, если нужно
             # matched_gt_bbox_norm = matched_gt_boxes_for_anchors[idx]
             # matched_gt_bbox_pixels = [f"{c:.1f}" for c in matched_gt_bbox_norm * np.array([w_target, h_target, w_target, h_target])]
             # info_str += f" | Matched GT bbox (pix): [{', '.join(matched_gt_bbox_pixels)}]"


        meaningful_anchors_info.append({
             'type': anchor_type,
             'iou': iou_val,
             'bbox_pixels': bbox_string,
             'info': info_str # Дополнительная информация
        })

    # Сортируем по убыванию IoU
    meaningful_anchors_info.sort(key=lambda x: x['iou'], reverse=True)

    if not meaningful_anchors_info:
        print(f"Нет якорей с IoU >= {config['anchor_ignore_iou_threshold']:.2f}.")
    else:
        # Печатаем только топ-K или все, если их мало
        print(f"Вывод топ-{len(meaningful_anchors_info)} якорей с IoU >= {config['anchor_ignore_iou_threshold']:.2f}:")
        for info in meaningful_anchors_info:
            # Выводим информацию в форматированном виде
            print(f"  - Тип: {info['type'].capitalize():<9} | IoU: {info['iou']:.4f} | Координаты (пикс): {info['bbox_pixels']}{info['info']}")


    print("=" * 80 + "\n")


    # --- Отображение графика ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Корректируем расположение элементов, чтобы заголовки не перекрывались

    # Сохранение графика (опционально)
    # output_dir = PROJECT_ROOT / "graphs" / "debug_analysis"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # # Формируем имя файла с учетом сида аугментации, если он задан
    # aug_seed_str = f"_seed_{aug_seed}" if use_augmentation and aug_seed is not None else ""
    # plot_path = output_dir / f"{Path(image_name).stem}_debug_analysis{aug_seed_str}.png"
    # logger.info(f"Попытка сохранения графика в {plot_path}...")
    # plot_utils.save_plot(fig, plot_path) # Используем функцию сохранения из plot_utils

    plot_utils.show_plot() # Отобразить окно с графиком


    logger.info(f"--- Детальный анализ для: {image_name} завершен ---")


# --- Точка входа скрипта ---
if __name__ == '__main__':
    # Настройка логирования в консоль для запуска как main скрипт
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Получим логгер еще раз после настройки
    logger = logging.getLogger(__name__)


    try:
        # Загружаем основной конфиг
        config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        if not config_path.exists():
             # Проверяем альтернативный путь, если скрипт в корне, а конфиги в src/configs
             config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
             if not config_path.exists():
                  logger.error(f"Конфигурационный файл не найден ни по одному из путей: {PROJECT_ROOT / 'src' / 'configs' / 'detector_config_v3_standard.yaml'} или {PROJECT_ROOT / 'src' / 'configs' / 'detector_config_v3_standard.yaml'}")
                  sys.exit(1)


        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        logger.info("Конфигурационный файл успешно загружен.")
    except FileNotFoundError:
        # Этот блок уже обработан выше, но оставим на всякий случай
        logger.error(f"Конфигурационный файл не найден по пути: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка загрузки или парсинга конфига: {e}")
        sys.exit(1)


    # --- Параметры запуска отладки ---
    # Замените на имя изображения из вашего датасета для отладки
    IMAGE_TO_DEBUG = "n_Pithole_Road_571.jpg" # Пример имени файла из вашего датасета
    USE_AUGMENTATION = True # Установить True, чтобы включить аугментацию
    AUGMENTATION_SEED = None # Установить конкретное число (int) для фиксации аугментации, None для случайной


    debug_one_image(
        config=main_config,
        image_name=IMAGE_TO_DEBUG,
        use_augmentation=USE_AUGMENTATION,
        aug_seed=AUGMENTATION_SEED, # Передаем сид в функцию отладки
    )