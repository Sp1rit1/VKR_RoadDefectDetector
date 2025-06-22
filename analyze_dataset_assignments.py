import sys
import yaml
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Настройка путей и импортов ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import create_dataset
    from src.utils import plot_utils
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}\nУбедитесь, что скрипт запускается из корня проекта.")
    sys.exit(1)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Основная функция анализа ---

def analyze_assignments(config, num_images_to_check=100, problem_threshold=1, max_visualize=20):
    logger.info("--- Запуск РАСШИРЕННОГО анализа назначений якорей ---")

    logger.info(f"Создание датасета для анализа {num_images_to_check} изображений...")
    dataset = create_dataset(config, is_training=True, debug_mode=True).take(num_images_to_check)

    problem_scenarios = []
    # --- Новые структуры для расширенной статистики ---
    # defaultdict(lambda: ...) создает словарь с дефолтными значениями, если ключ не найден
    cumulative_stats = defaultdict(lambda: {'positive': 0, 'ignored': 0, 'negative': 0})
    best_case = {'path': None, 'pos_count': -1}
    worst_case_non_zero = {'path': None, 'pos_count': float('inf')}
    max_ignored_case = {'path': None, 'ign_count': -1}

    processed_count = 0

    graphs_dir = PROJECT_ROOT / "graphs" / "problem_scenarios"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Графики проблемных сценариев будут сохранены в: {graphs_dir}")

    # --- Определение границ уровней FPN для статистики ---
    # Мы знаем, что якоря генерируются последовательно: сначала все для P3, потом для P4, потом для P5.
    strides = [8, 16, 32]
    input_h, input_w = config['input_shape'][:2]
    num_anchors_per_loc = len(config['anchor_scales']) * len(config['anchor_ratios'])

    level_boundaries = {}
    current_idx = 0
    for i, stride in enumerate(strides):
        level_name = f"P{i + 3}"
        num_cells = (input_h // stride) * (input_w // stride)
        num_anchors_in_level = num_cells * num_anchors_per_loc
        level_boundaries[level_name] = (current_idx, current_idx + num_anchors_in_level)
        current_idx += num_anchors_in_level

    # --- Основной цикл анализа ---
    for i, (image_norm, y_true, debug_info) in enumerate(dataset):
        processed_count += 1
        image_path = debug_info["image_path"].numpy().decode('utf-8')
        print(f"\rОбработка: {i + 1}/{num_images_to_check} - {Path(image_path).name}", end="")

        anchor_labels = debug_info["anchor_labels"].numpy()
        gt_boxes_aug = debug_info["gt_boxes_augmented"].numpy()

        # --- Сбор детальной статистики по уровням ---
        total_pos_current = 0
        total_ign_current = 0
        for level_name, (start_idx, end_idx) in level_boundaries.items():
            level_labels = anchor_labels[start_idx:end_idx]
            pos_count = np.sum(level_labels == 1)
            ign_count = np.sum(level_labels == 0)
            neg_count = np.sum(level_labels == -1)

            cumulative_stats[level_name]['positive'] += pos_count
            cumulative_stats[level_name]['ignored'] += ign_count
            cumulative_stats[level_name]['negative'] += neg_count

            total_pos_current += pos_count
            total_ign_current += ign_count

        # --- Поиск лучших/худших случаев ---
        if total_pos_current > best_case['pos_count']:
            best_case = {'path': image_path, 'pos_count': total_pos_current}
        if 0 < total_pos_current < worst_case_non_zero['pos_count']:
            worst_case_non_zero = {'path': image_path, 'pos_count': total_pos_current}
        if total_ign_current > max_ignored_case['ign_count']:
            max_ignored_case = {'path': image_path, 'ign_count': total_ign_current}

        # --- Проверка на "проблемность" ---
        if len(gt_boxes_aug) > 0 and total_pos_current < problem_threshold:
            logger.warning(
                f"\n! ПРОБЛЕМНЫЙ СЦЕНАРИЙ: {Path(image_path).name} | GT: {len(gt_boxes_aug)}, Positives: {total_pos_current}")
            problem_scenarios.append(
                {'image_path': image_path, 'num_gt': len(gt_boxes_aug), 'num_positives': total_pos_current,
                 'debug_info': debug_info})

    print("\n\nАнализ завершен.")
    logger.info(f"Всего обработано изображений: {processed_count}")
    logger.info(f"Найдено проблемных сценариев (positive < {problem_threshold}): {len(problem_scenarios)}")

    # --- Визуализация проблемных сценариев ---
    if problem_scenarios:
        # ... (код визуализации остается без изменений) ...
        pass

    # --- Вывод расширенной статистики ---
    print("\n" + "=" * 80)
    print("СВОДНАЯ СТАТИСТИКА НАЗНАЧЕНИЙ (в среднем на одно изображение)".center(80))
    print("=" * 80)
    if processed_count > 0:
        total_pos_all_levels, total_ign_all_levels, total_neg_all_levels = 0, 0, 0

        print(
            f"{'Уровень':<10} | {'Позитивных':<15} | {'Игнорируемых':<15} | {'Негативных':<15} | {'Всего якорей':<15}")
        print("-" * 80)

        for level_name, (start_idx, end_idx) in level_boundaries.items():
            stats = cumulative_stats[level_name]
            avg_pos = stats['positive'] / processed_count
            avg_ign = stats['ignored'] / processed_count
            avg_neg = stats['negative'] / processed_count
            total_anchors_level = end_idx - start_idx

            total_pos_all_levels += avg_pos
            total_ign_all_levels += avg_ign
            total_neg_all_levels += avg_neg

            print(
                f"{level_name:<10} | {avg_pos:<15.2f} | {avg_ign:<15.2f} | {avg_neg:<15.2f} | {total_anchors_level:<15}")

        print("-" * 80)
        print(
            f"{'ИТОГО':<10} | {total_pos_all_levels:<15.2f} | {total_ign_all_levels:<15.2f} | {total_neg_all_levels:<15.2f} | {current_idx:<15}")
        print("-" * 80)

        if total_pos_all_levels > 5:
            logger.info("ИТОГОВЫЙ ВЕРДИКТ: Отличный результат! В среднем назначается много позитивных якорей.")
        elif total_pos_all_levels > 1:
            logger.warning("ИТОГОВЫЙ ВЕРДИКТ: Результат приемлемый, но стоит проверить проблемные сценарии.")
        else:
            logger.error(
                "ИТОГОВЫЙ ВЕРДИКТ: Критически мало позитивных якорей! Нужно разбираться с порогами IoU или аугментациями.")

    else:
        print("Нет данных для статистики.")

    # --- Вывод информации о лучших/худших случаях ---
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВЫДАЮЩИХСЯ СЛУЧАЕВ".center(80))
    print("=" * 80)
    if best_case['path']:
        print(f"  - ЛУЧШИЙ СЛУЧАЙ (макс. позитивных):")
        print(f"    - Файл: {Path(best_case['path']).name}")
        print(f"    - Кол-во позитивных якорей: {best_case['pos_count']}")
    else:
        print("  - Лучший случай не найден (не было позитивных якорей).")

    if worst_case_non_zero['path']:
        print(f"\n  - ХУДШИЙ СЛУЧАЙ (мин. позитивных > 0):")
        print(f"    - Файл: {Path(worst_case_non_zero['path']).name}")
        print(f"    - Кол-во позитивных якорей: {worst_case_non_zero['pos_count']}")
    else:
        print("  - Худший случай не найден (все изображения имели 0 позитивных якорей).")

    if max_ignored_case['path']:
        print(f"\n  - МАКСИМУМ ИГНОРИРУЕМЫХ:")
        print(f"    - Файл: {Path(max_ignored_case['path']).name}")
        print(f"    - Кол-во игнорируемых якорей: {max_ignored_case['ign_count']}")
    else:
        print("  - Случай с макс. игнорируемых не найден.")
    print("=" * 80 + "\n")


# --- Запуск анализа ---
if __name__ == '__main__':
    try:
        config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Конфигурационный файл не найден по пути: {config_path}")
        sys.exit(1)

    NUM_IMAGES_TO_ANALYZE = 100
    PROBLEM_POSITIVE_THRESHOLD = 1
    MAX_VISUALIZATIONS = 20

    analyze_assignments(
        config=main_config,
        num_images_to_check=NUM_IMAGES_TO_ANALYZE,
        problem_threshold=PROBLEM_POSITIVE_THRESHOLD,
        max_visualize=MAX_VISUALIZATIONS
    )