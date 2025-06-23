import sys
from pathlib import Path

# --- [ИЗМЕНЕНИЕ] Надежное определение корня проекта ---
# Ищем корень проекта, двигаясь вверх от текущего файла, пока не найдем папку 'src'
try:
    # Этот путь будет работать, если скрипт лежит в корне или в подпапке
    current_path = Path(__file__).resolve()
    PROJECT_ROOT = current_path
    while not (PROJECT_ROOT / 'src').exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
        if PROJECT_ROOT == PROJECT_ROOT.parent:  # Дошли до корня диска
            raise FileNotFoundError
except (NameError, FileNotFoundError):
    # Если __file__ не определен (например, в интерактивной сессии), используем cwd
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / 'src').exists():
        print("Ошибка: Не удалось автоматически определить корень проекта. Убедитесь, что папка 'src' существует.")
        sys.exit(1)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

# --- Импорты из нашего проекта ---
try:
    from src.datasets.data_loader_v3_standard import DataGenerator, generate_all_anchors, assign_gt_to_anchors, \
        parse_voc_xml
    from src.utils import plot_utils
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}\nУбедитесь, что все зависимости установлены.")
    sys.exit(1)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Основная функция анализа ---
def analyze_assignments(config, num_images_to_check=100, problem_threshold=1, max_visualize=20):
    logger.info("--- Запуск РАСШИРЕННОГО анализа назначений якорей ---")

    logger.info(f"Создание генератора для анализа {num_images_to_check} аугментированных изображений...")
    all_anchors = generate_all_anchors(
        config['input_shape'], [8, 16, 32], config['anchor_scales'], config['anchor_ratios']
    )
    generator = DataGenerator(config, all_anchors, is_training=True)

    if len(generator) == 0:
        logger.error("Генератор данных не нашел ни одного изображения. Проверьте пути в конфиге.")
        return

    problem_scenarios = []
    cumulative_stats = defaultdict(lambda: {'positive': 0, 'ignored': 0, 'negative': 0})
    best_case = {'path': None, 'pos_count': -1}
    worst_case_non_zero = {'path': None, 'pos_count': float('inf')}
    max_ignored_case = {'path': None, 'ign_count': -1}
    processed_count = 0

    graphs_dir = PROJECT_ROOT / "graphs" / "assignment_analysis"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Графики проблемных сценариев будут сохранены в: {graphs_dir}")

    strides = [8, 16, 32]
    input_h, input_w = config['input_shape'][:2]
    num_anchors_per_loc = config['num_anchors_per_level']
    level_boundaries = {}
    current_idx = 0
    for i, stride in enumerate(strides):
        level_name = f"P{i + 3}"
        num_cells = (input_h // stride) * (input_w // stride)
        num_anchors_in_level = num_cells * num_anchors_per_loc
        level_boundaries[level_name] = (current_idx, current_idx + num_anchors_in_level)
        current_idx += num_anchors_in_level

    # Основной цикл анализа
    for i in range(min(num_images_to_check, len(generator))):
        processed_count += 1

        image_path = generator.image_paths[i]
        annot_path = generator.annot_paths[i]

        image_original = cv2.imread(str(image_path));
        image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image_rgb.shape
        gt_boxes_pixels, gt_class_names = parse_voc_xml(annot_path)
        image_resized = cv2.resize(image_rgb, (input_w, input_h))
        gt_boxes_resized_pixels = [
            [(b[0] / w_orig) * input_w, (b[1] / h_orig) * input_h, (b[2] / w_orig) * input_w, (b[3] / h_orig) * input_h]
            for b in gt_boxes_pixels]

        if generator.augmenter and gt_boxes_resized_pixels:
            augmented = generator.augmenter(image=image_resized, bboxes=gt_boxes_resized_pixels,
                                            class_labels_for_albumentations=gt_class_names)
            image_final = augmented['image']
            gt_boxes_final_pixels = augmented['bboxes']
        else:
            image_final = image_resized
            gt_boxes_final_pixels = gt_boxes_resized_pixels

        gt_boxes_aug = np.array(gt_boxes_final_pixels)
        gt_boxes_norm = gt_boxes_aug / np.array([input_w, input_h, input_w, input_h]) if len(
            gt_boxes_final_pixels) > 0 else np.empty((0, 4))
        gt_class_ids = np.array([generator.class_mapping.get(name) for name in gt_class_names], dtype=np.int32)

        anchor_labels, _, _, _ = assign_gt_to_anchors(
            gt_boxes_norm, gt_class_ids, all_anchors,
            config['anchor_positive_iou_threshold'], config['anchor_ignore_iou_threshold']
        )

        print(f"\rОбработка: {i + 1}/{num_images_to_check} - {Path(image_path).name}", end="")

        total_pos_current, total_ign_current = 0, 0
        for level_name, (start_idx, end_idx) in level_boundaries.items():
            pos_count = np.sum(anchor_labels[start_idx:end_idx] == 1)
            ign_count = np.sum(anchor_labels[start_idx:end_idx] == 0)
            neg_count = np.sum(anchor_labels[start_idx:end_idx] == -1)
            cumulative_stats[level_name]['positive'] += pos_count
            cumulative_stats[level_name]['ignored'] += ign_count
            cumulative_stats[level_name]['negative'] += neg_count
            total_pos_current += pos_count
            total_ign_current += ign_count

        if total_pos_current > best_case['pos_count']: best_case = {'path': str(image_path),
                                                                    'pos_count': total_pos_current}
        if 0 < total_pos_current < worst_case_non_zero['pos_count']: worst_case_non_zero = {'path': str(image_path),
                                                                                            'pos_count': total_pos_current}
        if total_ign_current > max_ignored_case['ign_count']: max_ignored_case = {'path': str(image_path),
                                                                                  'ign_count': total_ign_current}

        if len(gt_boxes_aug) > 0 and total_pos_current < problem_threshold:
            logger.warning(
                f"\n! ПРОБЛЕМНЫЙ СЦЕНАРИЙ: {Path(image_path).name} | GT: {len(gt_boxes_aug)}, Positives: {total_pos_current}")
            problem_scenarios.append(
                {'image_path': str(image_path), 'num_gt': len(gt_boxes_aug), 'num_positives': total_pos_current})

    print("\n\nАнализ завершен.")
    logger.info(f"Всего обработано изображений: {processed_count}")
    logger.info(f"Найдено проблемных сценариев (positive < {problem_threshold}): {len(problem_scenarios)}")

    # --- [ВОЗВРАЩЕННЫЙ БЛОК] Визуализация и вывод статистики ---

    # Визуализация проблемных сценариев
    if problem_scenarios and max_visualize > 0:
        logger.info(f"Визуализация до {min(len(problem_scenarios), max_visualize)} проблемных сценариев...")
        for i, scenario in enumerate(problem_scenarios):
            if i >= max_visualize: break

            path = Path(scenario['image_path'])
            image_to_plot = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image_to_plot)
            ax.set_title(f"Problem: {path.name}\nGTs: {scenario['num_gt']}, Positives: {scenario['num_positives']}")
            ax.axis('off')

            save_path = graphs_dir / f"problem_{i + 1}_{path.stem}.png"
            plt.savefig(save_path)
            plt.close(fig)
        logger.info(f"Визуализации сохранены в {graphs_dir}")

    # Вывод сводной статистики
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
                f"{level_name:<10} | {avg_pos:<15.2f} | {avg_ign:<15.2f} | {avg_neg:<15.2f} | {total_anchors_level:<15,}")
        print("-" * 80)
        print(
            f"{'ИТОГО':<10} | {total_pos_all_levels:<15.2f} | {total_ign_all_levels:<15.2f} | {total_neg_all_levels:<15.2f} | {current_idx:<15,}")
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

    # Вывод информации о лучших/худших случаях
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