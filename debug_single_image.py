import sys

import cv2
import yaml
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# --- Настройка путей и импортов ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import (
        generate_all_anchors, parse_voc_xml, assign_gt_to_anchors, encode_box_targets
    )
    from src.utils import plot_utils
    from src.datasets import augmentations
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}\nУбедитесь, что скрипт запускается из корня проекта.")
    sys.exit(1)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Основная функция отладки ---

def debug_one_image(config, image_name, use_augmentation=True, aug_seed=None, top_k_positive=5, top_k_ignored=5):
    logger.info(f"--- Запуск детального анализа для: {image_name} ---")

    # --- Шаг 1: Загрузка исходных данных ---
    dataset_path = Path(config['dataset_path'])
    image_path = dataset_path / config['train_images_subdir'] / image_name
    annot_path = dataset_path / config['train_annotations_subdir'] / (Path(image_name).stem + ".xml")
    # ... (проверка существования файлов) ...

    image_original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = image_original.shape
    gt_boxes_original, gt_class_names_original = parse_voc_xml(annot_path)

    # --- Шаг 2: Ресайз и Аугментация ---
    h_target, w_target = config['input_shape'][:2]
    image_resized = cv2.resize(image_original, (w_target, h_target))

    gt_boxes_resized = []
    if gt_boxes_original:
        for box in gt_boxes_original:
            x1, y1, x2, y2 = box
            gt_boxes_resized.append([(x1 / w_orig) * w_target, (y1 / h_orig) * h_target, (x2 / w_orig) * w_target,
                                     (y2 / h_orig) * h_target])

    if use_augmentation:
        logger.info(f"Аугментация: ВКЛЮЧЕНА (сид: {aug_seed if aug_seed is not None else 'случайный'})")
        if aug_seed is not None: np.random.seed(aug_seed)
        augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
        augmented = augmenter(image=image_resized, bboxes=gt_boxes_resized,
                              class_labels_for_albumentations=gt_class_names_original)
        image_augmented = augmented['image']
        gt_boxes_augmented = augmented['bboxes']
    else:
        logger.info("Аугментация: ВЫКЛЮЧЕНА")
        image_augmented = image_resized
        gt_boxes_augmented = gt_boxes_resized

    # --- Шаг 3: Генерация и назначение якорей ---
    all_anchors_norm = generate_all_anchors(config['input_shape'], [8, 16, 32], config['anchor_scales'],
                                            config['anchor_ratios'])

    gt_boxes_aug_norm = np.array(gt_boxes_augmented, dtype=np.float32) / np.array(
        [w_target, h_target, w_target, h_target])

    anchor_labels, _, max_iou_per_anchor = assign_gt_to_anchors(
        gt_boxes_aug_norm, all_anchors_norm,
        config['anchor_positive_iou_threshold'],
        config['anchor_ignore_iou_threshold']
    )

    # --- Шаг 4: Сбор информации для визуализации ---
    all_anchors_pixels = all_anchors_norm * np.array([w_target, h_target, w_target, h_target])
    positive_indices = np.where(anchor_labels == 1)[0]
    ignored_indices = np.where(anchor_labels == 0)[0]

    sorted_pos_indices = positive_indices[np.argsort(-max_iou_per_anchor[positive_indices])]
    sorted_ign_indices = ignored_indices[np.argsort(-max_iou_per_anchor[ignored_indices])]

    top_pos_info = [{'bbox': all_anchors_pixels[i], 'type': 'positive', 'iou': max_iou_per_anchor[i]} for i in
                    sorted_pos_indices[:top_k_positive]]
    top_ign_info = [{'bbox': all_anchors_pixels[i], 'type': 'ignored', 'iou': max_iou_per_anchor[i]} for i in
                    sorted_ign_indices[:top_k_ignored]]
    anchors_to_plot = top_pos_info + top_ign_info

    # --- Шаг 5: Визуализация ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(f"Детальный анализ: {image_name} | Сид аугментации: {aug_seed}", fontsize=16)

    # График 1
    plot_utils.plot_original_gt(axes[0], image_original,
                                [{'bbox': b, 'class': l} for b, l in zip(gt_boxes_original, gt_class_names_original)])

    # График 2
    axes[1].set_title(f"2. Аугментация + Топ-{top_k_positive} Pos / Топ-{top_k_ignored} Ign якорей")
    plot_utils.plot_image(image_augmented, axes[1])
    plot_utils.plot_boxes_on_image(axes[1], gt_boxes_augmented, labels=gt_class_names_original, color_index_base=10,
                                   linewidth=3)
    plot_utils.plot_specific_anchors_on_image(axes[1], image_augmented, anchors_to_plot, title="")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_utils.show_plot()

    # --- Шаг 6: Вывод в консоль ---
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ЛУЧШИХ ЯКОРЯХ".center(80))
    print("=" * 80)
    print(f"\n--- Топ-{len(top_pos_info)} Позитивных якорей (отсортировано по IoU) ---")
    if not top_pos_info: print("Позитивные якоря не найдены.")
    for info in top_pos_info:
        bbox = [f"{c:.1f}" for c in info['bbox']]
        print(f"  - IoU: {info['iou']:.4f} | Координаты (пикс): [{', '.join(bbox)}]")
    print(f"\n--- Топ-{len(top_ign_info)} Игнорируемых якорей (отсортировано по IoU) ---")
    if not top_ign_info: print("Игнорируемые якоря не найдены.")
    for info in top_ign_info:
        bbox = [f"{c:.1f}" for c in info['bbox']]
        print(f"  - IoU: {info['iou']:.4f} | Координаты (пикс): [{', '.join(bbox)}]")
    print("=" * 80 + "\n")


# --- Запуск отладки ---
if __name__ == '__main__':
    try:
        config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Конфигурационный файл не найден по пути: {config_path}")
        sys.exit(1)

    IMAGE_TO_DEBUG = "China_Drone_000091.jpg"
    USE_AUGMENTATION = True
    AUGMENTATION_SEED = 42
    TOP_K_POSITIVE = 10
    TOP_K_IGNORED = 10

    debug_one_image(
        config=main_config,
        image_name=IMAGE_TO_DEBUG,
        use_augmentation=USE_AUGMENTATION,
        aug_seed=AUGMENTATION_SEED,
        top_k_positive=TOP_K_POSITIVE,
        top_k_ignored=TOP_K_IGNORED
    )