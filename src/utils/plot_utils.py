import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path


# --- Установка глобального сида (остается без изменений) ---
def set_global_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


# ===> НОВЫЙ БЛОК: ФИКСИРОВАННЫЕ ЦВЕТА <===
FIXED_COLORS = {
    # Цвета для GT и Предсказаний
    "gt_pit": (255, 0, 0),  # GT Яма: Красный
    "gt_crack": (255, 128, 0),  # GT Трещина: Оранжевый
    "pred_pit": (0, 255, 0),  # Предсказанная Яма: Зеленый
    "pred_crack": (0, 128, 255),  # Предсказанная Трещина: Синий

    # Цвета для якорей (используется в plot_specific_anchors_on_image)
    "anchor_positive": (50, 205, 50),  # LimeGreen
    "anchor_ignored": (255, 215, 0),  # Gold
    "anchor_negative": (169, 169, 169),  # DarkGray

    "default": (220, 220, 220)  # Серый для всего остального
}


# ===> НОВАЯ ФУНКЦИЯ ВЫБОРА ЦВЕТА <===
def get_plot_color(box_type, label):
    """Возвращает фиксированный цвет на основе типа и класса рамки."""
    # box_type может быть 'gt', 'pred' или 'anchor'
    key = f"{box_type.lower()}_{label.lower()}"
    rgb_color = FIXED_COLORS.get(key, FIXED_COLORS["default"])
    return tuple(c / 255.0 for c in rgb_color)


# --- Основные функции отрисовки ---

def plot_image(image_np, ax, title=""):
    # (Функция остается без изменений)
    if image_np is None:
        ax.text(0.5, 0.5, "Изображение недоступно", ha='center', va='center', color='red')
        ax.set_title(title)
        ax.axis('off')
        return
    ax.imshow(image_np)
    if title:
        ax.set_title(title)
    ax.axis('off')


# ===> ИЗМЕНЕННАЯ ФУНКЦИЯ plot_boxes_on_image <===
def plot_boxes_on_image(ax, boxes, labels, box_type, scores=None, linewidth=2, fontsize=8):
    if boxes is None or len(boxes) == 0:
        return

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        xmin, ymin, xmax, ymax = map(int, box)
        width, height = xmax - xmin, ymax - ymin

        # Используем новую логику для выбора цвета
        box_color = get_plot_color(box_type, label)

        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=linewidth, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        text_parts = [label]
        if scores is not None and i < len(scores):
            text_parts.append(f"{scores[i]:.2f}")
        label_text = " ".join(text_parts)

        ax.text(xmin, ymin - 5, label_text, va='bottom', ha='left',
                color='white', fontsize=fontsize, weight='bold',
                bbox={'facecolor': box_color, 'alpha': 0.7, 'pad': 1})


# --- Вспомогательные функции отрисовки (адаптированные) ---

# ===> ИЗМЕНЕННАЯ ФУНКЦИЯ <===
def plot_original_gt(ax, image_np_original, gt_objects):
    """Отрисовывает оригинальное изображение и рамки Ground Truth."""
    plot_image(image_np_original, ax, title="Исходное изображение + GT")
    if gt_objects:
        boxes = [obj['bbox'] for obj in gt_objects]
        labels = [obj['class'] for obj in gt_objects]
        # Передаем новый аргумент box_type
        plot_boxes_on_image(ax, boxes, labels, box_type='gt', linewidth=2)


# ===> ИЗМЕНЕННАЯ ФУНКЦИЯ <===
def plot_augmented_gt(ax, image_np_augmented, augmented_gt_objects):
    """Отрисовывает аугментированное изображение и аугментированные рамки Ground Truth."""
    plot_image(image_np_augmented, ax, title="Аугментированное изображение + GT")
    if augmented_gt_objects:
        boxes = [obj['bbox'] for obj in augmented_gt_objects]
        labels = [obj['class'] for obj in augmented_gt_objects]
        # Передаем новый аргумент box_type
        plot_boxes_on_image(ax, boxes, labels, box_type='gt', linewidth=2)


# ===> ИЗМЕНЕННАЯ ФУНКЦИЯ <===
def plot_specific_anchors_on_image(ax, image_np, anchors_info_list, title="Якоря"):
    plot_image(image_np, ax, title=title)
    if not anchors_info_list: return

    for info in anchors_info_list:
        bbox = info.get('bbox')
        anchor_type = info.get('type', 'anchor')  # positive, ignored, negative
        iou_val = info.get('iou', None)

        if bbox is None: continue

        xmin, ymin, xmax, ymax = map(int, bbox)
        width, height = xmax - xmin, ymax - ymin

        # Используем цвета из палитры, предназначенные для якорей
        box_color = get_plot_color('anchor', anchor_type)

        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor=box_color, facecolor='none',
                                 linestyle='--')
        ax.add_patch(rect)

        text_parts = [anchor_type]
        if iou_val is not None: text_parts.append(f"{iou_val:.2f}")
        label_text = " ".join(text_parts)

        ax.text(xmin, ymin - 2, label_text, va='bottom', ha='left', color='black', fontsize=7,
                bbox={'facecolor': box_color, 'alpha': 0.6, 'pad': 0})


def show_plot():
    plt.show()


def save_plot(fig, filepath):
    try:
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        print(f"График сохранен в: {filepath}")
    except Exception as e:
        print(f"Ошибка сохранения графика в {filepath}: {e}")


# --- Тестовый Блок (адаптированный под новую логику) ---
if __name__ == '__main__':
    print("--- Тестирование plot_utils.py с новой логикой цветов ---")
    set_global_seed(42)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, 500)
    ax.set_ylim(500, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title("Тест отрисовки с фиксированными цветами")

    # --- GT данные ---
    gt_boxes = [[50, 50, 200, 150], [300, 50, 400, 250]]
    gt_labels = ["pit", "crack"]

    # --- Предсказания ---
    pred_boxes = [[55, 55, 205, 155], [290, 40, 410, 260]]
    pred_labels = ["pit", "crack"]
    pred_scores = [0.95, 0.88]

    # --- Якоря ---
    anchor_info = [
        {'bbox': [100, 300, 200, 400], 'type': 'positive', 'iou': 0.8},
        {'bbox': [250, 300, 350, 400], 'type': 'ignored', 'iou': 0.45}
    ]

    # --- Отрисовка ---
    print("Рисуем GT (pit-красный, crack-оранжевый)...")
    plot_boxes_on_image(ax, gt_boxes, gt_labels, box_type='gt', linewidth=3)

    print("Рисуем Предсказания (pit-зеленый, crack-синий)...")
    plot_boxes_on_image(ax, pred_boxes, pred_labels, box_type='pred', scores=pred_scores, linewidth=1.5)

    print("Рисуем якоря (positive-лаймовый, ignored-золотой)...")
    plot_specific_anchors_on_image(ax, None, anchor_info, title="")  # Рисуем только якоря на холсте

    plt.show()