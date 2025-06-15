# RoadDefectDetector/src/utils/plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf  # Понадобится для работы с тензорами из y_true

# Загрузка конфигурации для доступа к параметрам, если они нужны для декодирования
# (например, размеры сетки, якоря, классы)
import yaml
import os

_current_script_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils/
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))  # Корень проекта

_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')
_base_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'base_config.yaml')

DETECTOR_CONFIG = {}
BASE_CONFIG = {}

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"WARNING (plot_utils.py): Не удалось загрузить конфиги: {e}. Используются дефолты.")
    # Минимальные дефолты, чтобы модуль импортировался
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('classes', ['pit', 'crack'])
    DETECTOR_CONFIG.setdefault('num_anchors_per_location', 3)
    DETECTOR_CONFIG.setdefault('anchors_wh_normalized', [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]])
    BASE_CONFIG.setdefault('model_params', {'target_height': 416, 'target_width': 416})

TARGET_IMG_HEIGHT = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])[0]
TARGET_IMG_WIDTH = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])[1]
CLASSES_LIST_PLOT = DETECTOR_CONFIG.get('classes', ['class0', 'class1'])
NUM_ANCHORS_PLOT = DETECTOR_CONFIG.get('num_anchors_per_location', 3)
ANCHORS_WH_NORM_PLOT = np.array(DETECTOR_CONFIG.get('anchors_wh_normalized', [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]),
                                dtype=np.float32)
NETWORK_STRIDE_PLOT = 16  # Предполагаем, что это фиксировано для текущей архитектуры

GRID_HEIGHT_PLOT = TARGET_IMG_HEIGHT // NETWORK_STRIDE_PLOT
GRID_WIDTH_PLOT = TARGET_IMG_WIDTH // NETWORK_STRIDE_PLOT


def denormalize_box_xywh_to_xyxy(box_xywh_norm, img_width, img_height):
    """
    Преобразует нормализованные координаты (x_center, y_center, width, height)
    в пиксельные (xmin, ymin, xmax, ymax).
    """
    x_center, y_center, w, h = box_xywh_norm

    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height
    abs_w = w * img_width
    abs_h = h * img_height

    xmin = abs_x_center - abs_w / 2
    ymin = abs_y_center - abs_h / 2
    xmax = abs_x_center + abs_w / 2
    ymax = abs_y_center + abs_h / 2

    return int(xmin), int(ymin), int(xmax), int(ymax)


def decode_y_true_for_visualization(y_true_single_image_np,
                                    grid_h=GRID_HEIGHT_PLOT, grid_w=GRID_WIDTH_PLOT,
                                    anchors_wh_norm=ANCHORS_WH_NORM_PLOT,
                                    num_classes=len(CLASSES_LIST_PLOT),
                                    classes_list=CLASSES_LIST_PLOT):
    """
    Декодирует y_true для одного изображения обратно в читаемый формат (координаты рамок, классы).
    y_true_single_image_np: numpy array формы (grid_h, grid_w, num_anchors, 5 + num_classes)
    Возвращает список словарей, где каждый словарь:
        {'class_name': str, 'class_id': int, 'box_xywh_norm': [xc, yc, w, h], 'anchor_idx': int, 'grid_cell': [gy, gx]}
    """
    decoded_objects = []
    num_anchors = anchors_wh_norm.shape[0]

    # Ищем ячейки/якоря, где objectness > 0.5 (т.е., ответственные за объект)
    objectness_scores = y_true_single_image_np[..., 4]  # (grid_h, grid_w, num_anchors)
    responsible_indices = np.argwhere(objectness_scores > 0.5)  # Массив индексов [idx_gh, idx_gw, idx_anchor]

    for idx_gh, idx_gw, idx_anchor in responsible_indices:
        anchor_data = y_true_single_image_np[idx_gh, idx_gw, idx_anchor]

        # Декодируем координаты (tx, ty, tw, th) обратно в (xc_norm, yc_norm, w_norm, h_norm)
        tx, ty, tw, th = anchor_data[0:4]

        # Координаты центра ячейки (нормализованные относительно всего изображения)
        # (grid_x_center_norm, grid_y_center_norm)
        cell_x_norm = (idx_gw + 0.5) / grid_w  # +0.5 для центра ячейки
        cell_y_norm = (idx_gh + 0.5) / grid_h

        # Восстанавливаем box_center_x_norm и box_center_y_norm
        # tx = (box_center_x_norm * grid_w) - grid_x_idx  => box_center_x_norm = (tx + grid_x_idx) / grid_w
        # ty = (box_center_y_norm * grid_h) - grid_y_idx  => box_center_y_norm = (ty + grid_y_idx) / grid_h
        box_center_x_norm = (tx + idx_gw) / grid_w
        box_center_y_norm = (ty + idx_gh) / grid_h

        # Восстанавливаем box_width_norm и box_height_norm
        # tw = log(box_width_norm / anchor_width_norm) => box_width_norm = exp(tw) * anchor_width_norm
        # th = log(box_height_norm / anchor_height_norm) => box_height_norm = exp(th) * anchor_height_norm
        anchor_w_norm = anchors_wh_norm[idx_anchor, 0]
        anchor_h_norm = anchors_wh_norm[idx_anchor, 1]

        box_width_norm = np.exp(tw) * anchor_w_norm
        box_height_norm = np.exp(th) * anchor_h_norm

        box_xywh_norm = [box_center_x_norm, box_center_y_norm, box_width_norm, box_height_norm]

        # Класс
        class_one_hot = anchor_data[5:]
        class_id = np.argmax(class_one_hot)
        class_name = classes_list[class_id] if class_id < len(classes_list) else "Unknown"

        decoded_objects.append({
            "class_name": class_name,
            "class_id": class_id,
            "box_xywh_norm": box_xywh_norm,  # [xc_norm, yc_norm, w_norm, h_norm]
            "anchor_idx": idx_anchor,
            "grid_cell": [idx_gh, idx_gw]
        })

    return decoded_objects


def visualize_data_sample(image_np, y_true_np, title="Sample with Ground Truth"):
    """
    Визуализирует одно изображение с его ground truth y_true (для детектора).
    image_np: numpy array, изображение (предполагается в формате [0,1] RGB).
    y_true_np: numpy array, y_true для этого изображения (grid_h, grid_w, num_anchors, 5 + num_classes).
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title(title)

    img_height, img_width = image_np.shape[:2]

    # Декодируем y_true, чтобы получить информацию об объектах
    decoded_gt_objects = decode_y_true_for_visualization(y_true_np)

    if not decoded_gt_objects:
        print(f"INFO (visualize_data_sample): На изображении '{title}' не найдено ответственных якорей в y_true.")
        # Можно нарисовать сетку, если объектов нет, для наглядности
        for i in range(GRID_WIDTH_PLOT + 1):
            ax.axvline(i * NETWORK_STRIDE_PLOT * (img_width / TARGET_IMG_WIDTH), color='gray', linestyle=':',
                       linewidth=0.5)
        for i in range(GRID_HEIGHT_PLOT + 1):
            ax.axhline(i * NETWORK_STRIDE_PLOT * (img_height / TARGET_IMG_HEIGHT), color='gray', linestyle=':',
                       linewidth=0.5)
        plt.show()
        return

    for obj_info in decoded_gt_objects:
        class_name = obj_info['class_name']
        box_xywh_norm = obj_info['box_xywh_norm']
        grid_cell_y, grid_cell_x = obj_info['grid_cell']
        anchor_idx_responsible = obj_info['anchor_idx']

        # Преобразуем нормализованные [xc, yc, w, h] в пиксельные [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = denormalize_box_xywh_to_xyxy(box_xywh_norm, img_width, img_height)

        rect_width = xmax - xmin
        rect_height = ymax - ymin

        # Рисуем bounding box объекта
        rect_gt = patches.Rectangle((xmin, ymin), rect_width, rect_height, linewidth=2, edgecolor='g', facecolor='none',
                                    label=f"GT: {class_name}")
        ax.add_patch(rect_gt)
        ax.text(xmin, ymin - 5,
                f"GT: {class_name} (Anchor {anchor_idx_responsible} in cell [{grid_cell_y},{grid_cell_x}])",
                color='green', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=0))

        # (Опционально) Можно нарисовать сам ответственный якорь и центр ячейки
        # Это потребует передачи ANCHORS_WH_NORM_PLOT и размеров сетки
        # cell_pixel_width = img_width / GRID_WIDTH_PLOT
        # cell_pixel_height = img_height / GRID_HEIGHT_PLOT
        #
        # resp_anchor_w_norm, resp_anchor_h_norm = ANCHORS_WH_NORM_PLOT[anchor_idx_responsible]
        # resp_anchor_w_px = resp_anchor_w_norm * img_width
        # resp_anchor_h_px = resp_anchor_h_norm * img_height
        #
        # cell_center_x_px = (grid_cell_x + 0.5) * cell_pixel_width
        # cell_center_y_px = (grid_cell_y + 0.5) * cell_pixel_height
        #
        # anchor_xmin = cell_center_x_px - resp_anchor_w_px / 2
        # anchor_ymin = cell_center_y_px - resp_anchor_h_px / 2
        #
        # rect_anchor = patches.Rectangle((anchor_xmin, anchor_ymin), resp_anchor_w_px, resp_anchor_h_px, linewidth=1, edgecolor='yellow', linestyle='--', facecolor='none', label=f"Anchor {anchor_idx_responsible}")
        # ax.add_patch(rect_anchor)
        # ax.plot(cell_center_x_px, cell_center_y_px, 'yo', markersize=5) # Центр ячейки

    # Рисуем сетку для наглядности
    # Масштабируем шаг сетки к текущему размеру отображаемого изображения
    # Если image_np уже имеет TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, то множители не нужны
    # Но если мы визуализируем оригинальное изображение, а y_true относится к TARGET_*, то нужен пересчет.
    # В нашем data_loader'е image_np УЖЕ будет иметь TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH.
    x_step = img_width / GRID_WIDTH_PLOT
    y_step = img_height / GRID_HEIGHT_PLOT

    for i in range(GRID_WIDTH_PLOT + 1):
        ax.axvline(i * x_step, color='gray', linestyle=':', linewidth=0.5)
    for i in range(GRID_HEIGHT_PLOT + 1):
        ax.axhline(i * y_step, color='gray', linestyle=':', linewidth=0.5)

    plt.legend()  # Чтобы отобразить label у rect_gt
    plt.show()


if __name__ == '__main__':
    print("--- Тестирование plot_utils.py ---")

    # Создаем фиктивные данные для теста
    test_img_height = TARGET_IMG_HEIGHT
    test_img_width = TARGET_IMG_WIDTH
    test_num_classes = len(CLASSES_LIST_PLOT)
    test_num_anchors = NUM_ANCHORS_PLOT

    # 1. Изображение с одним объектом
    print("\nТест 1: Изображение с одним объектом")
    img_with_object = np.ones((test_img_height, test_img_width, 3), dtype=np.float32) * 0.8  # Светло-серый фон

    y_true_obj = np.zeros((GRID_HEIGHT_PLOT, GRID_WIDTH_PLOT, test_num_anchors, 5 + test_num_classes), dtype=np.float32)

    # Параметры объекта
    gt_xc_norm = 0.5
    gt_yc_norm = 0.5
    gt_w_norm = 0.2
    gt_h_norm = 0.3
    gt_class_id = 0  # Первый класс ('pit')

    # Находим ячейку сетки и "лучший" якорь (здесь просто первый якорь для простоты)
    obj_grid_x = int(gt_xc_norm * GRID_WIDTH_PLOT)
    obj_grid_y = int(gt_yc_norm * GRID_HEIGHT_PLOT)
    best_anchor_idx_test = 0  # Просто берем первый якорь

    # Кодируем координаты для y_true
    tx = (gt_xc_norm * GRID_WIDTH_PLOT) - obj_grid_x
    ty = (gt_yc_norm * GRID_HEIGHT_PLOT) - obj_grid_y
    anchor_w_test, anchor_h_test = ANCHORS_WH_NORM_PLOT[best_anchor_idx_test]
    tw = np.log(gt_w_norm / (anchor_w_test + 1e-9) + 1e-9)  # Добавляем epsilon для стабильности логарифма
    th = np.log(gt_h_norm / (anchor_h_test + 1e-9) + 1e-9)

    y_true_obj[obj_grid_y, obj_grid_x, best_anchor_idx_test, 0:4] = [tx, ty, tw, th]
    y_true_obj[obj_grid_y, obj_grid_x, best_anchor_idx_test, 4] = 1.0  # Objectness
    y_true_obj[obj_grid_y, obj_grid_x, best_anchor_idx_test, 5 + gt_class_id] = 1.0  # Class (one-hot)

    visualize_data_sample(img_with_object, y_true_obj,
                          title=f"Test 1: One Object (Class: {CLASSES_LIST_PLOT[gt_class_id]})")

    # 2. Пустое изображение (без объектов)
    print("\nТест 2: Пустое изображение")
    img_empty = np.ones((test_img_height, test_img_width, 3), dtype=np.float32) * 0.5  # Серый фон
    y_true_empty = np.zeros((GRID_HEIGHT_PLOT, GRID_WIDTH_PLOT, test_num_anchors, 5 + test_num_classes),
                            dtype=np.float32)
    visualize_data_sample(img_empty, y_true_empty, title="Test 2: Empty Image (No Objects)")

    print("\n--- Тестирование plot_utils.py завершено ---")