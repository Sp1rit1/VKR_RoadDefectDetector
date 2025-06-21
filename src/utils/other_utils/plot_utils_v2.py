# src/utils/plot_utils_v2.py
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import os
from pathlib import Path

# --- Загрузка Конфигурации (нужна для параметров сетки, якорей, классов) ---
_current_script_dir_plot_v2 = Path(__file__).resolve().parent
_project_root_plot_v2 = _current_script_dir_plot_v2.parent.parent
_detector_config_path_plot_v2 = _project_root_plot_v2 / 'src' / 'configs' / 'detector_config_single_level_v2.yaml'  # Используем v2 конфиг

DETECTOR_CONFIG_PLOT_V2 = {}
try:
    with open(_detector_config_path_plot_v2, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_PLOT_V2 = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_PLOT_V2, dict): DETECTOR_CONFIG_PLOT_V2 = {}
except Exception as e_cfg_plot_v2:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (plot_utils_v2.py): Не удалось загрузить detector_config_single_level_v2.yaml: {e_cfg_plot_v2}. Используются дефолты.")
    # Минимальные дефолты, чтобы модуль импортировался и работал автономно
    _fpn_params_default_plot_v2 = {
        'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
        'detector_fpn_levels': ['P4_debug'],
        'detector_fpn_strides': {'P4_debug': 16},
        'detector_fpn_anchor_configs': {'P4_debug': {'num_anchors_this_level': 7,
                                                     'anchors_wh_normalized': [[0.1832, 0.1104], [0.0967, 0.2753],
                                                                               [0.4083, 0.0925], [0.0921, 0.4968],
                                                                               [0.2919, 0.1936], [0.7358, 0.0843],
                                                                               [0.0743, 0.8969]]}}
    }
    DETECTOR_CONFIG_PLOT_V2.setdefault('fpn_detector_params', _fpn_params_default_plot_v2)

# --- Глобальные Параметры из Конфига v2 ---
_fpn_params_plot_v2 = DETECTOR_CONFIG_PLOT_V2.get('fpn_detector_params', {})
INPUT_SHAPE_PLOT_V2 = tuple(_fpn_params_plot_v2.get('input_shape', [416, 416, 3]))
TARGET_IMG_HEIGHT_PLOT_V2, TARGET_IMG_WIDTH_PLOT_V2 = INPUT_SHAPE_PLOT_V2[0], INPUT_SHAPE_PLOT_V2[1]
CLASSES_LIST_PLOT_V2 = _fpn_params_plot_v2.get('classes', ['pit', 'crack'])
NUM_CLASSES_PLOT_V2 = len(CLASSES_LIST_PLOT_V2)

# Конфигурация для нашего ОДНОГО уровня (P4_debug)
LEVEL_NAME_PLOT_V2 = _fpn_params_plot_v2.get('detector_fpn_levels', ['P4_debug'])[0]
_level_stride_plot_v2 = _fpn_params_plot_v2.get('detector_fpn_strides', {}).get(LEVEL_NAME_PLOT_V2, 16)
_level_anchor_cfg_yaml_plot_v2 = _fpn_params_plot_v2.get('detector_fpn_anchor_configs', {}).get(LEVEL_NAME_PLOT_V2, {})

ANCHORS_WH_PLOT_V2 = np.array(_level_anchor_cfg_yaml_plot_v2.get('anchors_wh_normalized', [[0.15, 0.15] * 7]),
                              dtype=np.float32)
NUM_ANCHORS_PLOT_V2 = _level_anchor_cfg_yaml_plot_v2.get('num_anchors_this_level', ANCHORS_WH_PLOT_V2.shape[0])
if ANCHORS_WH_PLOT_V2.ndim == 1 and ANCHORS_WH_PLOT_V2.shape[0] == 2: ANCHORS_WH_PLOT_V2 = np.expand_dims(
    ANCHORS_WH_PLOT_V2, axis=0)
if ANCHORS_WH_PLOT_V2.shape[0] != NUM_ANCHORS_PLOT_V2: NUM_ANCHORS_PLOT_V2 = ANCHORS_WH_PLOT_V2.shape[0]

GRID_H_PLOT_V2 = TARGET_IMG_HEIGHT_PLOT_V2 // _level_stride_plot_v2
GRID_W_PLOT_V2 = TARGET_IMG_WIDTH_PLOT_V2 // _level_stride_plot_v2

SINGLE_LEVEL_CONFIG_PLOT_V2 = {  # Для передачи в decode_single_level_y_true_for_viz
    'grid_h': GRID_H_PLOT_V2, 'grid_w': GRID_W_PLOT_V2,
    'anchors_wh_normalized': ANCHORS_WH_PLOT_V2,
    'num_anchors': NUM_ANCHORS_PLOT_V2,  # Это должно быть num_anchors_this_level
    'stride': _level_stride_plot_v2
}


# ----------------------------------------------------------------------------------

def denormalize_box_xywh_to_xyxy(box_xywh_norm, img_width_display, img_height_display):
    # ... (код этой функции остается таким же) ...
    if len(box_xywh_norm) != 4: return 0, 0, 0, 0
    x_center, y_center, w, h = box_xywh_norm
    abs_x_center = x_center * img_width_display;
    abs_y_center = y_center * img_height_display
    abs_w = w * img_width_display;
    abs_h = h * img_height_display
    xmin = abs_x_center - abs_w / 2;
    ymin = abs_y_center - abs_h / 2
    xmax = abs_x_center + abs_w / 2;
    ymax = abs_y_center + abs_h / 2
    return int(xmin), int(ymin), int(xmax), int(ymax)


def decode_single_level_y_true_for_viz(y_true_level_np,
                                       level_config_plot_arg,
                                       # Словарь с 'grid_h', 'grid_w', 'anchors_wh_normalized', 'num_anchors'
                                       classes_list_plot_arg):
    # ... (код этой функции остается таким же, как в plot_utils1.txt, но убедись, что она использует level_config_plot_arg)
    decoded_objects_on_level = []
    # Проверка, что level_config_plot_arg содержит все необходимые ключи
    expected_keys = ['grid_h', 'grid_w', 'anchors_wh_normalized', 'num_anchors']
    if not all(k in level_config_plot_arg for k in expected_keys):
        print(f"ОШИБКА (plot_utils_v2 - decode_single_level): Неполная конфигурация для уровня. "
              f"Отсутствуют ключи: {[k for k in expected_keys if k not in level_config_plot_arg]}")
        return decoded_objects_on_level

    grid_h, grid_w = level_config_plot_arg['grid_h'], level_config_plot_arg['grid_w']
    anchors_wh_norm_this_level = level_config_plot_arg['anchors_wh_normalized']
    num_anchors_from_y_true_shape = y_true_level_np.shape[2]

    if num_anchors_from_y_true_shape != level_config_plot_arg['num_anchors']:
        print(f"ПРЕДУПРЕЖДЕНИЕ (plot_utils_v2 - decode_single_level): Несоответствие якорей! "
              f"y_true имеет {num_anchors_from_y_true_shape}, конфиг ожидает {level_config_plot_arg['num_anchors']}.")
        # Можно либо упасть, либо попытаться использовать num_anchors_from_y_true_shape,
        # но это может привести к ошибке индексации в anchors_wh_norm_this_level.
        # Безопаснее вернуть пустой список, если есть несоответствие.
        return decoded_objects_on_level

    objectness_scores = y_true_level_np[..., 4]  # 0=фон, 1=объект, -1=игнор
    responsible_indices = np.argwhere(objectness_scores > 0.5)  # Только позитивные для отрисовки назначений

    for idx_gh, idx_gw, idx_anchor in responsible_indices:
        anchor_data = y_true_level_np[idx_gh, idx_gw, idx_anchor]
        tx, ty, tw, th = anchor_data[0:4]
        box_center_x_norm = (tx + float(idx_gw)) / float(grid_w)  # Убедимся, что все float
        box_center_y_norm = (ty + float(idx_gh)) / float(grid_h)
        anchor_w_norm = anchors_wh_norm_this_level[idx_anchor, 0]
        anchor_h_norm = anchors_wh_norm_this_level[idx_anchor, 1]
        box_width_norm = np.exp(tw) * anchor_w_norm
        box_height_norm = np.exp(th) * anchor_h_norm
        box_xywh_norm = [box_center_x_norm, box_center_y_norm, box_width_norm, box_height_norm]
        class_one_hot = anchor_data[5:]
        class_id = np.argmax(class_one_hot)
        class_name = classes_list_plot_arg[class_id] if 0 <= class_id < len(classes_list_plot_arg) else "Unknown"
        decoded_objects_on_level.append({
            "class_name": class_name, "class_id": class_id, "box_xywh_norm": box_xywh_norm,
            "anchor_idx": idx_anchor, "grid_cell": [idx_gh, idx_gw]
        })
    return decoded_objects_on_level


def visualize_single_level_gt_assignments(
        image_np_processed,  # Изображение [0,1] RGB (например, 416x416)
        y_true_single_level_np,  # y_true тензор для этого одного уровня
        level_config_for_drawing,  # SINGLE_LEVEL_CONFIG_PLOT_V2 (содержит grid_h/w, stride)
        classes_list_for_drawing,  # CLASSES_LIST_PLOT_V2
        original_gt_boxes_for_ref=None,  # Кортеж (scaled_boxes_xyxy_norm_np, class_ids_np)
        title_prefix=""):
    """
    Визуализирует назначения Ground Truth для ОДНОУРОВНЕВОЙ модели.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np_processed)
    ax.set_title(f"{title_prefix} - Single Level GT Assignments", fontsize=10)

    img_h_display, img_w_display = image_np_processed.shape[:2]
    legend_handles_dict = {}

    # 1. Отрисовка сетки
    grid_h_draw, grid_w_draw = level_config_for_drawing['grid_h'], level_config_for_drawing['grid_w']
    if grid_w_draw > 0 and grid_h_draw > 0:  # Проверка, чтобы не делить на ноль
        x_step_draw = img_w_display / grid_w_draw
        y_step_draw = img_h_display / grid_h_draw
        for x_line_idx_draw in range(grid_w_draw + 1):
            ax.axvline(x_line_idx_draw * x_step_draw, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        for y_line_idx_draw in range(grid_h_draw + 1):
            ax.axhline(y_line_idx_draw * y_step_draw, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    # 2. Отрисовка назначенных GT из y_true (ответственные якоря)
    # Передаем имя уровня фиктивно, так как decode_single_level_y_true_for_viz его ожидает, но не использует
    decoded_gt_objects = decode_single_level_y_true_for_viz(
        y_true_single_level_np,
        level_config_for_drawing,  # Передаем весь словарь конфигурации уровня
        classes_list_for_drawing
    )


    if decoded_gt_objects:
        print(f"  Visualizing {len(decoded_gt_objects)} responsible anchors from y_true for {title_prefix}:")
        for obj_info in decoded_gt_objects:
            class_name_viz = obj_info['class_name']
            box_xywh_norm_viz = obj_info['box_xywh_norm']
            xmin_viz, ymin_viz, xmax_viz, ymax_viz = denormalize_box_xywh_to_xyxy(
                box_xywh_norm_viz, img_w_display, img_h_display
            )
            rect_w_viz, rect_h_viz = xmax_viz - xmin_viz, ymax_viz - ymin_viz

            color_y_true_viz = 'lime'  # Зеленый для назначенных GT
            rect_assigned_gt = patches.Rectangle((xmin_viz, ymin_viz), rect_w_viz, rect_h_viz,
                                                 linewidth=2, edgecolor=color_y_true_viz, facecolor='none',
                                                 linestyle='-')
            ax.add_patch(rect_assigned_gt)
            ax.text(xmin_viz, ymin_viz - 5,
                    f"AssignedGT: {class_name_viz} (A{obj_info['anchor_idx']}@Cell[{obj_info['grid_cell'][0]},{obj_info['grid_cell'][1]}])",
                    color=color_y_true_viz, fontsize=7, bbox=dict(facecolor='white', alpha=0.6, pad=0))
            legend_handles_dict["Assigned GT (from y_true)"] = patches.Patch(color=color_y_true_viz,
                                                                             label="Assigned GT (from y_true)")
    else:
        print(f"  No responsible anchors found in y_true for {title_prefix}.")

    # 3. Отрисовка оригинальных GT рамок для справки (если переданы)
    if original_gt_boxes_for_ref:
        scaled_boxes_np_ref_xyxy, class_ids_np_ref = original_gt_boxes_for_ref
        if scaled_boxes_np_ref_xyxy.ndim == 2 and scaled_boxes_np_ref_xyxy.shape[1] == 4 and class_ids_np_ref.ndim == 1:
            print(f"  Visualizing {scaled_boxes_np_ref_xyxy.shape[0]} original GT boxes for reference:")
            for i_ref in range(scaled_boxes_np_ref_xyxy.shape[0]):
                # scaled_boxes_np_ref_xyxy УЖЕ нормализованы [xmin, ymin, xmax, ymax] относительно TARGET_IMG_SIZE
                xmin_n_ref, ymin_n_ref, xmax_n_ref, ymax_n_ref = scaled_boxes_np_ref_xyxy[i_ref]
                class_id_ref = int(class_ids_np_ref[i_ref])
                class_name_ref = classes_list_for_drawing[class_id_ref] if 0 <= class_id_ref < len(
                    classes_list_for_drawing) else "Unknown"

                xmin_px_ref = int(xmin_n_ref * img_w_display)
                ymin_px_ref = int(ymin_n_ref * img_h_display)
                xmax_px_ref = int(xmax_n_ref * img_w_display)
                ymax_px_ref = int(ymax_n_ref * img_h_display)
                w_ref, h_ref = xmax_px_ref - xmin_px_ref, ymax_px_ref - ymin_px_ref

                if w_ref > 0 and h_ref > 0:  # Рисуем только валидные рамки
                    rect_orig_gt_viz = patches.Rectangle((xmin_px_ref, ymin_px_ref), w_ref, h_ref,
                                                         linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
                    ax.add_patch(rect_orig_gt_viz)
                    ax.text(xmin_px_ref + 2, ymin_px_ref + h_ref - 2, f"OrigGT:{class_name_ref}", color='red',
                            fontsize=6,
                            verticalalignment='top')
                    legend_handles_dict["Original GT (reference)"] = patches.Patch(color='red', linestyle='--',
                                                                                   label="Original GT (reference)")

    if legend_handles_dict:
        ax.legend(handles=list(legend_handles_dict.values()), loc='best', fontsize='x-small')

    plt.tight_layout(pad=0.5)
    plt.show()


def draw_detections_on_image_single_level(
        image_np_bgr,  # Изображение в формате BGR (uint8) для OpenCV
        boxes_norm_yminxminymaxxmax,  # Нормализованные [ymin, xmin, ymax, xmax]
        scores,
        class_ids,
        class_names_list,  # Список имен классов (например, ['pit', 'crack'])
        img_disp_width,  # Ширина изображения, на котором рисуем
        img_disp_height  # Высота
):
    """
    Рисует предсказанные рамки, классы и уверенность на изображении.
    Возвращает изображение с нарисованными детекциями (BGR).
    """
    output_image = image_np_bgr.copy()
    num_detections = boxes_norm_yminxminymaxxmax.shape[0]

    for i in range(num_detections):
        if scores[i] < 0.001:  # Очень низкий порог, чтобы почти все показать (если NMS уже применен)
            continue

        ymin_n, xmin_n, ymax_n, xmax_n = boxes_norm_yminxminymaxxmax[i]

        # Преобразование в пиксельные координаты
        xmin = int(xmin_n * img_disp_width)
        ymin = int(ymin_n * img_disp_height)
        xmax = int(xmax_n * img_disp_width)
        ymax = int(ymax_n * img_disp_height)

        # Клиппинг координат по границам изображения
        xmin = max(0, min(xmin, img_disp_width - 1))
        ymin = max(0, min(ymin, img_disp_height - 1))
        xmax = max(0, min(xmax, img_disp_width - 1))
        ymax = max(0, min(ymax, img_disp_height - 1))

        if xmin >= xmax or ymin >= ymax:  # Пропускаем невалидные рамки
            continue

        class_id = int(class_ids[i])
        score_val = scores[i]

        label_text = f"Unknown: {score_val:.2f}"
        color = (128, 128, 128)  # Серый по умолчанию

        if 0 <= class_id < len(class_names_list):
            label_text = f"{class_names_list[class_id]}: {score_val:.2f}"
            if class_names_list[class_id] == 'pit':  # Используй точные имена классов
                color = (0, 0, 255)  # Красный для ям
            elif class_names_list[class_id] == 'crack':  # или 'treshina'
                color = (0, 255, 0)  # Зеленый для трещин
            # Добавь другие классы, если будут

        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(output_image, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image



if __name__ == '__main__':
    print("--- Тестирование plot_utils_v2.py (для ОДНОУРОВНЕВОЙ модели) ---")
    # Используем глобальные параметры, определенные в начале этого файла из конфига
    test_img_height_main = TARGET_IMG_HEIGHT_PLOT_V2
    test_img_width_main = TARGET_IMG_WIDTH_PLOT_V2
    test_classes_list_main = CLASSES_LIST_PLOT_V2
    test_num_classes_main = NUM_CLASSES_PLOT_V2
    test_level_config_main = SINGLE_LEVEL_CONFIG_PLOT_V2  # Это конфиг для нашего одного уровня

    # 1. Изображение (просто серое)
    img_test_single_level = np.ones((test_img_height_main, test_img_width_main, 3), dtype=np.float32) * 0.7

    # 2. y_true для одного уровня
    y_true_single_level_test = np.zeros((test_level_config_main['grid_h'],
                                         test_level_config_main['grid_w'],
                                         test_level_config_main['num_anchors'],
                                         5 + test_num_classes_main), dtype=np.float32)

    # Добавим один "ответственный" якорь для примера
    gt_xc_n_main, gt_yc_n_main, gt_w_n_main, gt_h_n_main = 0.5, 0.5, 0.2, 0.15
    gt_class_id_main = 0  # pit

    obj_grid_x_main = int(gt_xc_n_main * test_level_config_main['grid_w'])
    obj_grid_y_main = int(gt_yc_n_main * test_level_config_main['grid_h'])
    best_anchor_idx_main = min(0, test_level_config_main['num_anchors'] - 1)  # Просто берем первый якорь
    if best_anchor_idx_main < 0: best_anchor_idx_main = 0  # На случай, если num_anchors = 0

    if test_level_config_main['num_anchors'] > 0:  # Только если есть якоря
        tx_main = (gt_xc_n_main * test_level_config_main['grid_w']) - obj_grid_x_main
        ty_main = (gt_yc_n_main * test_level_config_main['grid_h']) - obj_grid_y_main
        anchor_w_main, anchor_h_main = test_level_config_main['anchors_wh_normalized'][best_anchor_idx_main]
        tw_main = np.log(max(gt_w_n_main, 1e-9) / (anchor_w_main + 1e-9))
        th_main = np.log(max(gt_h_n_main, 1e-9) / (anchor_h_main + 1e-9))

        y_true_single_level_test[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 0:4] = [tx_main, ty_main,
                                                                                                 tw_main, th_main]
        y_true_single_level_test[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 4] = 1.0  # Objectness
        y_true_single_level_test[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 5 + gt_class_id_main] = 1.0

    # 3. "Оригинальные" GT рамки для справки (нормализованные xmin, ymin, xmax, ymax)
    original_gt_boxes_norm_xyxy_ref = np.array([
        [gt_xc_n_main - gt_w_n_main / 2, gt_yc_n_main - gt_h_n_main / 2,
         gt_xc_n_main + gt_w_n_main / 2, gt_yc_n_main + gt_h_n_main / 2]
    ], dtype=np.float32)
    original_gt_class_ids_ref = np.array([gt_class_id_main], dtype=np.int32)

    print("\nВызов visualize_single_level_gt_assignments для тестовых данных...")
    visualize_single_level_gt_assignments(
        img_test_single_level,
        y_true_single_level_test,
        level_config_for_drawing=test_level_config_main,
        classes_list_for_drawing=test_classes_list_main,
        original_gt_boxes_for_ref=(original_gt_boxes_norm_xyxy_ref, original_gt_class_ids_ref),
        title_prefix="Single Level GT Test"
    )
    print("\n--- Тестирование plot_utils_v2.py завершено ---")