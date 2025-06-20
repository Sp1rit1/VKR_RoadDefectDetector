# src/utils/plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import os
from pathlib import Path  # Для более удобной работы с путями

# --- Константы и Дефолты, используемые ТОЛЬКО в __main__ или если конфиг не загружен ---
_PLOT_UTILS_DEFAULT_TARGET_IMG_HEIGHT = 416
_PLOT_UTILS_DEFAULT_TARGET_IMG_WIDTH = 416
_PLOT_UTILS_DEFAULT_CLASSES_LIST = ['class0_fallback', 'class1_fallback']
_PLOT_UTILS_DEFAULT_FPN_LEVEL_NAMES = ['P3', 'P4', 'P5']
# Эта структура _PLOT_UTILS_DEFAULT_FPN_CONFIGS_DICT используется для дефолтов в __main__,
# если detector_config.yaml не содержит полной информации.
_PLOT_UTILS_DEFAULT_FPN_CONFIGS_DICT = {
    level_name: {
        'grid_h': _PLOT_UTILS_DEFAULT_TARGET_IMG_HEIGHT // (8 * (2 ** idx)),
        'grid_w': _PLOT_UTILS_DEFAULT_TARGET_IMG_WIDTH // (8 * (2 ** idx)),
        'num_anchors': 3,  # Дефолтное количество якорей
        'anchors_wh_normalized': np.array([[0.05 * (idx + 1), 0.05 * (idx + 1)]] * 3, dtype=np.float32),
        # Дефолтные якоря
        'stride': 8 * (2 ** idx)
    } for idx, level_name in enumerate(_PLOT_UTILS_DEFAULT_FPN_LEVEL_NAMES)
}
FPN_LEVEL_COLORS_VIZ = {"P3": ('#FF6347', '#FFA07A'), "P4": ('#4682B4', '#B0C4DE'), "P5": ('#9370DB', '#DDA0DD')}


# ----------------------------------------------------------------------------------

def denormalize_box_xywh_to_xyxy(box_xywh_norm, img_width_display, img_height_display):
    """Преобразует нормализованные [xc, yc, w, h] в пиксельные [xmin, ymin, xmax, ymax]."""
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
                                       level_name_viz,
                                       fpn_configs_dict_arg,  # Явно переданный СЛОВАРЬ конфигураций FPN уровней
                                       classes_list_arg):  # Явно переданный СПИСОК имен классов
    """Декодирует y_true для ОДНОГО уровня FPN в список объектов для визуализации."""
    decoded_objects_on_level = []
    if not isinstance(fpn_configs_dict_arg, dict) or level_name_viz not in fpn_configs_dict_arg:
        print(
            f"ОШИБКА (decode_single_level): Конфигурация для уровня '{level_name_viz}' не найдена в fpn_configs_dict_arg.")
        return decoded_objects_on_level

    level_config_from_arg = fpn_configs_dict_arg[level_name_viz]
    expected_keys = ['grid_h', 'grid_w', 'anchors_wh_normalized', 'num_anchors']
    if not all(k in level_config_from_arg for k in expected_keys):
        print(f"ОШИБКА (decode_single_level): Неполная конфигурация для уровня '{level_name_viz}'. Отсутствуют ключи: "
              f"{[k for k in expected_keys if k not in level_config_from_arg]}")
        return decoded_objects_on_level

    grid_h, grid_w = level_config_from_arg['grid_h'], level_config_from_arg['grid_w']
    anchors_wh_norm_this_level = level_config_from_arg.get('anchors_wh_normalized')
    num_anchors_from_config = level_config_from_arg.get('num_anchors', 0)

    if anchors_wh_norm_this_level is None or not isinstance(anchors_wh_norm_this_level, np.ndarray) or \
            anchors_wh_norm_this_level.ndim != 2 or anchors_wh_norm_this_level.shape[1] != 2 or \
            anchors_wh_norm_this_level.shape[0] != num_anchors_from_config:
        print(f"ОШИБКА (decode_single_level): Некорректный формат anchors_wh_normalized для уровня '{level_name_viz}'. "
              f"Ожидался np.array формы ({num_anchors_from_config},2). "
              f"Получено: {type(anchors_wh_norm_this_level)} с формой {getattr(anchors_wh_norm_this_level, 'shape', 'N/A')}")
        return decoded_objects_on_level

    num_anchors_from_y_true = y_true_level_np.shape[2]
    if num_anchors_from_y_true != num_anchors_from_config:
        print(
            f"КРИТИЧЕСКАЯ ОШИБКА (plot_utils - decode_single): Несоответствие количества якорей для уровня {level_name_viz}!")
        print(f"  y_true имеет {num_anchors_from_y_true} якорей (из его формы).")
        print(f"  Конфигурация уровня (num_anchors) ожидает {num_anchors_from_config} якорей.")
        return decoded_objects_on_level

    objectness_scores = y_true_level_np[..., 4]
    responsible_indices = np.argwhere(objectness_scores > 0.5)

    for idx_gh, idx_gw, idx_anchor in responsible_indices:
        anchor_data = y_true_level_np[idx_gh, idx_gw, idx_anchor]
        tx, ty, tw, th = anchor_data[0:4]
        box_center_x_norm = (tx + float(idx_gw)) / float(grid_w)
        box_center_y_norm = (ty + float(idx_gh)) / float(grid_h)

        anchor_w_norm = anchors_wh_norm_this_level[idx_anchor, 0]
        anchor_h_norm = anchors_wh_norm_this_level[idx_anchor, 1]
        box_width_norm = np.exp(tw) * anchor_w_norm
        box_height_norm = np.exp(th) * anchor_h_norm
        box_xywh_norm = [box_center_x_norm, box_center_y_norm, box_width_norm, box_height_norm]

        class_one_hot = anchor_data[5:]
        class_id = np.argmax(class_one_hot)
        class_name = classes_list_arg[class_id] if 0 <= class_id < len(classes_list_arg) else "Unknown"

        decoded_objects_on_level.append({
            "class_name": class_name, "class_id": class_id, "box_xywh_norm": box_xywh_norm,
            "anchor_idx": idx_anchor, "grid_cell": [idx_gh, idx_gw]
        })
    return decoded_objects_on_level




def visualize_fpn_gt_assignments(
        image_np_processed,  # Изображение, как оно подавалось в модель (например, 416x416, нормализованное [0,1])
        y_true_fpn_tuple_np,  # Кортеж из (y_true_P3_np, y_true_P4_np, y_true_P5_np) для GT
        fpn_level_names,  # Список имен уровней FPN, например ['P3', 'P4', 'P5']
        fpn_configs,  # Словарь с конфигурацией для каждого уровня FPN (страйды, якоря, размеры сетки)
        classes_list,  # Список имен классов для детектора (['pit', 'crack'])
        title_prefix="",
        show_grid_for_level=None  # "P3", "P4", "P5" или None (не рисовать сетку). Можно также "ALL"
):
    """
    Визуализирует Ground Truth (из y_true) и Предсказания модели на одном изображении.
    Рисует сетку только для указанного уровня FPN или не рисует вообще.
    """
    if not fpn_level_names or not isinstance(fpn_configs, dict) or not classes_list:
        print(
            "ОШИБКА (visualize_fpn_detections_vs_gt): fpn_level_names, fpn_configs или classes_list не предоставлены.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Один subplot для всех
    ax.imshow(image_np_processed)
    display_title = title_prefix
    ax.set_title(display_title, fontsize=10)

    img_h_display, img_w_display = image_np_processed.shape[:2]
    legend_handles_dict = {}

    # 1. Отрисовка Ground Truth (из y_true, назначенных на якоря)
    print(f"  Visualizing GT assignments from y_true for {title_prefix}:")
    for i, level_name_gt in enumerate(fpn_level_names):
        if i >= len(y_true_fpn_tuple_np): continue  # Если y_true короче, чем список имен уровней

        level_y_true_np_gt = y_true_fpn_tuple_np[i]
        level_config_gt = fpn_configs.get(level_name_gt)
        if not level_config_gt: continue

        box_color_gt, _ = FPN_LEVEL_COLORS_VIZ.get(level_name_gt, ('green', 'lightgray'))  # GT зеленым по умолчанию

        # Рисуем сетку, если указано
        if show_grid_for_level == level_name_gt or show_grid_for_level == "ALL":
            grid_h_level_gt, grid_w_level_gt = level_config_gt['grid_h'], level_config_gt['grid_w']
            if grid_w_level_gt > 0 and grid_h_level_gt > 0:
                x_step_gt = img_w_display / grid_w_level_gt
                y_step_gt = img_h_display / grid_h_level_gt
                for x_line_idx_gt in range(grid_w_level_gt + 1): ax.axvline(x_line_idx_gt * x_step_gt, color='gray',
                                                                            linestyle=':', linewidth=0.3, alpha=0.5)
                for y_line_idx_gt in range(grid_h_level_gt + 1): ax.axhline(y_line_idx_gt * y_step_gt, color='gray',
                                                                            linestyle=':', linewidth=0.3, alpha=0.5)

        decoded_gt_objects_on_level = decode_single_level_y_true_for_viz(
            level_y_true_np_gt, level_name_gt, fpn_configs, classes_list
        )
        if decoded_gt_objects_on_level:
            # print(f"    GT on {level_name_gt}: {len(decoded_gt_objects_on_level)} responsible anchors")
            for obj_info_gt in decoded_gt_objects_on_level:
                # box_xywh_norm это [xc_n, yc_n, w_n, h_n]
                xmin_gt_viz, ymin_gt_viz, xmax_gt_viz, ymax_gt_viz = denormalize_box_xywh_to_xyxy(
                    obj_info_gt['box_xywh_norm'], img_w_display, img_h_display)
                rect_gt_viz = patches.Rectangle((xmin_gt_viz, ymin_gt_viz), xmax_gt_viz - xmin_gt_viz,
                                                ymax_gt_viz - ymin_gt_viz,
                                                linewidth=1.5, edgecolor=box_color_gt, facecolor='none',
                                                linestyle='--')  # GT пунктиром
                ax.add_patch(rect_gt_viz)
                ax.text(xmin_gt_viz, ymin_gt_viz - 5,
                        f"GT:{obj_info_gt['class_name']}@{level_name_gt}(A{obj_info_gt['anchor_idx']})",
                        color=box_color_gt, fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.4, pad=0, edgecolor='none'))

                gt_legend_label = f"GT Assignment ({level_name_gt})"
                if gt_legend_label not in legend_handles_dict:
                    legend_handles_dict[gt_legend_label] = patches.Patch(color=box_color_gt, linestyle='--',
                                                                         label=gt_legend_label)

    if legend_handles_dict:
        ax.legend(handles=list(legend_handles_dict.values()), loc='lower right', fontsize='xx-small')

    plt.tight_layout(pad=0.5)
    plt.show()


def visualize_fpn_detections_vs_gt(
        image_np_processed,  # Изображение, как оно подавалось в модель (например, 416x416, нормализованное [0,1])
        y_true_fpn_tuple_np,  # Кортеж из (y_true_P3_np, y_true_P4_np, y_true_P5_np) для GT
        pred_boxes_norm_yxyx,  # Предсказанные рамки ПОСЛЕ NMS, нормализованные [ymin, xmin, ymax, xmax]
        pred_scores,  # Уверенности для предсказанных рамок
        pred_class_ids,  # ID классов для предсказанных рамок
        fpn_level_names,  # Список имен уровней FPN, например ['P3', 'P4', 'P5']
        fpn_configs,  # Словарь с конфигурацией для каждого уровня FPN (страйды, якоря, размеры сетки)
        classes_list,  # Список имен классов для детектора (['pit', 'crack'])
        original_gt_boxes_for_reference=None,
        # Опционально: (scaled_boxes_np_xyxy_norm, class_ids_np) для оригинальных GT
        title_prefix="",
        show_grid_for_level=None  # "P3", "P4", "P5" или None (не рисовать сетку). Можно также "ALL"
):
    """
    Визуализирует Ground Truth (из y_true) и Предсказания модели на одном изображении.
    Рисует сетку только для указанного уровня FPN или не рисует вообще.
    """
    if not fpn_level_names or not isinstance(fpn_configs, dict) or not classes_list:
        print(
            "ОШИБКА (visualize_fpn_detections_vs_gt): fpn_level_names, fpn_configs или classes_list не предоставлены.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Один subplot для всех
    ax.imshow(image_np_processed)
    display_title = title_prefix
    ax.set_title(display_title, fontsize=10)

    img_h_display, img_w_display = image_np_processed.shape[:2]
    legend_handles_dict = {}

    # 1. Отрисовка Ground Truth (из y_true, назначенных на якоря)
    print(f"  Visualizing GT assignments from y_true for {title_prefix}:")
    for i, level_name_gt in enumerate(fpn_level_names):
        if i >= len(y_true_fpn_tuple_np): continue  # Если y_true короче, чем список имен уровней

        level_y_true_np_gt = y_true_fpn_tuple_np[i]
        level_config_gt = fpn_configs.get(level_name_gt)
        if not level_config_gt: continue

        box_color_gt, _ = FPN_LEVEL_COLORS_VIZ.get(level_name_gt, ('green', 'lightgray'))  # GT зеленым по умолчанию

        # Рисуем сетку, если указано
        if show_grid_for_level == level_name_gt or show_grid_for_level == "ALL":
            grid_h_level_gt, grid_w_level_gt = level_config_gt['grid_h'], level_config_gt['grid_w']
            if grid_w_level_gt > 0 and grid_h_level_gt > 0:
                x_step_gt = img_w_display / grid_w_level_gt
                y_step_gt = img_h_display / grid_h_level_gt
                for x_line_idx_gt in range(grid_w_level_gt + 1): ax.axvline(x_line_idx_gt * x_step_gt, color='gray',
                                                                            linestyle=':', linewidth=0.3, alpha=0.5)
                for y_line_idx_gt in range(grid_h_level_gt + 1): ax.axhline(y_line_idx_gt * y_step_gt, color='gray',
                                                                            linestyle=':', linewidth=0.3, alpha=0.5)

        decoded_gt_objects_on_level = decode_single_level_y_true_for_viz(
            level_y_true_np_gt, level_name_gt, fpn_configs, classes_list
        )
        if decoded_gt_objects_on_level:
            # print(f"    GT on {level_name_gt}: {len(decoded_gt_objects_on_level)} responsible anchors")
            for obj_info_gt in decoded_gt_objects_on_level:
                # box_xywh_norm это [xc_n, yc_n, w_n, h_n]
                xmin_gt_viz, ymin_gt_viz, xmax_gt_viz, ymax_gt_viz = denormalize_box_xywh_to_xyxy(
                    obj_info_gt['box_xywh_norm'], img_w_display, img_h_display)
                rect_gt_viz = patches.Rectangle((xmin_gt_viz, ymin_gt_viz), xmax_gt_viz - xmin_gt_viz,
                                                ymax_gt_viz - ymin_gt_viz,
                                                linewidth=1.5, edgecolor=box_color_gt, facecolor='none',
                                                linestyle='--')  # GT пунктиром
                ax.add_patch(rect_gt_viz)
                ax.text(xmin_gt_viz, ymin_gt_viz - 5,
                        f"GT:{obj_info_gt['class_name']}@{level_name_gt}(A{obj_info_gt['anchor_idx']})",
                        color=box_color_gt, fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.4, pad=0, edgecolor='none'))

                gt_legend_label = f"GT Assignment ({level_name_gt})"
                if gt_legend_label not in legend_handles_dict:
                    legend_handles_dict[gt_legend_label] = patches.Patch(color=box_color_gt, linestyle='--',
                                                                         label=gt_legend_label)

    # 2. Отрисовка Предсказаний Модели (pred_boxes_norm_yxyx, pred_scores, pred_class_ids)
    num_predictions = pred_boxes_norm_yxyx.shape[0]
    print(f"  Visualizing {num_predictions} Predictions for {title_prefix}:")
    if num_predictions > 0:
        for i_pred in range(num_predictions):
            ymin_n_pred, xmin_n_pred, ymax_n_pred, xmax_n_pred = pred_boxes_norm_yxyx[i_pred]
            score_pred = pred_scores[i_pred]
            class_id_pred = int(pred_class_ids[i_pred])

            class_name_pred = classes_list[class_id_pred] if 0 <= class_id_pred < len(classes_list) else "Unknown"

            xmin_px_pred = int(xmin_n_pred * img_w_display)
            ymin_px_pred = int(ymin_n_pred * img_h_display)
            xmax_px_pred = int(xmax_n_pred * img_w_display)
            ymax_px_pred = int(ymax_n_pred * img_h_display)

            rect_width_pred, rect_height_pred = xmax_px_pred - xmin_px_pred, ymax_px_pred - ymin_px_pred

            # Цвет для предсказаний (например, синий для pit, оранжевый для crack)
            pred_color = 'blue' if class_name_pred == classes_list[0] else (
                'orange' if len(classes_list) > 1 and class_name_pred == classes_list[1] else 'purple')

            rect_pred_viz = patches.Rectangle((xmin_px_pred, ymin_px_pred), rect_width_pred, rect_height_pred,
                                              linewidth=2, edgecolor=pred_color, facecolor='none',
                                              linestyle='-')  # Предсказания сплошной линией
            ax.add_patch(rect_pred_viz)
            ax.text(xmin_px_pred, ymin_px_pred + rect_height_pred + 5,  # Текст под рамкой
                    f"Pred: {class_name_pred} ({score_pred:.2f})",
                    color=pred_color, fontsize=7, bbox=dict(facecolor='white', alpha=0.7, pad=0, edgecolor='none'))

            pred_legend_label = f"Prediction ({class_name_pred})"
            if pred_legend_label not in legend_handles_dict:
                legend_handles_dict[pred_legend_label] = patches.Patch(color=pred_color, label=pred_legend_label)

    # 3. Опционально: отрисовка оригинальных GT рамок (если они сильно отличаются от y_true назначений)
    if original_gt_boxes_for_reference:
        # ... (код отрисовки original_gt_boxes_for_reference как был, но можно другим цветом/стилем)
        # Предположим, это кортеж (scaled_boxes_np_xyxy_norm, class_ids_np)
        if isinstance(original_gt_boxes_for_reference, tuple) and len(original_gt_boxes_for_reference) == 2:
            scaled_boxes_np_ref, class_ids_np_ref = original_gt_boxes_for_reference
            if scaled_boxes_np_ref.ndim == 2 and scaled_boxes_np_ref.shape[1] == 4 and class_ids_np_ref.ndim == 1:
                for i_ref in range(scaled_boxes_np_ref.shape[0]):
                    xmin_n_ref, ymin_n_ref, xmax_n_ref, ymax_n_ref = scaled_boxes_np_ref[
                        i_ref]  # Предполагаем, что это уже xmin,ymin,xmax,ymax
                    class_id_ref = int(class_ids_np_ref[i_ref])
                    class_name_ref = classes_list[class_id_ref] if 0 <= class_id_ref < len(classes_list) else "Unknown"

                    xmin_px_ref, ymin_px_ref, xmax_px_ref, ymax_px_ref = \
                        int(xmin_n_ref * img_w_display), int(ymin_n_ref * img_h_display), \
                            int(xmax_n_ref * img_w_display), int(ymax_n_ref * img_h_display)

                    rect_orig_gt = patches.Rectangle((xmin_px_ref, ymin_px_ref), xmax_px_ref - xmin_px_ref,
                                                     ymax_px_ref - ymin_px_ref,
                                                     linewidth=1, edgecolor='magenta', facecolor='none',
                                                     linestyle=':')  # Magenta, точечный
                    ax.add_patch(rect_orig_gt)
                    ax.text(xmin_px_ref + 5, ymin_px_ref + 5, f"OrigGT:{class_name_ref}", color='magenta', fontsize=5)

                    orig_gt_legend_label = "Original GT (for ref.)"
                    if orig_gt_legend_label not in legend_handles_dict:
                        legend_handles_dict[orig_gt_legend_label] = patches.Patch(color='magenta', linestyle=':',
                                                                                  label=orig_gt_legend_label)

    if legend_handles_dict:
        ax.legend(handles=list(legend_handles_dict.values()), loc='lower right', fontsize='xx-small')

    plt.tight_layout(pad=0.5)
    plt.show()



if __name__ == '__main__':
    print("--- Тестирование plot_utils.py (для FPN) ---")

    # --- Блок загрузки конфигурации для __main__ в plot_utils.py ---
    _plot_utils_main_current_script_dir = Path(__file__).resolve().parent
    _plot_utils_main_project_root_dir = _plot_utils_main_current_script_dir.parent.parent
    _plot_utils_main_detector_config_path = _plot_utils_main_project_root_dir / 'src' / 'configs' / 'detector_config.yaml'

    PLOT_UTILS_MAIN_DETECTOR_CONFIG = {}
    try:
        with open(_plot_utils_main_detector_config_path, 'r', encoding='utf-8') as f_main_cfg:
            PLOT_UTILS_MAIN_DETECTOR_CONFIG = yaml.safe_load(f_main_cfg)
        if not isinstance(PLOT_UTILS_MAIN_DETECTOR_CONFIG, dict):
            PLOT_UTILS_MAIN_DETECTOR_CONFIG = {}
    except Exception as e_cfg_plot_main_exc_local:
        print(
            f"ПРЕДУПРЕЖДЕНИЕ (__main__ plot_utils): Ошибка загрузки detector_config.yaml: {e_cfg_plot_main_exc_local}.")

    # Инициализируем тестовые конфигурации FPN и классы, используя дефолты модуля, если конфиг пуст или не найден
    main_fpn_params_from_cfg = PLOT_UTILS_MAIN_DETECTOR_CONFIG.get('fpn_detector_params', {})

    main_input_shape_list_from_cfg = main_fpn_params_from_cfg.get('input_shape',
                                                                  [_PLOT_UTILS_DEFAULT_TARGET_IMG_HEIGHT,
                                                                   _PLOT_UTILS_DEFAULT_TARGET_IMG_WIDTH, 3])

    main_target_h = main_input_shape_list_from_cfg[0]
    main_target_w = main_input_shape_list_from_cfg[1]

    main_classes_list_from_cfg = main_fpn_params_from_cfg.get('classes', _PLOT_UTILS_DEFAULT_CLASSES_LIST)
    main_num_classes = len(main_classes_list_from_cfg)
    main_fpn_level_names_from_cfg = main_fpn_params_from_cfg.get('detector_fpn_levels',
                                                                 _PLOT_UTILS_DEFAULT_FPN_LEVEL_NAMES)
    main_fpn_strides_yaml_from_cfg = main_fpn_params_from_cfg.get('detector_fpn_strides', {})
    main_fpn_anchors_yaml_from_cfg = main_fpn_params_from_cfg.get('detector_fpn_anchor_configs', {})

    main_fpn_configs_for_function_call = {}
    for ln_main_test_loop in main_fpn_level_names_from_cfg:
        lc_main_test_loop = main_fpn_anchors_yaml_from_cfg.get(ln_main_test_loop, {})
        default_module_level_config_main_loop = _PLOT_UTILS_DEFAULT_FPN_CONFIGS_DICT.get(ln_main_test_loop, {})

        ls_main_test_loop = main_fpn_strides_yaml_from_cfg.get(ln_main_test_loop,
                                                               default_module_level_config_main_loop.get('stride'))

        awh_main_list_from_cfg_loop = lc_main_test_loop.get('anchors_wh_normalized',
                                                            default_module_level_config_main_loop.get(
                                                                'anchors_wh_normalized'))
        na_main_from_cfg_loop = lc_main_test_loop.get('num_anchors_this_level')

        current_anchors_np_main_loop = np.array(awh_main_list_from_cfg_loop, dtype=np.float32)
        if current_anchors_np_main_loop.ndim != 2 or current_anchors_np_main_loop.shape[1] != 2 or \
                current_anchors_np_main_loop.shape[0] == 0:
            current_anchors_np_main_loop = default_module_level_config_main_loop.get('anchors_wh_normalized',
                                                                                     np.array([[0.1, 0.1]],
                                                                                              dtype=np.float32))

        na_main_final_loop = current_anchors_np_main_loop.shape[0]
        if na_main_from_cfg_loop is not None and na_main_from_cfg_loop != na_main_final_loop:
            pass

        main_fpn_configs_for_function_call[ln_main_test_loop] = {
            'stride': ls_main_test_loop,
            'anchors_wh_normalized': current_anchors_np_main_loop,
            'num_anchors': na_main_final_loop,
            'grid_h': main_target_h // ls_main_test_loop if ls_main_test_loop and ls_main_test_loop > 0 else 0,
            'grid_w': main_target_w // ls_main_test_loop if ls_main_test_loop and ls_main_test_loop > 0 else 0
        }
    # --- Конец блока загрузки конфига для __main__ plot_utils ---

    # Создаем фиктивное изображение
    img_test_fpn_main_plot = np.ones((main_target_h, main_target_w, 3), dtype=np.float32) * 0.7
    y_true_fpn_list_np_main_plot = []
    original_gt_for_viz_list_main_plot = []

    # Пример для одного уровня (например, P4)
    example_level_name_plot = 'P4'
    if example_level_name_plot in main_fpn_configs_for_function_call:
        level_cfg_p4_plot_main = main_fpn_configs_for_function_call[example_level_name_plot]
        y_p4_test_plot_main = np.zeros((level_cfg_p4_plot_main['grid_h'], level_cfg_p4_plot_main['grid_w'],
                                        level_cfg_p4_plot_main['num_anchors'],
                                        5 + main_num_classes), dtype=np.float32)
        if level_cfg_p4_plot_main['num_anchors'] > 0:
            gt_xc_n_main, gt_yc_n_main, gt_w_n_main, gt_h_n_main = 0.5, 0.5, 0.2, 0.15
            gt_class_id_main = 0
            original_gt_for_viz_list_main_plot.append(
                [gt_xc_n_main - gt_w_n_main / 2, gt_yc_n_main - gt_h_n_main / 2, gt_xc_n_main + gt_w_n_main / 2,
                 gt_yc_n_main + gt_h_n_main / 2, gt_class_id_main])
            obj_grid_x_main = int(gt_xc_n_main * level_cfg_p4_plot_main['grid_w'])
            obj_grid_y_main = int(gt_yc_n_main * level_cfg_p4_plot_main['grid_h'])
            best_anchor_idx_main = min(0, level_cfg_p4_plot_main['num_anchors'] - 1)
            tx_main = (gt_xc_n_main * level_cfg_p4_plot_main['grid_w']) - obj_grid_x_main
            ty_main = (gt_yc_n_main * level_cfg_p4_plot_main['grid_h']) - obj_grid_y_main
            anchor_w_main, anchor_h_main = level_cfg_p4_plot_main['anchors_wh_normalized'][best_anchor_idx_main]
            tw_main = np.log(max(gt_w_n_main, 1e-9) / (anchor_w_main + 1e-9))
            th_main = np.log(max(gt_h_n_main, 1e-9) / (anchor_h_main + 1e-9))
            y_p4_test_plot_main[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 0:4] = [tx_main, ty_main,
                                                                                                tw_main, th_main]
            y_p4_test_plot_main[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 4] = 1.0
            y_p4_test_plot_main[obj_grid_y_main, obj_grid_x_main, best_anchor_idx_main, 5 + gt_class_id_main] = 1.0

    for ln_fill_main_plot in main_fpn_level_names_from_cfg:
        if ln_fill_main_plot == example_level_name_plot and example_level_name_plot in main_fpn_configs_for_function_call and \
                level_cfg_p4_plot_main['num_anchors'] > 0:
            y_true_fpn_list_np_main_plot.append(y_p4_test_plot_main)
        else:
            cfg_fill_main_plot = main_fpn_configs_for_function_call.get(ln_fill_main_plot,
                                                                        _PLOT_UTILS_DEFAULT_FPN_CONFIGS_DICT.get(
                                                                            ln_fill_main_plot))
            y_true_fpn_list_np_main_plot.append(np.zeros((cfg_fill_main_plot['grid_h'], cfg_fill_main_plot['grid_w'],
                                                          cfg_fill_main_plot['num_anchors'], 5 + main_num_classes),
                                                         dtype=np.float32))

    if original_gt_for_viz_list_main_plot:
        temp_boxes_main_plot = np.array(
            [[item[0], item[1], item[2], item[3]] for item in original_gt_for_viz_list_main_plot], dtype=np.float32)
        temp_ids_main_plot = np.array([item[4] for item in original_gt_for_viz_list_main_plot], dtype=np.int32)
        original_gt_tuple_main_plot = (temp_boxes_main_plot, temp_ids_main_plot)
    else:
        original_gt_tuple_main_plot = (np.zeros((0, 4), dtype=np.float32), np.zeros((0), dtype=np.int32))

    print("\nВызов visualize_fpn_gt_assignments для тестовых данных из __main__ plot_utils...")
    visualize_fpn_gt_assignments(
        image_np_processed=img_test_fpn_main_plot,
        y_true_fpn_tuple_np=tuple(y_true_fpn_list_np_main_plot),
        fpn_level_names=main_fpn_level_names_from_cfg,
        fpn_configs=main_fpn_configs_for_function_call,
        classes_list=main_classes_list_from_cfg,
        title_prefix="FPN GT Test (plot_utils standalone)"
    )
    print("\n--- Тестирование plot_utils.py завершено ---")