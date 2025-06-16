# src/utils/plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import os

# --- Загрузка Конфигурации (для доступа к параметрам FPN и классам) ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

DETECTOR_CONFIG_PLOT = {}
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_PLOT = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_PLOT, dict):
        DETECTOR_CONFIG_PLOT = {}
        print(
            "ПРЕДУПРЕЖДЕНИЕ (plot_utils.py): detector_config.yaml пуст или имеет неверный формат. Используются дефолты.")
except FileNotFoundError:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (plot_utils.py): Файл detector_config.yaml не найден: {_detector_config_path}. Используются дефолты.")
except Exception as e:  # Более общее исключение для yaml.YAMLError и других
    print(f"ПРЕДУПРЕЖДЕНИЕ (plot_utils.py): Ошибка загрузки detector_config.yaml: {e}. Используются дефолты.")

# Используем дефолты, если конфиг не загружен или не содержит нужных ключей
INPUT_SHAPE_PLOT = tuple(DETECTOR_CONFIG_PLOT.get('input_shape', [416, 416, 3]))
TARGET_IMG_HEIGHT_PLOT = INPUT_SHAPE_PLOT[0]
TARGET_IMG_WIDTH_PLOT = INPUT_SHAPE_PLOT[1]
CLASSES_LIST_PLOT = DETECTOR_CONFIG_PLOT.get('classes', ['pit', 'crack'])  # Обновлено на более релевантные дефолты

# Глобальная переменная для этого модуля, инициализируемая из конфига
FPN_LEVELS_CONFIG_PLOT = {}
_fpn_anchor_configs_from_yaml_plot = DETECTOR_CONFIG_PLOT.get('fpn_anchor_configs',
                                                              {})  # Используем другое имя для избежания конфликта

for level_name_cfg_plot in ["P3", "P4", "P5"]:
    default_stride_cfg_plot = {'P3': 8, 'P4': 16, 'P5': 32}.get(level_name_cfg_plot, 16)
    # Дефолтное количество якорей, если не указано
    num_anchors_default = _fpn_anchor_configs_from_yaml_plot.get(level_name_cfg_plot, {}).get('num_anchors_this_level',
                                                                                              3)
    default_anchors_cfg_plot = [[0.1, 0.1]] * num_anchors_default

    level_cfg_yaml_plot = _fpn_anchor_configs_from_yaml_plot.get(level_name_cfg_plot, {})

    FPN_LEVELS_CONFIG_PLOT[level_name_cfg_plot] = {
        'anchors_wh': np.array(level_cfg_yaml_plot.get('anchors_wh_normalized', default_anchors_cfg_plot),
                               dtype=np.float32),
        'num_anchors': level_cfg_yaml_plot.get('num_anchors_this_level', 3),
        'grid_h': TARGET_IMG_HEIGHT_PLOT // level_cfg_yaml_plot.get('stride', default_stride_cfg_plot),
        'grid_w': TARGET_IMG_WIDTH_PLOT // level_cfg_yaml_plot.get('stride', default_stride_cfg_plot),
        'stride': level_cfg_yaml_plot.get('stride', default_stride_cfg_plot)
    }

NUM_CLASSES_PLOT = len(CLASSES_LIST_PLOT)

FPN_LEVEL_COLORS = {
    "P3": ('r', 'lightcoral'),
    "P4": ('b', 'lightskyblue'),
    "P5": ('m', 'plum')
}


def denormalize_box_xywh_to_xyxy(box_xywh_norm, img_width_display, img_height_display):
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


def decode_y_true_level_for_visualization(y_true_level_np, level_config_plot, classes_list_plot_arg):
    decoded_objects_level = []
    grid_h, grid_w = level_config_plot['grid_h'], level_config_plot['grid_w']
    anchors_wh_norm_level = level_config_plot['anchors_wh']

    objectness_scores = y_true_level_np[..., 4]
    responsible_indices = np.argwhere(objectness_scores > 0.5)

    for idx_gh, idx_gw, idx_anchor in responsible_indices:
        anchor_data = y_true_level_np[idx_gh, idx_gw, idx_anchor]
        tx, ty, tw, th = anchor_data[0:4]

        # Восстанавливаем нормализованные координаты центра bounding box'а относительно всего изображения
        # В detector_data_loader.py мы кодировали:
        # tx = grid_x_center_float - float(grid_x_idx)
        # ty = grid_y_center_float - float(grid_y_idx)
        box_center_x_norm = (idx_gw + tx) / grid_w
        box_center_y_norm = (idx_gh + ty) / grid_h

        anchor_w_norm = anchors_wh_norm_level[idx_anchor, 0]
        anchor_h_norm = anchors_wh_norm_level[idx_anchor, 1]
        box_width_norm = np.exp(tw) * anchor_w_norm
        box_height_norm = np.exp(th) * anchor_h_norm
        box_xywh_norm = [box_center_x_norm, box_center_y_norm, box_width_norm, box_height_norm]

        class_one_hot = anchor_data[5:]
        class_id = np.argmax(class_one_hot)
        class_name = classes_list_plot_arg[class_id] if 0 <= class_id < len(classes_list_plot_arg) else "Unknown"

        decoded_objects_level.append({
            "class_name": class_name, "class_id": class_id, "box_xywh_norm": box_xywh_norm,
            "anchor_idx": idx_anchor, "grid_cell": [idx_gh, idx_gw]
        })
    return decoded_objects_level


def visualize_fpn_data_sample(image_np, y_true_fpn_tuple,
                              fpn_levels_config_to_use, classes_list_to_use,
                              display_img_width, display_img_height,
                              title="FPN Ground Truth Sample"):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title(title, fontsize=10)
    legend_handles_dict = {}

    for level_idx, level_name in enumerate(["P3", "P4", "P5"]):
        y_true_level_np = y_true_fpn_tuple[level_idx]
        level_config = fpn_levels_config_to_use[level_name]
        box_color, grid_color = FPN_LEVEL_COLORS.get(level_name, ('k', 'gray'))

        grid_h, grid_w = level_config['grid_h'], level_config['grid_w']
        x_step = display_img_width / grid_w
        y_step = display_img_height / grid_h

        grid_label = f'{level_name} Grid (stride {level_config["stride"]})'
        if grid_label not in legend_handles_dict:
            legend_handles_dict[grid_label] = patches.Patch(color=grid_color, linestyle=':', label=grid_label,
                                                            alpha=0.5)

        for i in range(grid_w + 1): ax.axvline(i * x_step, color=grid_color, linestyle=':', linewidth=0.7, alpha=0.5)
        for i in range(grid_h + 1): ax.axhline(i * y_step, color=grid_color, linestyle=':', linewidth=0.7, alpha=0.5)

        decoded_objects_on_level = decode_y_true_level_for_visualization(
            y_true_level_np, level_config, classes_list_to_use
        )

        if decoded_objects_on_level:
            print(f"  На уровне {level_name} найдено {len(decoded_objects_on_level)} объектов для визуализации.")
            for obj_info in decoded_objects_on_level:
                class_name = obj_info['class_name']
                box_xywh_norm = obj_info['box_xywh_norm']
                xmin, ymin, xmax, ymax = denormalize_box_xywh_to_xyxy(box_xywh_norm, display_img_width,
                                                                      display_img_height)
                rect_gt = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1.5,
                                            edgecolor=box_color, facecolor='none')
                ax.add_patch(rect_gt)
                ax.text(xmin, ymin - 2, f"{class_name}@{level_name}(A{obj_info['anchor_idx']})",
                        color=box_color, fontsize=7, bbox=dict(facecolor='white', alpha=0.6, pad=0, edgecolor='none'))

                obj_label_legend = f"GT {level_name}: {class_name}"
                if obj_label_legend not in legend_handles_dict:
                    legend_handles_dict[obj_label_legend] = patches.Patch(color=box_color, label=obj_label_legend)

    if legend_handles_dict:
        ax.legend(handles=list(legend_handles_dict.values()), loc='best', fontsize='x-small')
    plt.show()


if __name__ == '__main__':
    print("--- Тестирование plot_utils.py (для FPN) ---")
    test_img_height = TARGET_IMG_HEIGHT_PLOT
    test_img_width = TARGET_IMG_WIDTH_PLOT
    test_num_classes = NUM_CLASSES_PLOT

    img_with_object = np.ones((test_img_height, test_img_width, 3), dtype=np.float32) * 0.8

    y_true_p3_test = np.zeros(shape=(FPN_LEVELS_CONFIG_PLOT["P3"]['grid_h'], FPN_LEVELS_CONFIG_PLOT["P3"]['grid_w'],
                                     FPN_LEVELS_CONFIG_PLOT["P3"]['num_anchors'], 5 + test_num_classes),
                              dtype=np.float32)
    y_true_p4_test = np.zeros(shape=(FPN_LEVELS_CONFIG_PLOT["P4"]['grid_h'], FPN_LEVELS_CONFIG_PLOT["P4"]['grid_w'],
                                     FPN_LEVELS_CONFIG_PLOT["P4"]['num_anchors'], 5 + test_num_classes),
                              dtype=np.float32)
    y_true_p5_test = np.zeros(shape=(FPN_LEVELS_CONFIG_PLOT["P5"]['grid_h'], FPN_LEVELS_CONFIG_PLOT["P5"]['grid_w'],
                                     FPN_LEVELS_CONFIG_PLOT["P5"]['num_anchors'], 5 + test_num_classes),
                              dtype=np.float32)

    gt_xc_norm_obj1 = 0.5;
    gt_yc_norm_obj1 = 0.5
    gt_w_norm_obj1 = 0.15;
    gt_h_norm_obj1 = 0.15
    gt_class_id_obj1 = 0

    level_to_assign_obj1 = "P4"
    config_obj1 = FPN_LEVELS_CONFIG_PLOT[level_to_assign_obj1]  # Используем FPN_LEVELS_CONFIG_PLOT
    y_true_target_obj1 = y_true_p4_test

    obj_grid_x_obj1 = int(gt_xc_norm_obj1 * config_obj1['grid_w'])
    obj_grid_y_obj1 = int(gt_yc_norm_obj1 * config_obj1['grid_h'])
    best_anchor_idx_obj1 = 1

    tx1 = (gt_xc_norm_obj1 * config_obj1['grid_w']) - obj_grid_x_obj1
    ty1 = (gt_yc_norm_obj1 * config_obj1['grid_h']) - obj_grid_y_obj1
    anchor_w1, anchor_h1 = config_obj1['anchors_wh'][best_anchor_idx_obj1]
    tw1 = np.log(max(gt_w_norm_obj1, 1e-9) / (anchor_w1 + 1e-9))
    th1 = np.log(max(gt_h_norm_obj1, 1e-9) / (anchor_h1 + 1e-9))

    y_true_target_obj1[obj_grid_y_obj1, obj_grid_x_obj1, best_anchor_idx_obj1, 0:4] = [tx1, ty1, tw1, th1]
    y_true_target_obj1[obj_grid_y_obj1, obj_grid_x_obj1, best_anchor_idx_obj1, 4] = 1.0
    y_true_target_obj1[obj_grid_y_obj1, obj_grid_x_obj1, best_anchor_idx_obj1, 5 + gt_class_id_obj1] = 1.0

    print("\nТест 1: Изображение с одним объектом (на P4)")
    visualize_fpn_data_sample(
        img_with_object, (y_true_p3_test, y_true_p4_test, y_true_p5_test),
        FPN_LEVELS_CONFIG_PLOT, CLASSES_LIST_PLOT,  # Передаем правильные переменные
        test_img_width, test_img_height,
        title=f"Test 1: One Object on P4 (Class: {CLASSES_LIST_PLOT[gt_class_id_obj1]})"
    )

    print("\nТест 2: Пустое изображение")
    img_empty = np.ones((test_img_height, test_img_width, 3), dtype=np.float32) * 0.5
    y_true_p3_empty = np.zeros_like(y_true_p3_test)
    y_true_p4_empty = np.zeros_like(y_true_p4_test)
    y_true_p5_empty = np.zeros_like(y_true_p5_test)
    visualize_fpn_data_sample(
        img_empty, (y_true_p3_empty, y_true_p4_empty, y_true_p5_empty),
        FPN_LEVELS_CONFIG_PLOT, CLASSES_LIST_PLOT,  # Передаем правильные переменные
        test_img_width, test_img_height,
        title="Test 2: Empty Image (No Objects)"
    )
    print("\n--- Тестирование plot_utils.py завершено ---")