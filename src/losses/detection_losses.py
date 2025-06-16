# src/losses/detection_losses.py
import tensorflow as tf
import yaml
import os
import numpy as np
import sys  # Для вывода в stderr в tf.print

# --- Загрузка Конфигурации ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

DETECTOR_CONFIG_LOSS = {}
CONFIG_LOAD_SUCCESS_LOSS = True
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_LOSS = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_LOSS, dict): DETECTOR_CONFIG_LOSS = {}; CONFIG_LOAD_SUCCESS_LOSS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_LOSS = False;
    print(f"ПРЕДУПРЕЖДЕНИЕ (detection_losses.py): Файл detector_config.yaml не найден: {_detector_config_path}.")
except yaml.YAMLError as e:
    CONFIG_LOAD_SUCCESS_LOSS = False;
    print(f"ОШИБКА (detection_losses.py): YAML в detector_config.yaml: {e}")

if not CONFIG_LOAD_SUCCESS_LOSS:
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки detector_config.yaml. Используются дефолты в detection_losses.py.")
    DETECTOR_CONFIG_LOSS.setdefault('num_classes', 2)
    DETECTOR_CONFIG_LOSS.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG_LOSS.setdefault('fpn_anchor_configs', {
        'P3': {'grid_h': 52, 'grid_w': 52, 'num_anchors_this_level': 3, 'stride': 8,
               'anchors_wh_normalized': [[0.1, 0.1]] * 3},
        'P4': {'grid_h': 26, 'grid_w': 26, 'num_anchors_this_level': 3, 'stride': 16,
               'anchors_wh_normalized': [[0.1, 0.1]] * 3},
        'P5': {'grid_h': 13, 'grid_w': 13, 'num_anchors_this_level': 3, 'stride': 32,
               'anchors_wh_normalized': [[0.1, 0.1]] * 3}
    })
    DETECTOR_CONFIG_LOSS.setdefault('loss_weights', {
        'coordinates': 1.0, 'objectness': 1.0, 'no_object': 0.5, 'classification': 1.0
    })

NUM_CLASSES_LOSS_FPN = DETECTOR_CONFIG_LOSS.get('num_classes', 2)

# --- ИНИЦИАЛИЗАЦИЯ FPN_LEVELS_CONFIG_LOSS_TEST (для блока if __name__ == '__main__') ---
_input_shape_list_loss_for_test = DETECTOR_CONFIG_LOSS.get('input_shape', [416, 416, 3])
_target_img_height_loss_for_test = _input_shape_list_loss_for_test[0]
_target_img_width_loss_for_test = _input_shape_list_loss_for_test[1]

FPN_LEVELS_CONFIG_LOSS_TEST = {}
_fpn_anchor_configs_from_yaml_loss_for_test = DETECTOR_CONFIG_LOSS.get('fpn_anchor_configs', {})

for _level_name_loss_test in ["P3", "P4", "P5"]:
    _default_stride_loss_test = {'P3': 8, 'P4': 16, 'P5': 32}.get(_level_name_loss_test, 16)
    _level_cfg_yaml_loss_test = _fpn_anchor_configs_from_yaml_loss_for_test.get(_level_name_loss_test, {})
    _num_anchors_default_loss_test = _level_cfg_yaml_loss_test.get('num_anchors_this_level', 3)
    _default_anchors_loss_test = [[0.1, 0.1]] * _num_anchors_default_loss_test

    FPN_LEVELS_CONFIG_LOSS_TEST[_level_name_loss_test] = {
        'anchors_wh': np.array(_level_cfg_yaml_loss_test.get('anchors_wh_normalized', _default_anchors_loss_test),
                               dtype=np.float32),
        'num_anchors': _level_cfg_yaml_loss_test.get('num_anchors_this_level', 3),
        'grid_h': _target_img_height_loss_for_test // _level_cfg_yaml_loss_test.get('stride',
                                                                                    _default_stride_loss_test),
        'grid_w': _target_img_width_loss_for_test // _level_cfg_yaml_loss_test.get('stride', _default_stride_loss_test),
        'stride': _level_cfg_yaml_loss_test.get('stride', _default_stride_loss_test)
    }
# --- КОНЕЦ ИНИЦИАЛИЗАЦИИ FPN_LEVELS_CONFIG_LOSS_TEST ---

COORD_LOSS_WEIGHT_FPN = DETECTOR_CONFIG_LOSS.get('loss_weights', {}).get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT_FPN = DETECTOR_CONFIG_LOSS.get('loss_weights', {}).get('objectness', 1.0)
NO_OBJECT_LOSS_WEIGHT_FPN = DETECTOR_CONFIG_LOSS.get('loss_weights', {}).get('no_object', 0.5)
CLASS_LOSS_WEIGHT_FPN = DETECTOR_CONFIG_LOSS.get('loss_weights', {}).get('classification', 1.0)


# --- Конец Загрузки Конфигурации ---


@tf.function
def calculate_single_level_loss(y_true_level, y_pred_level, num_classes_arg, level_name_debug="UnknownLevel"):
    # tf.print("--- calculate_single_level_loss for", level_name_debug, "---", output_stream=sys.stderr)
    # tf.print("  ", level_name_debug, "y_true_level INPUT shape:", tf.shape(y_true_level), output_stream=sys.stderr)
    # tf.print("  ", level_name_debug, "y_pred_level INPUT shape:", tf.shape(y_pred_level), output_stream=sys.stderr)

    shape_y_true = tf.shape(y_true_level)
    batch_size_dynamic = shape_y_true[0]

    # Ожидаемые признаки на якорь: 4 (коорд) + 1 (obj) + num_classes
    expected_features_per_anchor = 4 + 1 + num_classes_arg

    # Решейпим в (Batch, TotalPredictionsOnLevel, FeaturesPerPrediction)
    # TotalPredictionsOnLevel = grid_h * grid_w * num_anchors
    y_true_reshaped = tf.reshape(y_true_level, [batch_size_dynamic, -1, expected_features_per_anchor])
    y_pred_reshaped = tf.reshape(y_pred_level, [batch_size_dynamic, -1, expected_features_per_anchor])

    # tf.print(level_name_debug, "y_true_reshaped shape:", tf.shape(y_true_reshaped), output_stream=sys.stderr)

    true_boxes_encoded = y_true_reshaped[..., 0:4]
    pred_boxes_encoded_raw = y_pred_reshaped[..., 0:4]
    true_objectness = y_true_reshaped[..., 4:5]
    pred_objectness_logits = y_pred_reshaped[..., 4:5]
    true_classes_one_hot = y_true_reshaped[..., 5:5 + num_classes_arg]
    pred_classes_logits = y_pred_reshaped[..., 5:5 + num_classes_arg]

    objectness_bce_loss_per_item = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_objectness,
                                                                           logits=pred_objectness_logits)
    loss_weights_obj_tensor = tf.where(tf.equal(true_objectness, 1.0), OBJECTNESS_LOSS_WEIGHT_FPN,
                                       NO_OBJECT_LOSS_WEIGHT_FPN)
    weighted_objectness_loss_per_item = objectness_bce_loss_per_item * loss_weights_obj_tensor
    total_objectness_loss_level = tf.reduce_sum(weighted_objectness_loss_per_item) / tf.cast(
        tf.size(weighted_objectness_loss_per_item), tf.float32)

    objectness_mask_for_responsible = tf.squeeze(tf.cast(true_objectness, tf.float32), axis=-1)
    num_responsible_objects_float = tf.maximum(tf.reduce_sum(objectness_mask_for_responsible), 1e-6)

    class_bce_loss_per_item_per_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_classes_one_hot,
                                                                                logits=pred_classes_logits)
    class_loss_per_prediction = tf.reduce_sum(class_bce_loss_per_item_per_class, axis=-1)
    masked_class_loss_per_item = class_loss_per_prediction * objectness_mask_for_responsible
    total_class_loss_level = tf.reduce_sum(masked_class_loss_per_item) / num_responsible_objects_float
    total_class_loss_level = total_class_loss_level * CLASS_LOSS_WEIGHT_FPN

    delta_huber = 1.0;
    abs_error_coords = tf.abs(true_boxes_encoded - pred_boxes_encoded_raw)
    is_small_error_coords = abs_error_coords <= delta_huber
    squared_loss_coords = 0.5 * tf.square(abs_error_coords);
    linear_loss_coords = delta_huber * (abs_error_coords - 0.5 * delta_huber)
    coord_loss_individual_components = tf.where(is_small_error_coords, squared_loss_coords, linear_loss_coords)
    coord_loss_per_prediction = tf.reduce_sum(coord_loss_individual_components, axis=-1)
    masked_coord_loss_per_item = coord_loss_per_prediction * objectness_mask_for_responsible
    total_box_loss_level = tf.reduce_sum(masked_coord_loss_per_item) / num_responsible_objects_float
    total_box_loss_level = total_box_loss_level * COORD_LOSS_WEIGHT_FPN

    # tf.print(f"  {level_name_debug} Losses (TF context): Obj=", total_objectness_loss_level, " Cls=", total_class_loss_level, " Box=", total_box_loss_level, output_stream=sys.stderr)

    return total_objectness_loss_level, total_class_loss_level, total_box_loss_level


@tf.function
def compute_detector_loss_v2_fpn(y_true_fpn_tuple, y_pred_fpn_list):
    # tf.print("--- compute_detector_loss_v2_fpn (Top Level) ---", output_stream=sys.stderr)
    # tf.print("  y_true_P3 shape:", tf.shape(y_true_fpn_tuple[0]), "y_pred_P3 shape:", tf.shape(y_pred_fpn_list[0]), output_stream=sys.stderr)
    # ...

    total_loss_objectness_all_levels = tf.constant(0.0, dtype=tf.float32)
    total_loss_class_all_levels = tf.constant(0.0, dtype=tf.float32)
    total_loss_box_all_levels = tf.constant(0.0, dtype=tf.float32)
    fpn_level_loss_weights = [1.0, 1.0, 1.0]
    level_names_debug = ["P3", "P4", "P5"]

    for i in range(3):
        y_true_level = y_true_fpn_tuple[i]
        y_pred_level = y_pred_fpn_list[i]
        level_weight = fpn_level_loss_weights[i]

        obj_loss_level, cls_loss_level, box_loss_level = calculate_single_level_loss(
            y_true_level, y_pred_level, NUM_CLASSES_LOSS_FPN, level_name_debug=level_names_debug[i]
        )

        total_loss_objectness_all_levels += obj_loss_level * level_weight
        total_loss_class_all_levels += cls_loss_level * level_weight
        total_loss_box_all_levels += box_loss_level * level_weight

    final_total_loss = total_loss_objectness_all_levels + total_loss_class_all_levels + total_loss_box_all_levels
    # tf.print("Total Combined FPN Loss: ", final_total_loss, output_stream=sys.stderr)
    return final_total_loss


# --- Блок if __name__ == '__main__': для тестирования ---
if __name__ == '__main__':
    print(f"--- Тестирование detection_losses.py (для FPN) ---")
    if not CONFIG_LOAD_SUCCESS_LOSS: print("ОШИБКА: Конфиги не загружены...")
    print(f"Используемые веса потерь (глобальные):")
    print(f"  Координаты: {COORD_LOSS_WEIGHT_FPN}");
    print(f"  Objectness (позитивные): {OBJECTNESS_LOSS_WEIGHT_FPN}")
    print(f"  No Object (негативные): {NO_OBJECT_LOSS_WEIGHT_FPN}");
    print(f"  Классификация: {CLASS_LOSS_WEIGHT_FPN}")
    batch_size_test_loss, num_classes_test_loss = 2, NUM_CLASSES_LOSS_FPN
    y_true_list_test, y_pred_list_test = [], []
    level_names_test = ["P3", "P4", "P5"]
    for level_name_test in level_names_test:
        level_cfg_test = FPN_LEVELS_CONFIG_LOSS_TEST.get(level_name_test)
        if not level_cfg_test:
            level_cfg_test = {'grid_h': 13, 'grid_w': 13, 'num_anchors': 3, 'anchors_wh': np.array([[0.1, 0.1]] * 3)}
            print(f"WARN: Нет конфига для {level_name_test}")

        gh, gw, na = level_cfg_test['grid_h'], level_cfg_test['grid_w'], level_cfg_test['num_anchors']
        # Ожидаемая форма для y_true_level и y_pred_level: (batch, grid_h, grid_w, num_anchors, 5 + num_classes)
        y_true_shape_level = (batch_size_test_loss, gh, gw, na, 5 + num_classes_test_loss)
        y_true_level_np = np.zeros(y_true_shape_level, dtype=np.float32)

        if level_name_test == "P4":  # Добавляем один "настоящий" объект на уровне P4 для теста
            anchor_w_for_test = level_cfg_test['anchors_wh'][0, 0]
            anchor_h_for_test = level_cfg_test['anchors_wh'][0, 1]
            # Закодированные координаты (tx, ty, tw, th)
            y_true_level_np[0, gh // 2, gw // 2, 0, 0:4] = [0.5, 0.5, np.log(0.2 / (anchor_w_for_test + 1e-9)),
                                                            np.log(0.3 / (anchor_h_for_test + 1e-9))]
            y_true_level_np[0, gh // 2, gw // 2, 0, 4] = 1.0  # objectness = 1
            y_true_level_np[0, gh // 2, gw // 2, 0, 5 + 0] = 1.0  # class_id = 0 (например, "pit")
        y_true_list_test.append(tf.constant(y_true_level_np, dtype=np.float32))

        # Создаем y_pred (логиты), которые немного отличаются от y_true
        y_pred_level_logits_np = np.random.randn(*y_true_shape_level).astype(np.float32) * 0.1
        if level_name_test == "P4":
            y_pred_level_logits_np[0, gh // 2, gw // 2, 0, 0:4] = y_true_level_np[0, gh // 2, gw // 2, 0,
                                                                  0:4] + np.random.randn(4) * 0.05
            y_pred_level_logits_np[0, gh // 2, gw // 2, 0, 4] = 5.0  # Высокий логит для objectness
            pred_cls_obj_level = np.full(num_classes_test_loss, -5.0, dtype=np.float32);
            pred_cls_obj_level[0] = 5.0
            y_pred_level_logits_np[0, gh // 2, gw // 2, 0, 5:] = pred_cls_obj_level
        y_pred_list_test.append(tf.constant(y_pred_level_logits_np, dtype=np.float32))

    print("\nТестирование функции потерь FPN (compute_detector_loss_v2_fpn)...")
    # Важно передавать y_true как кортеж, а y_pred как список (согласно сигнатуре функции и выходам модели)
    total_loss_fpn = compute_detector_loss_v2_fpn(tuple(y_true_list_test), y_pred_list_test)
    print(f"  Общая потеря FPN (случайные+1 объект): {total_loss_fpn.numpy():.4f}")

    y_pred_perfect_list_test = []
    for y_true_level_tf_perfect in y_true_list_test:
        y_true_np_perf = y_true_level_tf_perfect.numpy();
        y_pred_lvl_perf_np = np.zeros_like(y_true_np_perf, dtype=np.float32)
        y_pred_lvl_perf_np[..., 0:4] = y_true_np_perf[..., 0:4]  # Идеальные предсказания координат
        y_pred_lvl_perf_np[..., 4:5] = np.where(y_true_np_perf[..., 4:5] > 0.5, 10.0,
                                                -10.0)  # Идеальные логиты objectness
        true_cls_one_hot_perf_lvl = y_true_np_perf[..., 5:]
        pred_cls_logits_perf_lvl = np.where(true_cls_one_hot_perf_lvl > 0.5, 10.0, -10.0)  # Идеальные логиты классов
        y_pred_lvl_perf_np[..., 5:] = pred_cls_logits_perf_lvl
        y_pred_perfect_list_test.append(tf.constant(y_pred_lvl_perf_np, dtype=np.float32))

    loss_fpn_perfect = compute_detector_loss_v2_fpn(tuple(y_true_list_test), y_pred_perfect_list_test)
    print(f"  Общая потеря FPN (для 'идеальных' логитов): {loss_fpn_perfect.numpy():.4f}")
    print("\n--- Тестирование detection_losses.py (FPN) завершено ---")