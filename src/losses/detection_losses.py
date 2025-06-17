# src/losses/detection_losses.py
import tensorflow as tf
import yaml
import os
import numpy as np

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))  # Корень проекта
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

DETECTOR_CONFIG = {}
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict):
        print(
            f"ПРЕДУПРЕЖДЕНИЕ (detection_losses.py): detector_config.yaml пуст или неверный формат. Используются дефолты.")
        DETECTOR_CONFIG = {}  # Инициализируем пустым словарем, чтобы .get не падал
except FileNotFoundError:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (detection_losses.py): Файл detector_config.yaml не найден: {_detector_config_path}. Используются дефолты.")
except Exception as e:
    print(f"ОШИБКА (detection_losses.py): Не удалось загрузить detector_config.yaml: {e}. Используются дефолты.")

# --- Глобальные Параметры из Конфига (для использования в этом модуле) ---
FPN_PARAMS_FOR_LOSS_MODULE = DETECTOR_CONFIG.get('fpn_detector_params', {})
_input_shape_list_loss = FPN_PARAMS_FOR_LOSS_MODULE.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT_LOSS_MODULE = _input_shape_list_loss[0]
TARGET_IMG_WIDTH_LOSS_MODULE = _input_shape_list_loss[1]
NUM_CLASSES_LOSS_MODULE = FPN_PARAMS_FOR_LOSS_MODULE.get('num_classes', 2)

FPN_LEVEL_NAMES_LOSS_MODULE = FPN_PARAMS_FOR_LOSS_MODULE.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
FPN_STRIDES_CONFIG_LOSS_MODULE = FPN_PARAMS_FOR_LOSS_MODULE.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})
FPN_ANCHOR_CONFIGS_YAML_LOSS_MODULE = FPN_PARAMS_FOR_LOSS_MODULE.get('detector_fpn_anchor_configs', {})

FPN_LEVELS_CONFIG_FOR_LOSS_MODULE = {}
for _level_name in FPN_LEVEL_NAMES_LOSS_MODULE:
    _level_stride = FPN_STRIDES_CONFIG_LOSS_MODULE.get(_level_name,
                                                       8 * (2 ** FPN_LEVEL_NAMES_LOSS_MODULE.index(_level_name)))
    _level_anchor_data = FPN_ANCHOR_CONFIGS_YAML_LOSS_MODULE.get(_level_name, {})
    _num_anchors_default = 3  # Дефолтное количество якорей, если не указано
    _anchors_list_wh_default = [[0.1, 0.1]] * _num_anchors_default  # Дефолтные якоря, если не указаны

    _num_anchors = _level_anchor_data.get('num_anchors_this_level', _num_anchors_default)
    _anchors_list_wh = _level_anchor_data.get('anchors_wh_normalized', _anchors_list_wh_default)

    # Убедимся, что количество якорей соответствует списку, если список не пустой
    if _anchors_list_wh and len(_anchors_list_wh) != _num_anchors:
        # print(f"Warning (detection_losses.py): Mismatch in num_anchors for {_level_name}. Adjusting num_anchors to list_length.")
        _num_anchors = len(_anchors_list_wh)

    FPN_LEVELS_CONFIG_FOR_LOSS_MODULE[_level_name] = {
        'stride': _level_stride,
        'anchors_wh_normalized': np.array(_anchors_list_wh, dtype=np.float32),
        'num_anchors': _num_anchors,
        'grid_h': TARGET_IMG_HEIGHT_LOSS_MODULE // _level_stride,
        'grid_w': TARGET_IMG_WIDTH_LOSS_MODULE // _level_stride
    }

LOSS_WEIGHTS_CFG_MODULE = DETECTOR_CONFIG.get('loss_weights', {})
COORD_LOSS_WEIGHT_MODULE = LOSS_WEIGHTS_CFG_MODULE.get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT_MODULE = LOSS_WEIGHTS_CFG_MODULE.get('objectness', 1.0)
NO_OBJECT_LOSS_WEIGHT_MODULE = LOSS_WEIGHTS_CFG_MODULE.get('no_object', 0.5)
CLASS_LOSS_WEIGHT_MODULE = LOSS_WEIGHTS_CFG_MODULE.get('classification', 1.0)

FOCAL_LOSS_OBJ_PARAMS_MODULE = DETECTOR_CONFIG.get('focal_loss_objectness_params', {})
USE_FOCAL_FOR_OBJECTNESS_MODULE_DEFAULT = FOCAL_LOSS_OBJ_PARAMS_MODULE.get('use_focal_loss', True)
FOCAL_ALPHA_MODULE_DEFAULT = FOCAL_LOSS_OBJ_PARAMS_MODULE.get('alpha', 0.25)
FOCAL_GAMMA_MODULE_DEFAULT = FOCAL_LOSS_OBJ_PARAMS_MODULE.get('gamma', 2.0)


# --- Конец Загрузки Конфигурации ---


@tf.function
def focal_loss_sigmoid(y_true, y_pred_logits, alpha=0.25, gamma=2.0):
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    p = tf.sigmoid(y_pred_logits)
    p_t = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    focal_loss_unreduced = alpha_weight_factor * modulating_factor * bce
    return focal_loss_unreduced


@tf.function
def calculate_single_level_loss(y_true_level, y_pred_level,
                                num_classes_sl=NUM_CLASSES_LOSS_MODULE,
                                coord_weight_sl=COORD_LOSS_WEIGHT_MODULE,
                                obj_weight_sl=OBJECTNESS_LOSS_WEIGHT_MODULE,
                                no_obj_weight_sl=NO_OBJECT_LOSS_WEIGHT_MODULE,
                                cls_weight_sl=CLASS_LOSS_WEIGHT_MODULE,
                                use_focal_for_objectness_sl=USE_FOCAL_FOR_OBJECTNESS_MODULE_DEFAULT,
                                focal_alpha_sl_arg=FOCAL_ALPHA_MODULE_DEFAULT,  # Переименовал, чтобы не конфликтовать
                                focal_gamma_sl_arg=FOCAL_GAMMA_MODULE_DEFAULT):
    shape_y_true = tf.shape(y_true_level)
    batch_size_sl = shape_y_true[0]
    num_features_total_sl = shape_y_true[-1]  # Последнее измерение

    y_true_reshaped_sl = tf.reshape(y_true_level, [batch_size_sl, -1, num_features_total_sl])
    y_pred_reshaped_sl = tf.reshape(y_pred_level, [batch_size_sl, -1, num_features_total_sl])

    true_boxes_encoded_sl = y_true_reshaped_sl[..., :4]
    pred_boxes_encoded_raw_sl = y_pred_reshaped_sl[..., :4]
    true_objectness_sl = y_true_reshaped_sl[..., 4:5]
    pred_objectness_logits_sl = y_pred_reshaped_sl[..., 4:5]
    true_classes_one_hot_sl = y_true_reshaped_sl[..., 5:]
    pred_classes_logits_sl = y_pred_reshaped_sl[..., 5:]

    if use_focal_for_objectness_sl:
        objectness_loss_per_item = focal_loss_sigmoid(
            y_true=true_objectness_sl, y_pred_logits=pred_objectness_logits_sl,
            alpha=focal_alpha_sl_arg, gamma=focal_gamma_sl_arg)
    else:
        objectness_loss_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_objectness_sl, logits=pred_objectness_logits_sl)

    loss_weights_obj_sl_tensor = tf.where(tf.equal(true_objectness_sl, 1.0), obj_weight_sl, no_obj_weight_sl)
    weighted_objectness_loss = objectness_loss_per_item * loss_weights_obj_sl_tensor
    total_objectness_loss_sl = tf.reduce_sum(weighted_objectness_loss) / tf.cast(tf.size(weighted_objectness_loss),
                                                                                 tf.float32)

    objectness_mask_sl = tf.squeeze(tf.cast(true_objectness_sl, tf.float32), axis=-1)
    num_responsible_objects_sl = tf.maximum(tf.reduce_sum(objectness_mask_sl), 1.0)

    class_bce_loss_per_item_per_class_sl = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot_sl, logits=pred_classes_logits_sl)
    class_loss_per_prediction_sl = tf.reduce_sum(class_bce_loss_per_item_per_class_sl, axis=-1)
    masked_class_loss_sl = class_loss_per_prediction_sl * objectness_mask_sl
    total_class_loss_sl = (tf.reduce_sum(masked_class_loss_sl) / num_responsible_objects_sl) * cls_weight_sl

    delta_huber = 1.0
    abs_error_coords_sl = tf.abs(true_boxes_encoded_sl - pred_boxes_encoded_raw_sl)
    is_small_error_coords_sl = abs_error_coords_sl <= delta_huber
    squared_loss_coords_sl = 0.5 * tf.square(abs_error_coords_sl)
    linear_loss_coords_sl = delta_huber * (abs_error_coords_sl - 0.5 * delta_huber)
    coord_loss_individual_components_sl = tf.where(is_small_error_coords_sl, squared_loss_coords_sl,
                                                   linear_loss_coords_sl)
    coord_loss_per_prediction_sl = tf.reduce_sum(coord_loss_individual_components_sl, axis=-1)
    masked_coord_loss_sl = coord_loss_per_prediction_sl * objectness_mask_sl
    total_box_loss_sl = (tf.reduce_sum(masked_coord_loss_sl) / num_responsible_objects_sl) * coord_weight_sl

    return total_objectness_loss_sl, total_class_loss_sl, total_box_loss_sl, num_responsible_objects_sl


# @tf.function # Убрал декоратор для compute_detector_loss_v2_fpn, чтобы debug_mode работал корректно
def compute_detector_loss_v2_fpn(y_true_fpn_tuple, y_pred_fpn_list,
                                 use_focal_for_obj_param=USE_FOCAL_FOR_OBJECTNESS_MODULE_DEFAULT,
                                 # Берем из конфига по умолчанию
                                 focal_alpha_param=FOCAL_ALPHA_MODULE_DEFAULT,
                                 focal_gamma_param=FOCAL_GAMMA_MODULE_DEFAULT):
    total_loss_across_levels = tf.constant(0.0, dtype=tf.float32)
    detailed_losses = {}
    debug_mode = os.environ.get("DEBUG_TRAINING_LOOP_ACTIVE") == "1"

    for i, level_name in enumerate(FPN_LEVEL_NAMES_LOSS_MODULE):
        y_true_level = y_true_fpn_tuple[i]
        y_pred_level = y_pred_fpn_list[i]

        obj_loss_level, cls_loss_level, box_loss_level, num_pos_level = calculate_single_level_loss(
            y_true_level, y_pred_level,
            use_focal_for_objectness_sl=use_focal_for_obj_param,  # Передаем флаг
            focal_alpha_sl_arg=focal_alpha_param,  # Передаем параметры Focal Loss
            focal_gamma_sl_arg=focal_gamma_param
            # Остальные веса потерь берутся из глобальных переменных модуля
        )
        level_total_loss = obj_loss_level + cls_loss_level + box_loss_level
        total_loss_across_levels += level_total_loss

        if debug_mode:
            detailed_losses[f'obj_loss_{level_name}'] = obj_loss_level
            detailed_losses[f'cls_loss_{level_name}'] = cls_loss_level
            detailed_losses[f'box_loss_{level_name}'] = box_loss_level
            detailed_losses[f'num_pos_anchors_{level_name}'] = tf.cast(num_pos_level,
                                                                       tf.float32)  # Приводим к float для единообразия

    if debug_mode:
        detailed_losses['total_loss'] = total_loss_across_levels
        return detailed_losses
    else:
        return total_loss_across_levels


# --- Тестовый блок if __name__ == '__main__': ---
if __name__ == '__main__':
    print(f"--- Тестирование detection_losses.py (FPN с Focal Loss опцией) ---")
    # Используем глобальные переменные модуля, загруженные из конфига в начале файла
    print(
        f"Focal Loss для Objectness будет использован (в тесте, если use_focal_for_obj_param=True): {USE_FOCAL_FOR_OBJECTNESS_MODULE_DEFAULT}")
    if USE_FOCAL_FOR_OBJECTNESS_MODULE_DEFAULT:
        print(f"  Focal Alpha: {FOCAL_ALPHA_MODULE_DEFAULT}, Focal Gamma: {FOCAL_GAMMA_MODULE_DEFAULT}")
    print(f"Веса потерь: COORD={COORD_LOSS_WEIGHT_MODULE}, OBJ_POS={OBJECTNESS_LOSS_WEIGHT_MODULE}, "
          f"OBJ_NEG={NO_OBJECT_LOSS_WEIGHT_MODULE}, CLS={CLASS_LOSS_WEIGHT_MODULE}")
    print(f"Количество классов: {NUM_CLASSES_LOSS_MODULE}")
    print("Параметры FPN уровней для теста (из FPN_LEVELS_CONFIG_FOR_LOSS_MODULE):")
    for _lname_test, _lcfg_test in FPN_LEVELS_CONFIG_FOR_LOSS_MODULE.items():
        print(f"  Уровень {_lname_test}: Сетка({_lcfg_test['grid_h']}x{_lcfg_test['grid_w']}), "
              f"Якорей {_lcfg_test['num_anchors']}, Страйд {_lcfg_test['stride']}")

    batch_size_test = 2
    y_true_all_levels_list_test = []
    y_pred_all_levels_list_test_ideal = []
    y_pred_all_levels_list_test_random = []

    for level_name_test_main in FPN_LEVEL_NAMES_LOSS_MODULE:
        level_cfg_main_test = FPN_LEVELS_CONFIG_FOR_LOSS_MODULE.get(level_name_test_main)
        if not level_cfg_main_test: continue  # Пропускаем, если конфига для уровня нет

        gh, gw, num_a = level_cfg_main_test['grid_h'], level_cfg_main_test['grid_w'], level_cfg_main_test['num_anchors']

        y_true_shape_level = (batch_size_test, gh, gw, num_a, 5 + NUM_CLASSES_LOSS_MODULE)
        y_true_level_np = np.zeros(y_true_shape_level, dtype=np.float32)

        # Добавляем позитивные примеры для теста
        if gh > 0 and gw > 0 and num_a > 0:  # Проверка, что размеры не нулевые
            if level_name_test_main == "P3":
                y_true_level_np[0, gh // 2, gw // 2, 0, 0:4] = [0.5, 0.5, np.log(0.1 / 0.1 + 1e-9),
                                                                np.log(0.1 / 0.1 + 1e-9)]
                y_true_level_np[0, gh // 2, gw // 2, 0, 4] = 1.0
                y_true_level_np[0, gh // 2, gw // 2, 0, 5 + 0] = 1.0  # class 0
            elif level_name_test_main == "P4" and batch_size_test > 1:
                y_true_level_np[1, gh // 3, gw // 3, min(1, num_a - 1), 0:4] = [0.3, 0.3, np.log(0.3 / 0.1 + 1e-9),
                                                                                np.log(0.3 / 0.1 + 1e-9)]
                y_true_level_np[1, gh // 3, gw // 3, min(1, num_a - 1), 4] = 1.0
                y_true_level_np[1, gh // 3, gw // 3, min(1, num_a - 1), 5 + 1] = 1.0  # class 1

        y_true_all_levels_list_test.append(tf.constant(y_true_level_np, dtype=tf.float32))

        y_pred_logits_level_np_ideal = np.zeros_like(y_true_level_np, dtype=np.float32)
        y_pred_logits_level_np_ideal[..., 0:4] = y_true_level_np[..., 0:4]
        y_pred_logits_level_np_ideal[..., 4:5] = np.where(y_true_level_np[..., 4:5] > 0.5, 10.0, -10.0)
        true_cls_one_hot_level = y_true_level_np[..., 5:]
        pred_cls_logits_level_ideal = np.where(true_cls_one_hot_level > 0.5, 10.0, -10.0)
        y_pred_logits_level_np_ideal[..., 5:] = pred_cls_logits_level_ideal
        y_pred_all_levels_list_test_ideal.append(tf.constant(y_pred_logits_level_np_ideal, dtype=tf.float32))

        y_pred_all_levels_list_test_random.append(tf.random.normal(shape=y_true_shape_level, dtype=tf.float32) * 0.5)

    y_true_fpn_tuple_test = tuple(y_true_all_levels_list_test)

    if not y_true_all_levels_list_test:  # Если не создалось ни одного y_true (например, из-за проблем с конфигом FPN_LEVELS_CONFIG)
        print("ОШИБКА: Не удалось создать тестовые y_true тензоры. Проверьте FPN_LEVELS_CONFIG_FOR_LOSS_MODULE.")
    else:
        print("\nТестирование функции потерь FPN с ИДЕАЛЬНЫМИ предсказаниями (loss должен быть близок к 0):")
        os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] = "1"

        print("\n  --- С Focal Loss для Objectness (Идеальные) ---")
        loss_dict_perfect_focal = compute_detector_loss_v2_fpn(
            y_true_fpn_tuple_test, y_pred_all_levels_list_test_ideal, use_focal_for_obj_param=True
        )
        if isinstance(loss_dict_perfect_focal, dict):
            for k, v in loss_dict_perfect_focal.items(): print(f"    {k}: {v.numpy():.6f}")
        else:
            print(f"    total_loss (идеальные, фокал): {loss_dict_perfect_focal.numpy():.6f}")

        print("\n  --- С BCE для Objectness (Идеальные) ---")
        loss_dict_perfect_bce = compute_detector_loss_v2_fpn(
            y_true_fpn_tuple_test, y_pred_all_levels_list_test_ideal, use_focal_for_obj_param=False
        )
        if isinstance(loss_dict_perfect_bce, dict):
            for k, v in loss_dict_perfect_bce.items(): print(f"    {k}: {v.numpy():.6f}")
        else:
            print(f"    total_loss (идеальные, BCE): {loss_dict_perfect_bce.numpy():.6f}")

        print("\nТестирование функции потерь FPN со СЛУЧАЙНЫМИ предсказаниями:")
        print("\n  --- С Focal Loss для Objectness (Случайные) ---")
        loss_dict_random_focal = compute_detector_loss_v2_fpn(
            y_true_fpn_tuple_test, y_pred_all_levels_list_test_random, use_focal_for_obj_param=True
        )
        if isinstance(loss_dict_random_focal, dict):
            for k, v in loss_dict_random_focal.items(): print(f"    {k}: {v.numpy():.4f}")
        else:
            print(f"    total_loss (случайные, фокал): {loss_dict_random_focal.numpy():.4f}")

        print("\n  --- С BCE для Objectness (Случайные) ---")
        loss_dict_random_bce = compute_detector_loss_v2_fpn(
            y_true_fpn_tuple_test, y_pred_all_levels_list_test_random, use_focal_for_obj_param=False
        )
        if isinstance(loss_dict_random_bce, dict):
            for k, v in loss_dict_random_bce.items(): print(f"    {k}: {v.numpy():.4f}")
        else:
            print(f"    total_loss (случайные, BCE): {loss_dict_random_bce.numpy():.4f}")

    if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ: del os.environ["DEBUG_TRAINING_LOOP_ACTIVE"]
    print("\n--- Тестирование detection_losses.py (FPN) завершено ---")