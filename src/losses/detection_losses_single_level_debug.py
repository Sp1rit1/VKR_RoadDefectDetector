# src/losses/detection_losses_single_level_debug.py
import tensorflow as tf
import yaml
import os
import numpy as np
from pathlib import Path

# --- Загрузка ОТЛАДОЧНОЙ Конфигурации ---
_current_script_dir_loss_sdl = Path(__file__).resolve().parent  # src/losses/
_project_root_loss_sdl = _current_script_dir_loss_sdl.parent.parent  # Корень проекта
_debug_config_path_loss_sdl = _project_root_loss_sdl / 'src' / 'configs' / 'detector_config_single_level_debug.yaml'

DEBUG_LOSS_MODULE_CONFIG = {}  # Конфиг, загруженный в этом модуле
CONFIG_LOAD_SUCCESS_LOSS_MODULE = True  # Флаг для этого модуля

try:
    with open(_debug_config_path_loss_sdl, 'r', encoding='utf-8') as f:
        DEBUG_LOSS_MODULE_CONFIG = yaml.safe_load(f)
    if not isinstance(DEBUG_LOSS_MODULE_CONFIG, dict) or not DEBUG_LOSS_MODULE_CONFIG:
        print(f"ПРЕДУПРЕЖДЕНИЕ (loss_sdl_module): Файл {_debug_config_path_loss_sdl.name} пуст или неверный формат.")
        CONFIG_LOAD_SUCCESS_LOSS_MODULE = False
except FileNotFoundError:
    print(f"ОШИБКА (loss_sdl_module): Файл {_debug_config_path_loss_sdl.name} не найден: {_debug_config_path_loss_sdl}")
    CONFIG_LOAD_SUCCESS_LOSS_MODULE = False
except yaml.YAMLError as e_cfg_loss_sdl:
    print(f"ОШИБКА YAML (loss_sdl_module): Не удалось прочитать {_debug_config_path_loss_sdl.name}: {e_cfg_loss_sdl}")
    CONFIG_LOAD_SUCCESS_LOSS_MODULE = False

if not CONFIG_LOAD_SUCCESS_LOSS_MODULE:
    print(
        "ПРЕДУПРЕЖДЕНИЕ (loss_sdl_module): Используются АВАРИЙНЫЕ ДЕФОЛТЫ из-за ошибки загрузки detector_config_single_level_debug.yaml.")
    DEBUG_LOSS_MODULE_CONFIG = {
        'fpn_detector_params': {
            'num_classes': 2,
            'input_shape': [416, 416, 3],
            'detector_fpn_strides': {'P4_debug': 16},
            'detector_fpn_anchor_configs': {'P4_debug': {'num_anchors_this_level': 1}},
            'focal_loss_objectness_params': {'use_focal_loss': False, 'alpha': 0.25, 'gamma': 2.0}
        },
        'loss_weights': {'coordinates': 1.0, 'objectness': 1.0, 'no_object': 0.5, 'classification': 1.0}
    }

# --- Глобальные Параметры для этого МОДУЛЯ ПОТЕРЬ ---
_fpn_params_loss_module = DEBUG_LOSS_MODULE_CONFIG.get('fpn_detector_params', {})
NUM_CLASSES_LOSS_MODULE = _fpn_params_loss_module.get('num_classes', 2)

# Параметры для единственного уровня 'P4_debug' (или как он назван в отладочном конфиге)
# Предполагаем, что отладочный конфиг ИМЕЕТ только один уровень в detector_fpn_levels
_debug_level_name_loss_module = _fpn_params_loss_module.get('detector_fpn_levels', ['P4_debug'])[0]
_debug_level_stride_loss_module = _fpn_params_loss_module.get('detector_fpn_strides', {}).get(
    _debug_level_name_loss_module, 16)
_debug_level_anchor_cfg_loss_module = _fpn_params_loss_module.get('detector_fpn_anchor_configs', {}).get(
    _debug_level_name_loss_module, {})

NUM_ANCHORS_DEBUG_LEVEL_LOSS_MODULE = _debug_level_anchor_cfg_loss_module.get('num_anchors_this_level', 1)

# Размеры сетки для отладочного уровня
_input_shape_list_loss_module = _fpn_params_loss_module.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT_LOSS_MODULE = _input_shape_list_loss_module[0]
TARGET_IMG_WIDTH_LOSS_MODULE = _input_shape_list_loss_module[1]
GRID_H_DEBUG_LEVEL_LOSS_MODULE = TARGET_IMG_HEIGHT_LOSS_MODULE // _debug_level_stride_loss_module
GRID_W_DEBUG_LEVEL_LOSS_MODULE = TARGET_IMG_WIDTH_LOSS_MODULE // _debug_level_stride_loss_module

# Параметры Focal Loss для objectness
_focal_loss_obj_params_loss_module = _fpn_params_loss_module.get('focal_loss_objectness_params', {})
USE_FOCAL_FOR_OBJECTNESS_LOSS_MODULE = _focal_loss_obj_params_loss_module.get('use_focal_loss', False)
FOCAL_ALPHA_LOSS_MODULE = _focal_loss_obj_params_loss_module.get('alpha', 0.25)
FOCAL_GAMMA_LOSS_MODULE = _focal_loss_obj_params_loss_module.get('gamma', 2.0)

# Веса потерь
_loss_weights_cfg_loss_module = DEBUG_LOSS_MODULE_CONFIG.get('loss_weights', {})
COORD_LOSS_WEIGHT_LOSS_MODULE = _loss_weights_cfg_loss_module.get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT_LOSS_MODULE = _loss_weights_cfg_loss_module.get('objectness', 1.0)
NO_OBJECT_LOSS_WEIGHT_LOSS_MODULE = _loss_weights_cfg_loss_module.get('no_object', 0.5)
CLASS_LOSS_WEIGHT_LOSS_MODULE = _loss_weights_cfg_loss_module.get('classification', 1.0)

# Флаг для детального вывода (управляется извне через os.environ)
DEBUG_ACTIVE_LOSS_MODULE = True


# @tf.function # Убираем для отладки с tf.print
def compute_detector_loss_single_level_debug(y_true_single_level, y_pred_single_level):
    shape_y_true = tf.shape(y_true_single_level)
    batch_size_loss = shape_y_true[0]
    num_features_total_loss = shape_y_true[4]

    y_true_reshaped = tf.reshape(y_true_single_level, [batch_size_loss, -1, num_features_total_loss])
    y_pred_reshaped = tf.reshape(y_pred_single_level, [batch_size_loss, -1, num_features_total_loss])

    true_boxes_encoded = y_true_reshaped[..., 0:4]
    pred_boxes_encoded_raw = y_pred_reshaped[..., 0:4]
    true_objectness = y_true_reshaped[..., 4:5]
    pred_objectness_logits = y_pred_reshaped[..., 4:5]
    true_classes_one_hot = y_true_reshaped[..., 5:]  # Индекс 5, так как 0-3 coords, 4 objectness
    pred_classes_logits = y_pred_reshaped[..., 5:]

    if USE_FOCAL_FOR_OBJECTNESS_LOSS_MODULE:  # Используем глобальный флаг этого модуля
        obj_loss_per_item = tf.keras.losses.binary_focal_crossentropy(
            y_true=true_objectness, y_pred=pred_objectness_logits,
            apply_class_balancing=True, alpha=FOCAL_ALPHA_LOSS_MODULE,  # Используем глобальные параметры
            gamma=FOCAL_GAMMA_LOSS_MODULE, from_logits=True
        )
        obj_loss_per_item = tf.expand_dims(obj_loss_per_item, axis=-1)
    else:
        obj_loss_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_objectness, logits=pred_objectness_logits
        )

    loss_weights_obj_map = tf.where(tf.equal(true_objectness, 1.0),
                                    OBJECTNESS_LOSS_WEIGHT_LOSS_MODULE,  # Глобальный вес
                                    NO_OBJECT_LOSS_WEIGHT_LOSS_MODULE)  # Глобальный вес

    weighted_objectness_loss_val = obj_loss_per_item * loss_weights_obj_map
    total_objectness_loss_val = tf.reduce_sum(weighted_objectness_loss_val) / tf.cast(
        tf.size(weighted_objectness_loss_val), tf.float32)

    objectness_mask_val = tf.squeeze(tf.cast(tf.equal(true_objectness, 1.0), tf.float32), axis=-1)
    num_responsible_objects_val = tf.maximum(tf.reduce_sum(objectness_mask_val), 1.0)

    class_bce_loss_per_item_per_class_val = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot, logits=pred_classes_logits
    )
    class_loss_per_prediction_val = tf.reduce_sum(class_bce_loss_per_item_per_class_val, axis=-1)
    masked_class_loss_val = class_loss_per_prediction_val * objectness_mask_val
    total_class_loss_val = tf.reduce_sum(masked_class_loss_val) / num_responsible_objects_val
    total_class_loss_val = total_class_loss_val * CLASS_LOSS_WEIGHT_LOSS_MODULE  # Глобальный вес

    delta_huber = 1.0
    abs_error_coords = tf.abs(true_boxes_encoded - pred_boxes_encoded_raw)
    is_small_error_coords = abs_error_coords <= delta_huber
    squared_loss_coords = 0.5 * tf.square(abs_error_coords)
    linear_loss_coords = delta_huber * (abs_error_coords - 0.5 * delta_huber)
    coord_loss_individual_components_val = tf.where(is_small_error_coords, squared_loss_coords, linear_loss_coords)
    coord_loss_per_prediction_val = tf.reduce_sum(coord_loss_individual_components_val, axis=-1)
    masked_coord_loss_val = coord_loss_per_prediction_val * objectness_mask_val
    total_box_loss_val = tf.reduce_sum(masked_coord_loss_val) / num_responsible_objects_val
    total_box_loss_val = total_box_loss_val * COORD_LOSS_WEIGHT_LOSS_MODULE  # Глобальный вес

    final_total_loss = total_objectness_loss_val + total_class_loss_val + total_box_loss_val

    if DEBUG_ACTIVE_LOSS_MODULE:  # Используем флаг этого модуля
        return {
            'total_loss': final_total_loss,
            f'obj_loss_{_debug_level_name_loss_module}': total_objectness_loss_val,  # Используем имя уровня
            f'cls_loss_{_debug_level_name_loss_module}': total_class_loss_val,
            f'box_loss_{_debug_level_name_loss_module}': total_box_loss_val,
            f'num_pos_anchors_{_debug_level_name_loss_module}': tf.reduce_sum(objectness_mask_val) / tf.cast(
                batch_size_loss, tf.float32)
        }
    else:
        return final_total_loss


# --- Блок if __name__ == '__main__': для тестирования этого файла ---
if __name__ == '__main__':
    print(f"--- Тестирование detection_losses_single_level_debug.py ---")
    # Используем CONFIG_LOAD_SUCCESS_LOSS_MODULE, определенный в этом файле
    if not CONFIG_LOAD_SUCCESS_LOSS_MODULE:
        print("\n!!! ВНИМАНИЕ: Отладочный конфигурационный файл не был загружен корректно.")

    # Используем глобальные переменные этого модуля для вывода
    print(f"Focal Loss для Objectness будет использован (в тесте): {USE_FOCAL_FOR_OBJECTNESS_LOSS_MODULE}")
    if USE_FOCAL_FOR_OBJECTNESS_LOSS_MODULE:
        print(f"  Focal Alpha: {FOCAL_ALPHA_LOSS_MODULE}, Focal Gamma: {FOCAL_GAMMA_LOSS_MODULE}")
    print(f"Веса потерь: COORD={COORD_LOSS_WEIGHT_LOSS_MODULE}, OBJ_POS={OBJECTNESS_LOSS_WEIGHT_LOSS_MODULE}, "
          f"OBJ_NEG={NO_OBJECT_LOSS_WEIGHT_LOSS_MODULE}, CLS={CLASS_LOSS_WEIGHT_LOSS_MODULE}")
    print(f"Количество классов: {NUM_CLASSES_LOSS_MODULE}")
    print(f"Параметры для отладочного уровня '{_debug_level_name_loss_module}': "
          f"Сетка({GRID_H_DEBUG_LEVEL_LOSS_MODULE}x{GRID_W_DEBUG_LEVEL_LOSS_MODULE}), Якорей={NUM_ANCHORS_DEBUG_LEVEL_LOSS_MODULE}")

    batch_size_test_loss_main = 2
    # Используем глобальные переменные этого модуля для форм
    y_true_shape_test_main = (batch_size_test_loss_main, GRID_H_DEBUG_LEVEL_LOSS_MODULE, GRID_W_DEBUG_LEVEL_LOSS_MODULE,
                              NUM_ANCHORS_DEBUG_LEVEL_LOSS_MODULE, 5 + NUM_CLASSES_LOSS_MODULE)
    y_pred_shape_test_main = y_true_shape_test_main

    # --- Тест 1: Идеальные предсказания ---
    print("\nТестирование с ИДЕАЛЬНЫМИ предсказаниями (loss должен быть близок к 0):")
    y_true_np_ideal_main = np.zeros(y_true_shape_test_main, dtype=np.float32)
    # Позитивный якорь для первого элемента батча
    y_true_np_ideal_main[0, 0, 0, 0, 0:4] = [0.5, 0.5, np.log(1.0), np.log(1.0)]
    y_true_np_ideal_main[0, 0, 0, 0, 4] = 1.0
    y_true_np_ideal_main[0, 0, 0, 0, 5 + 0] = 1.0  # class 'pit' (ID 0)
    # Позитивный якорь для второго элемента батча
    y_true_np_ideal_main[1, 5, 5, 0, 0:4] = [0.3, 0.7, np.log(1.0), np.log(1.0)]
    y_true_np_ideal_main[1, 5, 5, 0, 4] = 1.0
    y_true_np_ideal_main[1, 5, 5, 0, 5 + 1] = 1.0  # class 'crack' (ID 1)
    y_true_tf_ideal_main = tf.constant(y_true_np_ideal_main, dtype=tf.float32)

    y_pred_logits_ideal_np_main = np.zeros_like(y_true_np_ideal_main, dtype=np.float32)
    y_pred_logits_ideal_np_main[..., 0:4] = y_true_np_ideal_main[..., 0:4]
    y_pred_logits_ideal_np_main[..., 4:5] = np.where(y_true_np_ideal_main[..., 4:5] > 0.5, 10.0, -10.0)
    y_pred_logits_ideal_np_main[..., 5:] = np.where(y_true_np_ideal_main[..., 5:] > 0.5, 10.0, -10.0)
    y_pred_tf_ideal_main = tf.constant(y_pred_logits_ideal_np_main, dtype=np.float32)

    loss_details_ideal_main = compute_detector_loss_single_level_debug(y_true_tf_ideal_main, y_pred_tf_ideal_main)
    print("  Детальные Потери (Идеальные):")
    if isinstance(loss_details_ideal_main, dict): # <<<--- ПРОВЕРКА ТИПА
        for k_main, v_tensor_main in loss_details_ideal_main.items():
            print(f"    {k_main}: {v_tensor_main.numpy():.8f}")
    else: # Если вернулся только total_loss
        print(f"    total_loss: {loss_details_ideal_main.numpy():.8f}")
        print("    (Для детальных потерь установите переменную окружения DEBUG_TRAINING_LOOP_ACTIVE='1' перед запуском)")

    # --- Тест 2: Случайные предсказания ---
    print("\nТестирование со СЛУЧАЙНЫМИ предсказаниями:")
    y_pred_logits_random_np_main = np.random.randn(*y_pred_shape_test_main).astype(np.float32) * 0.5
    y_pred_tf_random_main = tf.constant(y_pred_logits_random_np_main,
                                        dtype=tf.float32)  # <<<--- УБЕДИСЬ, ЧТО ЭТА СТРОКА ЕСТЬ И ИМЯ ПЕРЕМЕННОЙ ПРАВИЛЬНОЕ

    loss_details_random_main = compute_detector_loss_single_level_debug(y_true_tf_ideal_main,
                                                                        y_pred_tf_random_main)  # Теперь y_pred_tf_random_main определена
    print("  Детальные Потери (Случайные):")
    if isinstance(loss_details_random_main, dict): # <<<--- ПРОВЕРКА ТИПА
        for k_main, v_tensor_main in loss_details_random_main.items():
            print(f"    {k_main}: {v_tensor_main.numpy():.6f}")
    else:
        print(f"    total_loss: {loss_details_random_main.numpy():.6f}")
        print("    (Для детальных потерь установите переменную окружения DEBUG_TRAINING_LOOP_ACTIVE='1' перед запуском)")

    print("\n--- Тестирование detection_losses_single_level_debug.py завершено ---")