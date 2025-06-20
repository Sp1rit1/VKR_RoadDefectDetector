# src/losses/detection_losses_single_level_v2.py
import tensorflow as tf
import yaml
import os
import numpy as np
from pathlib import Path

# --- Загрузка Конфигурации ---
_current_script_dir_loss_v2 = Path(__file__).resolve().parent
_project_root_loss_v2 = _current_script_dir_loss_v2.parent.parent
_detector_config_path_loss_v2_primary = _project_root_loss_v2 / 'src' / 'configs' / 'detector_config_single_level_v2.yaml'
_detector_config_path_loss_v2_fallback = _project_root_loss_v2 / 'src' / 'configs' / 'detector_config_single_level_debug.yaml'

DETECTOR_CONFIG_LOSS_V2 = {}
CONFIG_LOAD_SUCCESS_LOSS_V2 = True

_config_to_load_loss_v2 = _detector_config_path_loss_v2_primary
if not _config_to_load_loss_v2.exists():
    _config_to_load_loss_v2 = _detector_config_path_loss_v2_fallback

try:
    with open(_config_to_load_loss_v2, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_LOSS_V2 = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_LOSS_V2, dict) or not DETECTOR_CONFIG_LOSS_V2:
        CONFIG_LOAD_SUCCESS_LOSS_V2 = False;
        DETECTOR_CONFIG_LOSS_V2 = {}
except Exception:
    CONFIG_LOAD_SUCCESS_LOSS_V2 = False;
    DETECTOR_CONFIG_LOSS_V2 = {}

if not CONFIG_LOAD_SUCCESS_LOSS_V2:
    print("ПРЕДУПРЕЖДЕНИЕ (loss_sdl_v2): Используются АВАРИЙНЫЕ ДЕФОЛТЫ для detector_config.")
    _fpn_params_default_loss_v2 = {'num_classes': 2, 'input_shape': [416, 416, 3],
                                   'detector_fpn_strides': {'P4_debug': 16},
                                   'detector_fpn_anchor_configs': {'P4_debug': {'num_anchors_this_level': 7}},
                                   # Обновлен дефолт якорей
                                   'focal_loss_objectness_params': {'use_focal_loss': True, 'alpha': 0.25,
                                                                    'gamma': 2.0}}  # Focal Loss по умолчанию
    DETECTOR_CONFIG_LOSS_V2.setdefault('fpn_detector_params', _fpn_params_default_loss_v2)
    DETECTOR_CONFIG_LOSS_V2.setdefault('loss_weights',
                                       {'coordinates': 1.0, 'objectness': 2.0, 'no_object': 0.7, 'classification': 1.5})

# --- Глобальные Параметры из Конфига v2 ---
_fpn_params_loss_v2 = DETECTOR_CONFIG_LOSS_V2.get('fpn_detector_params', {})
NUM_CLASSES_LOSS_V2 = _fpn_params_loss_v2.get('num_classes', 2)

_input_shape_list_loss_v2 = _fpn_params_loss_v2.get('input_shape', [416, 416, 3])
TARGET_IMG_HEIGHT_LOSS_V2 = _input_shape_list_loss_v2[0]
TARGET_IMG_WIDTH_LOSS_V2 = _input_shape_list_loss_v2[1]

_level_name_loss_v2 = _fpn_params_loss_v2.get('detector_fpn_levels', ['P4_debug'])[0]
_level_stride_loss_v2 = _fpn_params_loss_v2.get('detector_fpn_strides', {}).get(_level_name_loss_v2, 16)
_level_anchor_cfg_loss_v2 = _fpn_params_loss_v2.get('detector_fpn_anchor_configs', {}).get(_level_name_loss_v2, {})
NUM_ANCHORS_LOSS_V2 = _level_anchor_cfg_loss_v2.get('num_anchors_this_level', 7)  # Используем 7 якорей по умолчанию

GRID_H_LOSS_V2 = TARGET_IMG_HEIGHT_LOSS_V2 // _level_stride_loss_v2
GRID_W_LOSS_V2 = TARGET_IMG_WIDTH_LOSS_V2 // _level_stride_loss_v2

_focal_loss_obj_params_loss_v2 = _fpn_params_loss_v2.get('focal_loss_objectness_params', {})
USE_FOCAL_FOR_OBJECTNESS_LOSS_V2 = _focal_loss_obj_params_loss_v2.get('use_focal_loss',
                                                                      True)  # По умолчанию Focal Loss включен
FOCAL_ALPHA_OBJ_LOSS_V2 = _focal_loss_obj_params_loss_v2.get('alpha', 0.25)
FOCAL_GAMMA_OBJ_LOSS_V2 = _focal_loss_obj_params_loss_v2.get('gamma', 2.0)

_loss_weights_cfg_loss_v2 = DETECTOR_CONFIG_LOSS_V2.get('loss_weights', {})
COORD_LOSS_WEIGHT_V2 = _loss_weights_cfg_loss_v2.get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT_V2 = _loss_weights_cfg_loss_v2.get('objectness', 2.0)  # Позитивные
NO_OBJECT_LOSS_WEIGHT_V2 = _loss_weights_cfg_loss_v2.get('no_object', 0.7)  # Негативные (фон)
CLASS_LOSS_WEIGHT_V2 = _loss_weights_cfg_loss_v2.get('classification', 1.5)


# Флаг для детального вывода из функции потерь (управляется извне через os.environ)
# DEBUG_LOSS_DETAILS_ACTIVE = os.environ.get('DEBUG_TRAINING_LOOP_ACTIVE', '0') == '1'
# Заменим на более простой способ передачи аргумента в функцию

# --- Реализация Focal Loss ---
@tf.function
def focal_loss_sigmoid(y_true, y_pred_logits, alpha=0.25, gamma=2.0, from_logits=True):
    """
    Вычисляет Focal Loss для бинарной классификации с логитами на выходе.
    y_true: тензор истинных меток (0 или 1).
    y_pred_logits: тензор логитов (выход модели до sigmoid).
    """
    if from_logits:
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    else:  # Если y_pred уже вероятности после sigmoid
        y_pred = y_pred_logits  # Переименуем для ясности
        # Защита от log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        ce_loss = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))

    p_t = tf.exp(-ce_loss)  # Это эквивалентно y_pred если y_true=1, и 1-y_pred если y_true=0
    alpha_factor = tf.ones_like(y_true) * alpha
    alpha_t = tf.where(tf.equal(y_true, 1.0), alpha_factor, 1.0 - alpha_factor)
    focal_weight = alpha_t * tf.pow((1.0 - p_t), gamma)

    return focal_weight * ce_loss


# --- Основная Функция Потерь ---
@tf.function
def compute_detector_loss_single_level_v2(y_true_single_level, y_pred_single_level, return_details_flag=False):
    """
    Рассчитывает потери для одноуровневого детектора с учетом ignore_mask.
    y_true_single_level[..., 4] может быть: 1.0 (объект), 0.0 (фон), -1.0 (игнорировать).
    """
    shape_y_true = tf.shape(y_true_single_level)
    batch_size_loss = shape_y_true[0]
    # grid_h_loss = shape_y_true[1] # Не используется напрямую, но для информации
    # grid_w_loss = shape_y_true[2]
    # num_anchors_loss = shape_y_true[3]
    num_features_total_loss = shape_y_true[4]

    # Решейпим для удобства обработки всех якорей как одного измерения M = grid_h*grid_w*num_anchors
    y_true_reshaped = tf.reshape(y_true_single_level, [batch_size_loss, -1, num_features_total_loss])
    y_pred_reshaped = tf.reshape(y_pred_single_level, [batch_size_loss, -1, num_features_total_loss])

    # Извлекаем компоненты
    true_boxes_encoded = y_true_reshaped[..., 0:4]  # (B, M, 4) - tx,ty,tw,th
    pred_boxes_encoded_raw = y_pred_reshaped[..., 0:4]  # (B, M, 4) - предсказанные tx,ty,tw,th

    true_objectness_raw = y_true_reshaped[..., 4:5]  # (B, M, 1) - содержит 1.0, 0.0, или -1.0
    pred_objectness_logits = y_pred_reshaped[..., 4:5]  # (B, M, 1) - логиты для objectness

    true_classes_one_hot = y_true_reshaped[..., 5:]  # (B, M, NumClasses)
    pred_classes_logits = y_pred_reshaped[..., 5:]  # (B, M, NumClasses) - логиты для классов

    # --- Создаем маски ---
    # 1. ignore_mask: 1.0 там, где y_true_objectness == -1.0, иначе 0.0
    ignore_mask_bool = tf.equal(true_objectness_raw, -1.0)  # (B, M, 1)
    # 2. object_mask (позитивные якоря): 1.0 там, где y_true_objectness == 1.0, иначе 0.0
    object_mask_bool = tf.equal(true_objectness_raw, 1.0)  # (B, M, 1)
    object_mask_float = tf.cast(object_mask_bool, tf.float32)  # (B, M, 1) для умножения

    # 3. no_object_mask (негативные/фоновые якоря): 1.0 там, где y_true_objectness == 0.0, иначе 0.0
    no_object_mask_bool = tf.equal(true_objectness_raw, 0.0)  # (B, M, 1)

    # Маска для всех НЕИГНОРИРУЕМЫХ якорей (позитивные + негативные)
    # non_ignored_mask_float = tf.cast(tf.logical_not(ignore_mask_bool), tf.float32) # (B, M, 1)
    # Или что то же самое:
    non_ignored_mask_bool = tf.logical_or(object_mask_bool, no_object_mask_bool)
    non_ignored_mask_float = tf.cast(non_ignored_mask_bool, tf.float32)

    # --- 1. Потери для Objectness (для НЕИГНОРИРУЕМЫХ якорей) ---
    # Целевые значения для objectness (только 0 или 1, игнорируемые не участвуют)
    true_objectness_for_loss = tf.cast(object_mask_bool,
                                       tf.float32)  # 1.0 для объектов, 0.0 для фона и игнорируемых (игнорируемые отфильтруются маской)

    if USE_FOCAL_FOR_OBJECTNESS_LOSS_V2:
        obj_loss_per_item_raw = focal_loss_sigmoid(
            y_true=true_objectness_for_loss,
            y_pred_logits=pred_objectness_logits,
            alpha=FOCAL_ALPHA_OBJ_LOSS_V2,
            gamma=FOCAL_GAMMA_OBJ_LOSS_V2
        )  # (B, M, 1)
    else:
        obj_loss_per_item_raw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_objectness_for_loss, logits=pred_objectness_logits
        )  # (B, M, 1)

    # Применяем веса для позитивных и негативных (только к неигнорируемым)
    objectness_sample_weights = tf.where(object_mask_bool, OBJECTNESS_LOSS_WEIGHT_V2, NO_OBJECT_LOSS_WEIGHT_V2)
    weighted_obj_loss_per_item = obj_loss_per_item_raw * objectness_sample_weights * non_ignored_mask_float  # (B, M, 1)

    num_non_ignored_anchors = tf.maximum(tf.reduce_sum(non_ignored_mask_float),
                                         1.0)  # Общее число неигнорируемых якорей
    total_objectness_loss_val = tf.reduce_sum(weighted_obj_loss_per_item) / num_non_ignored_anchors

    # --- 2. Потери для Классов (только для ПОЗИТИВНЫХ якорей) ---
    class_bce_loss_per_item_per_class_val = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot, logits=pred_classes_logits
    )  # (B,M,C)
    # Умножаем на object_mask_float, чтобы потери считались только для позитивных якорей
    # И каждая компонента класса тоже умножается
    masked_class_loss_components = class_bce_loss_per_item_per_class_val * object_mask_float  # (B,M,C) * (B,M,1) -> (B,M,C)
    summed_class_loss_per_anchor = tf.reduce_sum(masked_class_loss_components,
                                                 axis=-1)  # Суммируем потери по классам для каждого якоря -> (B,M)

    num_positive_anchors = tf.maximum(tf.reduce_sum(object_mask_float), 1.0)  # Скаляр
    total_class_loss_val = tf.reduce_sum(summed_class_loss_per_anchor) / num_positive_anchors
    total_class_loss_val = total_class_loss_val * CLASS_LOSS_WEIGHT_V2

    # --- 3. Потери для Координат Bounding Box (только для ПОЗИТИВНЫХ якорей, Huber Loss) ---
    delta_huber = 1.0
    abs_error_coords = tf.abs(true_boxes_encoded - pred_boxes_encoded_raw)  # (B,M,4)
    is_small_error_coords = abs_error_coords <= delta_huber
    squared_loss_coords = 0.5 * tf.square(abs_error_coords)
    linear_loss_coords = delta_huber * (abs_error_coords - 0.5 * delta_huber)
    coord_loss_individual_components_val = tf.where(is_small_error_coords, squared_loss_coords,
                                                    linear_loss_coords)  # (B,M,4)

    # Умножаем на object_mask_float (расширенный до 4х каналов)
    object_mask_for_box_loss = tf.tile(object_mask_float, [1, 1, 4])  # (B,M,4)
    masked_coord_loss_components = coord_loss_individual_components_val * object_mask_for_box_loss  # (B,M,4)
    summed_coord_loss_per_anchor = tf.reduce_sum(masked_coord_loss_components,
                                                 axis=-1)  # Суммируем по 4 координатам -> (B,M)

    total_box_loss_val = tf.reduce_sum(summed_coord_loss_per_anchor) / num_positive_anchors
    total_box_loss_val = total_box_loss_val * COORD_LOSS_WEIGHT_V2

    final_total_loss = total_objectness_loss_val + total_class_loss_val + total_box_loss_val

    if return_details_flag:
        # Для отладки можно выводить средние значения не нормализованных на batch_size потерь
        # или даже суммы, чтобы видеть абсолютный вклад
        return {
            'total_loss': final_total_loss,
            'obj_loss_total': total_objectness_loss_val,
            'cls_loss_total': total_class_loss_val,
            'box_loss_total': total_box_loss_val,
            'num_pos_anchors_avg_per_img': num_positive_anchors / tf.cast(batch_size_loss, tf.float32),
            'num_non_ignored_avg_per_img': num_non_ignored_anchors / tf.cast(batch_size_loss, tf.float32)
        }
    else:
        return final_total_loss


# --- Обертка для model.compile() ---
def wrapped_detector_loss_for_compile_v2(y_true, y_pred):
    return compute_detector_loss_single_level_v2(y_true, y_pred, return_details_flag=True)


if __name__ == '__main__':
    print(f"--- Тестирование detection_losses_single_level_v2.py (с ignore mask и Focal Loss) ---")
    if not CONFIG_LOAD_SUCCESS_LOSS_V2:
        print("\n!!! ВНИМАНИЕ: Отладочный конфигурационный файл не был загружен корректно. "
              "Тестирование может использовать неактуальные или аварийные дефолтные параметры.")

    print(f"\nПараметры, используемые для теста потерь (из конфига или дефолты):")
    print(f"  Focal Loss для Objectness будет использован: {USE_FOCAL_FOR_OBJECTNESS_LOSS_V2}")
    if USE_FOCAL_FOR_OBJECTNESS_LOSS_V2:
        print(f"    Focal Alpha: {FOCAL_ALPHA_OBJ_LOSS_V2}, Focal Gamma: {FOCAL_GAMMA_OBJ_LOSS_V2}")
    print(f"  Веса потерь: COORD={COORD_LOSS_WEIGHT_V2}, OBJ_POS={OBJECTNESS_LOSS_WEIGHT_V2}, "
          f"OBJ_NEG={NO_OBJECT_LOSS_WEIGHT_V2}, CLS={CLASS_LOSS_WEIGHT_V2}")
    print(f"  Количество классов: {NUM_CLASSES_LOSS_V2}")
    print(f"  Параметры для уровня '{_level_name_loss_v2}': "
          f"Сетка({GRID_H_LOSS_V2}x{GRID_W_LOSS_V2}), Якорей={NUM_ANCHORS_LOSS_V2}")

    # --- Создаем тестовые y_true и y_pred ---
    batch_size_test = 2
    # Форма для y_true и y_pred для одного уровня
    y_shape_test = (batch_size_test, GRID_H_LOSS_V2, GRID_W_LOSS_V2, NUM_ANCHORS_LOSS_V2, 5 + NUM_CLASSES_LOSS_V2)

    # --- Пример 1: y_true с позитивными, негативными и игнорируемыми якорями ---
    y_true_np_test1 = np.zeros(y_shape_test, dtype=np.float32)
    # Позитивный якорь 1 (в первом изображении батча)
    y_true_np_test1[0, 1, 1, 0, 0:4] = [0.5, 0.5, np.log(1.0),
                                        np.log(1.0)]  # tx, ty, tw, th (tw,th=0 -> w/h = anchor_w/h)
    y_true_np_test1[0, 1, 1, 0, 4] = 1.0  # Objectness = 1 (позитивный)
    y_true_np_test1[0, 1, 1, 0, 5 + 0] = 1.0  # class 'pit' (ID 0)

    # Игнорируемый якорь 1 (в первом изображении батча)
    y_true_np_test1[0, 2, 2, 1, 4] = -1.0  # Objectness = -1 (игнорировать)
    # Координаты и классы для игнорируемых не важны, но для полноты можно заполнить чем-то
    y_true_np_test1[0, 2, 2, 1, 0:4] = [0.1, 0.1, 0.0, 0.0]
    y_true_np_test1[0, 2, 2, 1, 5 + 1] = 1.0  # Не будет учитываться

    # Позитивный якорь 2 (во втором изображении батча)
    y_true_np_test1[1, 3, 3, 2, 0:4] = [0.2, 0.8, np.log(1.2), np.log(0.8)]
    y_true_np_test1[1, 3, 3, 2, 4] = 1.0  # Objectness = 1
    y_true_np_test1[1, 3, 3, 2, 5 + 1] = 1.0  # class 'crack' (ID 1)

    y_true_tf_test1 = tf.constant(y_true_np_test1, dtype=tf.float32)

    # --- Предсказания (y_pred) ---
    # 1. Идеальные предсказания для позитивных, случайные для остального
    y_pred_logits_ideal_for_pos_np = np.random.randn(*y_shape_test).astype(np.float32) * 0.1  # Небольшой шум для фона

    # Для позитивного якоря 1
    y_pred_logits_ideal_for_pos_np[0, 1, 1, 0, 0:4] = y_true_np_test1[0, 1, 1, 0, 0:4]  # Идеальные координаты
    y_pred_logits_ideal_for_pos_np[0, 1, 1, 0, 4] = 10.0  # Высокий логит для objectness
    pred_cls_obj1 = np.full(NUM_CLASSES_LOSS_V2, -10.0, dtype=np.float32);
    pred_cls_obj1[0] = 10.0  # Идеальный класс 'pit'
    y_pred_logits_ideal_for_pos_np[0, 1, 1, 0, 5:] = pred_cls_obj1

    # Для игнорируемого якоря 1 - предсказание не должно влиять на loss (кроме случая, если бы он был негативным и use_focal_loss=false)
    y_pred_logits_ideal_for_pos_np[0, 2, 2, 1, 4] = 0.0  # Например, модель не уверена
    y_pred_logits_ideal_for_pos_np[0, 2, 2, 1, 5 + 1] = 5.0  # Предсказывает какой-то класс

    # Для позитивного якоря 2
    y_pred_logits_ideal_for_pos_np[1, 3, 3, 2, 0:4] = y_true_np_test1[1, 3, 3, 2, 0:4]
    y_pred_logits_ideal_for_pos_np[1, 3, 3, 2, 4] = 10.0
    pred_cls_obj2 = np.full(NUM_CLASSES_LOSS_V2, -10.0, dtype=np.float32);
    pred_cls_obj2[1] = 10.0
    y_pred_logits_ideal_for_pos_np[1, 3, 3, 2, 5:] = pred_cls_obj2

    # Для всех остальных (фон) - предсказываем низкий objectness
    mask_pos_ign = (y_true_np_test1[..., 4:5] != 0.0)  # Маска, где не фон (т.е. позитивные или игнор)
    y_pred_logits_ideal_for_pos_np[..., 4:5] = np.where(mask_pos_ign, y_pred_logits_ideal_for_pos_np[..., 4:5], -10.0)

    y_pred_tf_test1 = tf.constant(y_pred_logits_ideal_for_pos_np, dtype=tf.float32)

    print("\nТест 1: y_true с позитивными, негативными и ИГНОРИРУЕМЫМИ якорями.")
    print("        Предсказания 'почти идеальные' для позитивных, низкий objectness для фона.")

    # Устанавливаем флаг для вывода деталей
    os.environ['DEBUG_TRAINING_LOOP_ACTIVE'] = '1'  # Чтобы loss_fn вернула детали, если она это использует
    # Наша функция использует return_details_flag

    loss_details_test1 = compute_detector_loss_single_level_v2(y_true_tf_test1, y_pred_tf_test1,
                                                               return_details_flag=True)

    print("  Детальные Потери (Тест 1):")
    if isinstance(loss_details_test1, dict):
        for k_test1, v_tensor_test1 in loss_details_test1.items():
            print(f"    {k_test1}: {v_tensor_test1.numpy():.6f}")
    else:  # Если вернулся только total_loss
        print(f"    total_loss: {loss_details_test1.numpy():.6f}")

    # --- Тест 2: Все предсказания очень плохие (высокие логиты для objectness везде) ---
    y_pred_logits_all_obj_np = np.random.randn(*y_shape_test).astype(np.float32) * 0.1
    y_pred_logits_all_obj_np[..., 4:5] = 5.0  # Высокая уверенность в объекте ВЕЗДЕ
    y_pred_tf_test2 = tf.constant(y_pred_logits_all_obj_np, dtype=np.float32)

    print("\nТест 2: y_true такое же, но предсказания ВЕЗДЕ с высокой уверенностью в объекте.")
    loss_details_test2 = compute_detector_loss_single_level_v2(y_true_tf_test1, y_pred_tf_test2,
                                                               return_details_flag=True)
    print("  Детальные Потери (Тест 2):")
    if isinstance(loss_details_test2, dict):
        for k_test2, v_tensor_test2 in loss_details_test2.items():
            print(f"    {k_test2}: {v_tensor_test2.numpy():.6f}")
    else:
        print(f"    total_loss: {loss_details_test2.numpy():.6f}")

    # Сбрасываем флаг, если он больше не нужен
    # if 'DEBUG_TRAINING_LOOP_ACTIVE' in os.environ:
    #     del os.environ['DEBUG_TRAINING_LOOP_ACTIVE']

    print("\n--- Тестирование detection_losses_single_level_v2.py завершено ---")