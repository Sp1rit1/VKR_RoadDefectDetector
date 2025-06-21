# src/losses/detection_losses_single_level_v2.py
import sys

import tensorflow as tf
import yaml
import os
import numpy as np

# --- Настройка sys.path (если нужно для прямого запуска, но лучше чтобы конфиги читались относительно) ---
_current_script_dir_loss_v2 = os.path.dirname(os.path.abspath(__file__))
_project_root_loss_v2 = os.path.abspath(os.path.join(_current_script_dir_loss_v2, '..', '..'))

# --- Загрузка Конфигурации ---
_detector_config_path_loss_v2_primary = os.path.join(_project_root_loss_v2, 'src', 'configs',
                                                     'detector_config_single_level_v2.yaml')
_detector_config_path_loss_v2_fallback = os.path.join(_project_root_loss_v2, 'src', 'configs',
                                                      'detector_config_single_level_debug.yaml')

DETECTOR_CONFIG_LOSS_V2 = {}
_config_to_load_loss_v2 = _detector_config_path_loss_v2_primary
if not os.path.exists(_config_to_load_loss_v2):
    _config_to_load_loss_v2 = _detector_config_path_loss_v2_fallback
    if os.path.exists(_config_to_load_loss_v2):
        print(f"INFO (losses_v2): Используется fallback конфиг: {_config_to_load_loss_v2.name}")
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ (losses_v2): Основной и fallback конфиги не найдены.")

if os.path.exists(_config_to_load_loss_v2):
    try:
        with open(_config_to_load_loss_v2, 'r', encoding='utf-8') as f:
            DETECTOR_CONFIG_LOSS_V2 = yaml.safe_load(f)
        if not isinstance(DETECTOR_CONFIG_LOSS_V2, dict) or not DETECTOR_CONFIG_LOSS_V2:
            DETECTOR_CONFIG_LOSS_V2 = {}
            print(f"ПРЕДУПРЕЖДЕНИЕ (losses_v2): Конфиг {_config_to_load_loss_v2.name} пуст или некорректен.")
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML (losses_v2) в {_config_to_load_loss_v2.name}: {e}.")
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ (losses_v2): Файл конфигурации {_config_to_load_loss_v2.name} не найден.")

# --- Параметры из Конфига с дефолтами ---
_fpn_params_loss_v2 = DETECTOR_CONFIG_LOSS_V2.get('fpn_detector_params', {})
NUM_CLASSES_LOSS_V2 = len(_fpn_params_loss_v2.get('classes', ['class0', 'class1']))  # Берем длину списка классов

_loss_weights_cfg_v2 = DETECTOR_CONFIG_LOSS_V2.get('loss_weights', {})
COORD_LOSS_WEIGHT_V2 = float(_loss_weights_cfg_v2.get('coordinates', 1.0))
OBJECTNESS_LOSS_WEIGHT_POS_V2 = float(_loss_weights_cfg_v2.get('objectness', 1.0))
NO_OBJECT_LOSS_WEIGHT_NEG_V2 = float(_loss_weights_cfg_v2.get('no_object', 0.5))
CLASS_LOSS_WEIGHT_V2 = float(_loss_weights_cfg_v2.get('classification', 1.0))

_focal_loss_obj_params_cfg_v2 = DETECTOR_CONFIG_LOSS_V2.get('focal_loss_objectness_params', {})
USE_FOCAL_LOSS_FOR_OBJECTNESS_V2 = bool(_focal_loss_obj_params_cfg_v2.get('use_focal_loss', False))
FOCAL_LOSS_ALPHA_OBJ_V2 = float(_focal_loss_obj_params_cfg_v2.get('alpha', 0.25))
FOCAL_LOSS_GAMMA_OBJ_V2 = float(_focal_loss_obj_params_cfg_v2.get('gamma', 2.0))

if not DETECTOR_CONFIG_LOSS_V2:  # Если конфиг вообще не загрузился, ставим самые базовые дефолты
    print("ПРЕДУПРЕЖДЕНИЕ (losses_v2): Используются АВАРИЙНЫЕ ДЕФОЛТЫ для всех параметров потерь.")
    NUM_CLASSES_LOSS_V2 = 2
    COORD_LOSS_WEIGHT_V2, OBJECTNESS_LOSS_WEIGHT_POS_V2, NO_OBJECT_LOSS_WEIGHT_NEG_V2, CLASS_LOSS_WEIGHT_V2 = 1.0, 1.0, 0.5, 1.0
    USE_FOCAL_LOSS_FOR_OBJECTNESS_V2 = False;
    FOCAL_LOSS_ALPHA_OBJ_V2 = 0.25;
    FOCAL_LOSS_GAMMA_OBJ_V2 = 2.0


def huber_loss_manual_fn(y_true, y_pred, delta=1.0):
    """Ручная реализация Huber loss, возвращающая поэлементные потери."""
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * tf.square(quadratic) + delta * linear


@tf.function
def compute_detector_loss_single_level_v2(y_true, y_pred):
    """
    Рассчитывает общую потерю для одноуровневого детектора (Версия 2).
    y_true: (batch, grid_h, grid_w, num_anchors, 5 + num_classes)
            y_true[..., 4] (Objectness): 1.0 (позитивный), 0.0 (фон), -1.0 (игнорировать)
    y_pred: (batch, grid_h, grid_w, num_anchors, 5 + num_classes) - сырые логиты от модели
    """
    shape_y_true = tf.shape(y_true)
    batch_size_f = tf.cast(shape_y_true[0], tf.float32)

    y_true_flat = tf.reshape(y_true, [shape_y_true[0], -1, shape_y_true[4]])
    y_pred_flat = tf.reshape(y_pred, [shape_y_true[0], -1, shape_y_true[4]])

    true_boxes_encoded = y_true_flat[..., 0:4]
    pred_boxes_encoded_raw = y_pred_flat[..., 0:4]
    true_objectness_labels = y_true_flat[..., 4:5]
    pred_objectness_logits = y_pred_flat[..., 4:5]
    true_classes_one_hot = y_true_flat[..., 5:]
    pred_classes_logits = y_pred_flat[..., 5:]

    # --- Маски ---
    positive_mask = tf.cast(tf.equal(true_objectness_labels, 1.0), dtype=tf.float32)
    num_positives_float = tf.maximum(tf.reduce_sum(positive_mask), 1.0)  # Скаляр, защита от деления на ноль

    not_ignore_mask = tf.cast(tf.not_equal(true_objectness_labels, -1.0), dtype=tf.float32)
    num_not_ignored_float = tf.maximum(tf.reduce_sum(not_ignore_mask), 1.0)

    # --- 1. Потери для Координат (Box Loss) ---
    box_loss_per_coord_component = huber_loss_manual_fn(true_boxes_encoded, pred_boxes_encoded_raw, delta=1.0)
    box_loss_per_anchor = tf.reduce_sum(box_loss_per_coord_component, axis=-1, keepdims=True)
    masked_sum_box_loss = tf.reduce_sum(box_loss_per_anchor * positive_mask)
    total_box_loss = (masked_sum_box_loss / num_positives_float) * COORD_LOSS_WEIGHT_V2

    # --- 2. Потери для Классификации (Class Loss) ---
    class_bce_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot, logits=pred_classes_logits
    )
    class_loss_per_anchor = tf.reduce_sum(class_bce_per_item, axis=-1, keepdims=True)
    masked_sum_class_loss = tf.reduce_sum(class_loss_per_anchor * positive_mask)
    total_class_loss = (masked_sum_class_loss / num_positives_float) * CLASS_LOSS_WEIGHT_V2

    # --- 3. Потери для Objectness ---
    true_objectness_binary = tf.where(tf.equal(true_objectness_labels, 1.0), 1.0, 0.0)

    if USE_FOCAL_LOSS_FOR_OBJECTNESS_V2:
        bce_obj_raw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_objectness_binary, logits=pred_objectness_logits
        )
        pred_probs_obj = tf.sigmoid(pred_objectness_logits)
        p_t_obj = tf.where(tf.equal(true_objectness_binary, 1.0), pred_probs_obj, 1.0 - pred_probs_obj)
        alpha_factor_obj = tf.where(tf.equal(true_objectness_binary, 1.0), FOCAL_LOSS_ALPHA_OBJ_V2,
                                    1.0 - FOCAL_LOSS_ALPHA_OBJ_V2)
        gamma_factor_obj = tf.pow(1.0 - p_t_obj + 1e-6, FOCAL_LOSS_GAMMA_OBJ_V2)  # Добавлен epsilon для стабильности
        objectness_loss_per_item = alpha_factor_obj * gamma_factor_obj * bce_obj_raw
        # При Focal Loss веса POS/NEG обычно не применяются дополнительно, либо alpha уже их роль выполняет
        weighted_objectness_loss = objectness_loss_per_item * not_ignore_mask
    else:
        objectness_bce_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_objectness_binary, logits=pred_objectness_logits
        )
        obj_loss_weights_map = tf.where(tf.equal(true_objectness_labels, 1.0),
                                        OBJECTNESS_LOSS_WEIGHT_POS_V2,
                                        NO_OBJECT_LOSS_WEIGHT_NEG_V2)
        weighted_objectness_loss = objectness_bce_per_item * obj_loss_weights_map * not_ignore_mask

    total_objectness_loss = tf.reduce_sum(weighted_objectness_loss) / num_not_ignored_float

    # --- Общая Потеря ---
    final_total_loss = total_box_loss + total_class_loss + total_objectness_loss

    # Отладочный вывод (проверяем переменную окружения)
    # Чтобы это работало при запуске из другого скрипта, переменная должна быть установлена до вызова model.compile()
    # или перед началом model.fit()
    if os.environ.get('DEBUG_DETECTOR_LOSS_V2') == '1':
        tf.print("\n--- Loss Components (batch avg) ---", output_stream=sys.stdout)
        tf.print("Box Loss:", total_box_loss, output_stream=sys.stdout)
        tf.print("Class Loss:", total_class_loss, output_stream=sys.stdout)
        tf.print("Objectness Loss:", total_objectness_loss, output_stream=sys.stdout)
        tf.print("Num Positives (avg per image):", tf.reduce_sum(positive_mask) / batch_size_f,
                 output_stream=sys.stdout)
        tf.print("Num Not Ignored (avg per image):", tf.reduce_sum(not_ignore_mask) / batch_size_f,
                 output_stream=sys.stdout)
        tf.print("Total Loss:", final_total_loss, output_stream=sys.stdout)
        return {  # Возвращаем словарь, чтобы Keras мог логировать компоненты, если они добавлены как метрики
            'total_loss_debug': final_total_loss,  # Keras и так логирует 'loss'
            'box_loss_debug': total_box_loss,
            'class_loss_debug': total_class_loss,
            'objectness_loss_debug': total_objectness_loss
        }  # Примечание: чтобы эти имена появились в логах history, их нужно будет добавить в model.compile(metrics=...)
        # или использовать кастомный train_step. Для простого вывода в консоль это не обязательно.

    return final_total_loss


# --- Блок if __name__ == '__main__': для тестирования ---
if __name__ == '__main__':
    print(f"--- Тестирование detection_losses_single_level_v2.py ---")
    print(f"  Используется Focal Loss для Objectness: {USE_FOCAL_LOSS_FOR_OBJECTNESS_V2}")
    if USE_FOCAL_LOSS_FOR_OBJECTNESS_V2:
        print(f"  Focal Loss Alpha: {FOCAL_LOSS_ALPHA_OBJ_V2}, Gamma: {FOCAL_LOSS_GAMMA_OBJ_V2}")
    print(
        f"  Веса потерь (если не Focal): Coords={COORD_LOSS_WEIGHT_V2}, ObjPos={OBJECTNESS_LOSS_WEIGHT_POS_V2}, ObjNeg={NO_OBJECT_LOSS_WEIGHT_NEG_V2}, Class={CLASS_LOSS_WEIGHT_V2}")

    # Загружаем параметры сетки из DETECTOR_CONFIG_LOSS_V2, так же как в data_loader
    _test_fpn_params_l = DETECTOR_CONFIG_LOSS_V2.get('fpn_detector_params', {})
    _test_level_name_l = _test_fpn_params_l.get('detector_fpn_levels', ['P4_debug'])[0]
    _test_stride_l = _test_fpn_params_l.get('detector_fpn_strides', {}).get(_test_level_name_l, 16)
    _test_anchor_cfg_yaml_l = _test_fpn_params_l.get('detector_fpn_anchor_configs', {}).get(_test_level_name_l, {})
    _test_input_shape_l = tuple(_test_fpn_params_l.get('input_shape', [416, 416, 3]))

    grid_h_test = _test_input_shape_l[0] // _test_stride_l
    grid_w_test = _test_input_shape_l[1] // _test_stride_l
    num_anchors_test = _test_anchor_cfg_yaml_l.get('num_anchors_this_level', 3)
    num_classes_test = NUM_CLASSES_LOSS_V2

    batch_size_test = 2
    y_true_shape = (batch_size_test, grid_h_test, grid_w_test, num_anchors_test, 5 + num_classes_test)
    print(f"  Тестовая форма y_true/y_pred: {y_true_shape}")

    if grid_h_test == 0 or grid_w_test == 0 or num_anchors_test == 0:
        print("ОШИБКА: Некорректные размеры сетки или количество якорей для теста. Проверьте конфиг.")
        exit()

    # --- Тест 1: Один позитивный, один фон, один игнорируемый ---
    print("\nТест 1: Позитивный, негативный, игнорируемый якоря")
    y_true_np = np.zeros(y_true_shape, dtype=np.float32)
    y_true_np[..., 4] = -1.0  # Все сначала ignore

    # Пример 1 в батче
    y_true_np[0, 5, 5, 0, 0:4] = [0.1, 0.2, 0.05, -0.05]  # tx, ty, tw, th (tw,th могут быть <0 после log)
    y_true_np[0, 5, 5, 0, 4] = 1.0  # objectness = 1 (позитивный)
    y_true_np[0, 5, 5, 0, 5 + 0] = 1.0  # class 0

    y_true_np[0, 5, 6, 0, 4] = 0.0  # objectness = 0 (фон)

    # y_true_np[0, 5, 7, 0, 4] остается -1.0 (игнорируемый)

    # Пример 2 в батче - весь фон
    y_true_np[1, ..., 4] = 0.0

    y_true_tf = tf.constant(y_true_np, dtype=tf.float32)

    y_pred_logits_np = np.random.randn(*y_true_shape).astype(np.float32) * 0.5
    # Почти идеальные для позитивного
    y_pred_logits_np[0, 5, 5, 0, 0:4] = y_true_np[0, 5, 5, 0, 0:4]
    y_pred_logits_np[0, 5, 5, 0, 4] = 10.0  # objectness logit (очень уверен в объекте)
    _pred_cls_perfect_t1 = np.full(num_classes_test, -10.0);
    _pred_cls_perfect_t1[0] = 10.0
    y_pred_logits_np[0, 5, 5, 0, 5:] = _pred_cls_perfect_t1
    # Плохие для фона (предсказываем объект)
    y_pred_logits_np[0, 5, 6, 0, 4] = 10.0
    # Средние для игнора
    y_pred_logits_np[0, 5, 7, 0, 4] = 0.0
    # Для второго примера (весь фон), предскажем везде "нет объекта"
    y_pred_logits_np[1, ..., 4] = -10.0

    y_pred_tf = tf.constant(y_pred_logits_np, dtype=tf.float32)

    print("\n  --- Результаты Теста 1 (детальный вывод потерь включен): ---")
    os.environ['DEBUG_DETECTOR_LOSS_V2'] = '1'
    loss_values_test1 = compute_detector_loss_single_level_v2(y_true_tf, y_pred_tf)
    os.environ['DEBUG_DETECTOR_LOSS_V2'] = '0'

    if isinstance(loss_values_test1, dict):
        for name, val in loss_values_test1.items():
            print(f"    {name}: {val.numpy():.4f}")
    else:  # Если вернулся только total_loss
        print(f"    total_loss: {loss_values_test1.numpy():.4f}")

    # --- Тест 2: Идеальные предсказания для Теста 1 ---
    print("\nТест 2: Идеальные предсказания для данных Теста 1")
    y_pred_perfect_np = np.zeros_like(y_true_np)
    # Координаты
    y_pred_perfect_np[..., 0:4] = y_true_np[..., 0:4]
    # Objectness (логиты)
    y_pred_perfect_np[..., 4:5] = tf.where(tf.equal(y_true_np[..., 4:5], 1.0), 10.0,
                                           -10.0).numpy()  # Для фона и игнора - низкие логиты
    # Классы (логиты)
    y_pred_perfect_np[..., 5:] = tf.where(y_true_np[..., 5:] > 0.5, 10.0, -10.0).numpy()

    y_pred_perfect_tf = tf.constant(y_pred_perfect_np, dtype=tf.float32)

    os.environ['DEBUG_DETECTOR_LOSS_V2'] = '1'
    loss_values_test2 = compute_detector_loss_single_level_v2(y_true_tf, y_pred_perfect_tf)
    os.environ['DEBUG_DETECTOR_LOSS_V2'] = '0'

    print("  --- Результаты Теста 2 (идеальные предсказания для данных Теста 1): ---")
    if isinstance(loss_values_test2, dict):
        for name, val in loss_values_test2.items():
            print(f"    {name}: {val.numpy():.6f}")  # Больше знаков после запятой
    else:
        print(f"    total_loss: {loss_values_test2.numpy():.6f}")
    # Ожидаем все потери очень близкими к нулю.

    print("\n--- Тестирование detection_losses_single_level_v2.py завершено ---")