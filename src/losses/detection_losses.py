# src/losses/detection_losses.py
import tensorflow as tf
import yaml
import os
import numpy as np

# --- Загрузка Конфигурации ---
# ... (код загрузки конфигов как был)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}
except Exception as e:
    print(f"ОШИБКА: Не удалось загрузить detector_config.yaml в detection_losses.py: {e}")
    DETECTOR_CONFIG = {'num_classes': 2, 'input_shape': [416, 416, 3],
                       'anchors_wh_normalized': [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]],
                       'num_anchors_per_location': 3,
                       'max_boxes_per_image': 10}

NUM_CLASSES_LOSS = DETECTOR_CONFIG.get('num_classes', 2)
_input_shape_list_loss_test = DETECTOR_CONFIG.get('input_shape', [416, 416, 3])
_target_img_height_loss_test = _input_shape_list_loss_test[0]
_target_img_width_loss_test = _input_shape_list_loss_test[1]
_network_stride_loss_test = 16
GRID_HEIGHT_LOSS_TEST = _target_img_height_loss_test // _network_stride_loss_test
GRID_WIDTH_LOSS_TEST = _target_img_width_loss_test // _network_stride_loss_test
_anchors_wh_normalized_list_loss_test = DETECTOR_CONFIG.get('anchors_wh_normalized',
                                                            [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]])
_anchors_wh_normalized_loss_test = np.array(_anchors_wh_normalized_list_loss_test, dtype=np.float32)
NUM_ANCHORS_PER_LOCATION_LOSS_TEST = DETECTOR_CONFIG.get('num_anchors_per_location',
                                                         _anchors_wh_normalized_loss_test.shape[0])

COORD_LOSS_WEIGHT = DETECTOR_CONFIG.get('loss_weights', {}).get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT = DETECTOR_CONFIG.get('loss_weights', {}).get('objectness', 1.0)
NO_OBJECT_LOSS_WEIGHT = DETECTOR_CONFIG.get('loss_weights', {}).get('no_object', 0.5)
CLASS_LOSS_WEIGHT = DETECTOR_CONFIG.get('loss_weights', {}).get('classification', 1.0)


# binary_crossentropy_logits не нужен глобально, будем использовать tf.nn.sigmoid_cross_entropy_with_logits
# huber_loss_fn не нужен глобально, будем реализовывать вручную
# --- Конец Загрузки Конфигурации ---


@tf.function
def compute_detector_loss_v1(y_true, y_pred):
    shape_y_true = tf.shape(y_true)
    batch_size = shape_y_true[0]
    grid_h = shape_y_true[1]  # Не используется напрямую в этой версии, но для информации
    grid_w = shape_y_true[2]
    num_anchors = shape_y_true[3]
    num_features_total = shape_y_true[4]

    y_true_reshaped = tf.reshape(y_true, [batch_size, -1, num_features_total])
    y_pred_reshaped = tf.reshape(y_pred, [batch_size, -1, num_features_total])

    true_boxes_encoded = y_true_reshaped[..., :4]  # (B, M, 4) - это tx,ty,tw,th
    pred_boxes_encoded_raw = y_pred_reshaped[..., :4]  # (B, M, 4) - это pred_tx, pred_ty, pred_tw, pred_th

    true_objectness = y_true_reshaped[..., 4:5]  # (B, M, 1)
    pred_objectness_logits = y_pred_reshaped[..., 4:5]  # (B, M, 1)

    true_classes_one_hot = y_true_reshaped[..., 5:]  # (B, M, C)
    pred_classes_logits = y_pred_reshaped[..., 5:]  # (B, M, C)

    # --- 1. Потери для Objectness ---
    objectness_bce_loss_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_objectness, logits=pred_objectness_logits
    )  # (B, M, 1)

    loss_weights_obj = tf.where(tf.equal(true_objectness, 1.0),
                                OBJECTNESS_LOSS_WEIGHT,
                                NO_OBJECT_LOSS_WEIGHT)  # (B, M, 1)

    weighted_objectness_loss = objectness_bce_loss_per_item * loss_weights_obj
    # Усредняем по всем M * B элементам (так как objectness loss считается для всех якорей)
    total_objectness_loss = tf.reduce_sum(weighted_objectness_loss) / tf.cast(tf.size(weighted_objectness_loss),
                                                                              tf.float32)

    # Маска для объектов (где true_objectness == 1.0)
    objectness_mask = tf.squeeze(tf.cast(true_objectness, tf.float32), axis=-1)  # (B, M)
    num_responsible_objects = tf.maximum(tf.reduce_sum(objectness_mask), 1.0)  # Скаляр

    # --- 2. Потери для Классов ---
    class_bce_loss_per_item_per_class = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot, logits=pred_classes_logits
    )  # (B,M,C)
    class_loss_per_prediction = tf.reduce_sum(class_bce_loss_per_item_per_class, axis=-1)  # (B,M)
    masked_class_loss = class_loss_per_prediction * objectness_mask
    total_class_loss = tf.reduce_sum(masked_class_loss) / num_responsible_objects
    total_class_loss = total_class_loss * CLASS_LOSS_WEIGHT

    # --- 3. Потери для Координат Bounding Box (РУЧНОЙ HUBER) ---
    delta_huber = 1.0
    abs_error_coords = tf.abs(true_boxes_encoded - pred_boxes_encoded_raw)  # (B,M,4)
    is_small_error_coords = abs_error_coords <= delta_huber
    squared_loss_coords = 0.5 * tf.square(abs_error_coords)
    linear_loss_coords = delta_huber * (abs_error_coords - 0.5 * delta_huber)
    coord_loss_individual_components = tf.where(is_small_error_coords, squared_loss_coords,
                                                linear_loss_coords)  # (B,M,4)

    coord_loss_per_prediction = tf.reduce_sum(coord_loss_individual_components, axis=-1)  # (B,M)

    masked_coord_loss = coord_loss_per_prediction * objectness_mask  # (B,M) * (B,M)
    total_box_loss = tf.reduce_sum(masked_coord_loss) / num_responsible_objects
    total_box_loss = total_box_loss * COORD_LOSS_WEIGHT

    final_loss = total_objectness_loss + total_class_loss + total_box_loss
    return final_loss


simple_detector_loss = compute_detector_loss_v1

if __name__ == '__main__':
    # КОД БЛОКА if __name__ == '__main__' ОСТАЕТСЯ ТОЧНО ТАКИМ ЖЕ, КАК В МОЕМ ПОЛНОМ ОТВЕТЕ от 01:06
    print(f"--- Тестирование detection_losses.py (для сетки и якорей) ---")
    # ... (остальной тестовый код)
    print(f"NUM_CLASSES_LOSS: {NUM_CLASSES_LOSS}")
    print(
        f"GRID_H_TEST={GRID_HEIGHT_LOSS_TEST}, GRID_W_TEST={GRID_WIDTH_LOSS_TEST}, NUM_ANCHORS_TEST={NUM_ANCHORS_PER_LOCATION_LOSS_TEST}")

    batch_size_test = 2
    y_true_shape = (batch_size_test, GRID_HEIGHT_LOSS_TEST, GRID_WIDTH_LOSS_TEST, NUM_ANCHORS_PER_LOCATION_LOSS_TEST,
                    5 + NUM_CLASSES_LOSS)
    y_pred_shape = y_true_shape

    y_true_np = np.zeros(y_true_shape, dtype=np.float32)
    y_true_np[0, 0, 0, 0, 0:4] = [0.5, 0.5, np.log(0.2 / 0.1 + 1e-9), np.log(0.3 / 0.05 + 1e-9)]
    y_true_np[0, 0, 0, 0, 4] = 1.0
    y_true_np[0, 0, 0, 0, 5 + 0] = 1.0
    y_true_np[1, 5, 5, 1, 0:4] = [0.3, 0.7, np.log(0.1 / 0.05 + 1e-9), np.log(0.4 / 0.1 + 1e-9)]
    y_true_np[1, 5, 5, 1, 4] = 1.0
    y_true_np[1, 5, 5, 1, 5 + 1] = 1.0
    y_true_tf = tf.constant(y_true_np, dtype=np.float32)

    y_pred_logits_np = np.random.randn(*y_pred_shape).astype(np.float32) * 0.1
    y_pred_logits_np[0, 0, 0, 0, 0:4] = y_true_np[0, 0, 0, 0, 0:4] + np.random.randn(4) * 0.01
    y_pred_logits_np[0, 0, 0, 0, 4] = 5.0
    pred_cls_obj1 = np.full(NUM_CLASSES_LOSS, -5.0, dtype=np.float32);
    pred_cls_obj1[0] = 5.0
    y_pred_logits_np[0, 0, 0, 0, 5:] = pred_cls_obj1
    y_pred_tf = tf.constant(y_pred_logits_np, dtype=np.float32)

    print("\nТестирование функции потерь...")
    total_loss = compute_detector_loss_v1(y_true_tf, y_pred_tf)
    print(f"  Общая потеря (случайные предсказания): {total_loss.numpy():.4f}")

    y_pred_perfect_logits_np = np.zeros_like(y_true_np, dtype=np.float32)
    y_pred_perfect_logits_np[..., 0:4] = y_true_np[..., 0:4]
    y_pred_perfect_logits_np[..., 4:5] = np.where(y_true_np[..., 4:5] > 0.5, 10.0, -10.0)
    true_cls_one_hot_perfect = y_true_np[..., 5:]
    pred_cls_logits_perfect = np.where(true_cls_one_hot_perfect > 0.5, 10.0, -10.0)
    y_pred_perfect_logits_np[..., 5:] = pred_cls_logits_perfect
    y_pred_perfect_tf = tf.constant(y_pred_perfect_logits_np, dtype=np.float32)

    loss_perfect = compute_detector_loss_v1(y_true_tf, y_pred_perfect_tf)
    print(f"  Общая потеря (для 'идеальных' логитов): {loss_perfect.numpy():.4f} (должна быть очень близка к 0)")

    print("\n--- Тестирование detection_losses.py завершено ---")