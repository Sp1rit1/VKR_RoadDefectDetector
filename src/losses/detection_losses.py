# src/losses/detection_losses.py
import tensorflow as tf
import yaml
import os
import numpy as np
import sys

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"ОШИБКА: Не удалось загрузить detector_config.yaml в detection_losses.py: {e}")
    DETECTOR_CONFIG = {'num_classes': 2, 'max_boxes_per_image': 10}

NUM_CLASSES_LOSSES = DETECTOR_CONFIG.get('num_classes', 2)
MAX_BOXES_PER_IMAGE_LOSSES = DETECTOR_CONFIG.get('max_boxes_per_image', 10)

COORD_LOSS_WEIGHT = 1.0
OBJ_LOSS_WEIGHT = 1.0
CLASS_LOSS_WEIGHT = 1.0


# Внутренняя реализация с Python-логикой
def _simple_detector_loss_impl(y_true, y_pred):
    # tf.print("--- _simple_detector_loss_impl ---", output_stream=sys.stdout)
    # tf.print("y_true shape:", tf.shape(y_true), output_stream=sys.stdout)
    # tf.print("y_pred shape:", tf.shape(y_pred), output_stream=sys.stdout)

    true_boxes = y_true[..., :4]
    pred_boxes_raw = y_pred[..., :4]
    pred_boxes_sigmoid = tf.sigmoid(pred_boxes_raw)

    true_objectness = y_true[..., 4:5]
    pred_objectness_logits = y_pred[..., 4:5]

    true_classes_one_hot = y_true[..., 5:]
    pred_classes_logits = y_pred[..., 5:]

    # --- Потери для Objectness ---
    objectness_loss_elementwise = tf.keras.losses.binary_crossentropy(
        y_true=true_objectness, y_pred=pred_objectness_logits, from_logits=True
    )
    total_objectness_loss = tf.reduce_mean(objectness_loss_elementwise)

    # --- Потери для Классов ---
    class_loss_values_per_class_elementwise = tf.keras.losses.binary_crossentropy(
        y_true=true_classes_one_hot, y_pred=pred_classes_logits, from_logits=True
    )

    # Мы ожидаем, что class_loss_values_per_class_elementwise будет (B,M,C)
    # и после суммирования по классам станет (B,M)
    # Если поведение BCE другое, tf.rank поможет это отловить
    rank_class_loss = tf.rank(class_loss_values_per_class_elementwise)
    # tf.print("Rank of class_loss_values_per_class_elementwise:", rank_class_loss, output_stream=sys.stdout)

    if rank_class_loss == 3:  # (B,M,C)
        class_loss_per_box_summed = tf.reduce_sum(class_loss_values_per_class_elementwise, axis=-1)  # -> (B,M)
    elif rank_class_loss == 2:  # Уже (B,M)
        class_loss_per_box_summed = class_loss_values_per_class_elementwise
    else:  # Неожиданно
        tf.print(
            "ERROR (loss_fn): Unexpected rank for class_loss_values_per_class_elementwise! Assuming it's (B,M). Shape:",
            tf.shape(class_loss_values_per_class_elementwise), output_stream=sys.stdout)
        class_loss_per_box_summed = class_loss_values_per_class_elementwise  # Попытка продолжить
    # tf.print("Shape of class_loss_per_box_summed (after rank check):", tf.shape(class_loss_per_box_summed), output_stream=sys.stdout)

    objectness_mask = tf.squeeze(tf.cast(true_objectness, tf.float32), axis=-1)
    masked_class_loss_per_box = class_loss_per_box_summed * objectness_mask

    num_true_objects = tf.maximum(tf.reduce_sum(objectness_mask), 1.0)
    total_class_loss = tf.reduce_sum(masked_class_loss_per_box) / num_true_objects

    # --- Потери для Координат Bounding Box ---
    huber_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    coord_loss_individual_coords = huber_loss_fn(true_boxes, pred_boxes_sigmoid)  # Ожидаем (B,M,4)
    # tf.print("Shape of coord_loss_individual_coords (from Huber):", tf.shape(coord_loss_individual_coords), output_stream=sys.stdout)

    rank_coord_loss = tf.rank(coord_loss_individual_coords)
    # tf.print("Rank of coord_loss_individual_coords:", rank_coord_loss, output_stream=sys.stdout)

    if rank_coord_loss == 3:  # Ожидаем (B,M,4)
        coord_loss_per_box_summed = tf.reduce_sum(coord_loss_individual_coords, axis=-1)  # -> (B,M)
    elif rank_coord_loss == 2:  # Если уже (B,M)
        coord_loss_per_box_summed = coord_loss_individual_coords
    else:
        tf.print("ERROR (loss_fn): Unexpected rank for coord_loss_individual_coords! Assuming it's (B,M). Shape:",
                 tf.shape(coord_loss_individual_coords), output_stream=sys.stdout)
        coord_loss_per_box_summed = coord_loss_individual_coords
    # tf.print("Shape of coord_loss_per_box_summed (after rank check for coords):", tf.shape(coord_loss_per_box_summed), output_stream=sys.stdout)

    masked_coord_loss_per_box = coord_loss_per_box_summed * objectness_mask
    total_coord_loss = tf.reduce_sum(masked_coord_loss_per_box) / num_true_objects

    final_loss = (COORD_LOSS_WEIGHT * total_coord_loss +
                  OBJ_LOSS_WEIGHT * total_objectness_loss +
                  CLASS_LOSS_WEIGHT * total_class_loss)
    # tf.print("Final Loss (in impl):", final_loss, output_stream=sys.stdout)
    return final_loss


# Обертка с @tf.function, которую будем ИМПОРТИРОВАТЬ и использовать в model.compile()
@tf.function
def simple_detector_loss(y_true, y_pred):
    return _simple_detector_loss_impl(y_true, y_pred)


if __name__ == '__main__':
    # ... (блок if __name__ == '__main__' остается таким же, как в твоей рабочей версии,
    # он не изменился и успешно проходил) ...
    print("--- Тестирование detection_losses.py ---")
    batch_s = 2
    max_b = MAX_BOXES_PER_IMAGE_LOSSES
    num_c = NUM_CLASSES_LOSSES
    num_feat_box = 4 + 1 + num_c
    print(f"  Тестовые параметры: batch_size={batch_s}, max_boxes={max_b}, num_classes={num_c}")
    y_true_np = np.zeros((batch_s, max_b, num_feat_box), dtype=np.float32)
    y_true_np[0, 0, :4] = [0.1, 0.1, 0.3, 0.3];
    y_true_np[0, 0, 4] = 1.0;
    y_true_np[0, 0, 5 + 0] = 1.0
    y_true_np[1, 0, :4] = [0.5, 0.5, 0.8, 0.8];
    y_true_np[1, 0, 4] = 1.0;
    y_true_np[1, 0, 5 + 1] = 1.0
    y_true_tf = tf.constant(y_true_np, dtype=np.float32)
    y_pred_np = np.random.randn(batch_s, max_b, num_feat_box).astype(np.float32) * 0.1
    coords_0_true = y_true_np[0, 0, :4];
    coords_0_true_clipped = np.clip(coords_0_true, 1e-7, 1.0 - 1e-7)
    coords_0_logit = np.log(coords_0_true_clipped / (1.0 - coords_0_true_clipped))
    y_pred_np[0, 0, :4] = coords_0_logit + np.random.randn(4) * 0.01
    y_pred_np[0, 0, 4] = 2.0;
    y_pred_np[0, 0, 5 + 0] = 3.0;
    y_pred_np[0, 0, 5 + 1] = -1.0
    coords_1_true = y_true_np[1, 0, :4];
    coords_1_true_clipped = np.clip(coords_1_true, 1e-7, 1.0 - 1e-7)
    coords_1_logit = np.log(coords_1_true_clipped / (1.0 - coords_1_true_clipped))
    y_pred_np[1, 0, :4] = coords_1_logit + np.random.randn(4) * 0.01
    y_pred_np[1, 0, 4] = 1.5;
    y_pred_np[1, 0, 5 + 0] = -2.0;
    y_pred_np[1, 0, 5 + 1] = 2.5
    y_pred_tf = tf.constant(y_pred_np, dtype=np.float32)
    print("\nПример y_true[0,0]:", y_true_tf[0, 0, :].numpy())
    pred_boxes_example_00 = tf.sigmoid(y_pred_tf[0, 0, :4]).numpy()
    pred_obj_example_00 = tf.sigmoid(y_pred_tf[0, 0, 4:5]).numpy()
    pred_cls_example_00 = tf.sigmoid(y_pred_tf[0, 0, 5:]).numpy()
    print(
        f"Пример y_pred[0,0] (после sigmoid): box={pred_boxes_example_00}, obj={pred_obj_example_00}, cls={pred_cls_example_00}")
    loss = simple_detector_loss(y_true_tf, y_pred_tf)
    print(f"\nРассчитанная потеря: {loss.numpy()}")
    y_pred_perfect_logits_np = np.full_like(y_true_np, -10.0, dtype=np.float32)
    obj_mask_true_np = y_true_np[..., 4] > 0.5
    coords_true_for_logit = np.clip(y_true_np[..., :4], 1e-7, 1.0 - 1e-7)
    logit_coords = tf.math.log(coords_true_for_logit / (1.0 - coords_true_for_logit)).numpy()
    for b_idx in range(batch_s):
        for box_idx in range(max_b):
            if obj_mask_true_np[b_idx, box_idx]:
                y_pred_perfect_logits_np[b_idx, box_idx, :4] = logit_coords[b_idx, box_idx]
    y_pred_perfect_logits_np[..., 4:5] = np.where(y_true_np[..., 4:5] > 0.5, 10.0, -10.0)
    classes_true_for_logit = y_true_np[..., 5:]
    for b_idx in range(batch_s):
        for box_idx in range(max_b):
            if obj_mask_true_np[b_idx, box_idx]:
                y_pred_perfect_logits_np[b_idx, box_idx, 5:] = np.where(classes_true_for_logit[b_idx, box_idx] > 0.5,
                                                                        10.0, -10.0)
    y_pred_perfect_tf = tf.constant(y_pred_perfect_logits_np, dtype=np.float32)
    loss_perfect = simple_detector_loss(y_true_tf, y_pred_perfect_tf)
    print(f"Потеря при 'идеальных' логитах: {loss_perfect.numpy()}")