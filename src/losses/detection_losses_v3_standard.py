import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


# --- Компонент Функции Потерь ---

def focal_loss(y_true_cls, y_pred_cls, alpha=0.25, gamma=1.5):  # Используем gamma=1.5 по нашему решению
    """Вычисляет Focal Loss для задачи классификации."""
    y_pred_cls = tf.nn.sigmoid(y_pred_cls)
    indices = tf.where(tf.reduce_all(y_true_cls != -1.0, axis=-1))

    if tf.shape(indices)[0] == 0:
        return 0.0

    y_true_cls = tf.gather_nd(y_true_cls, indices)
    y_pred_cls = tf.gather_nd(y_pred_cls, indices)

    cross_entropy = tf.keras.backend.binary_crossentropy(y_true_cls, y_pred_cls)
    p_t = y_true_cls * y_pred_cls + (1 - y_true_cls) * (1 - y_pred_cls)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_t = y_true_cls * alpha + (1 - y_true_cls) * (1 - alpha)
    focal_loss_value = alpha_t * modulating_factor * cross_entropy

    positive_mask = tf.reduce_any(y_true_cls > 0, axis=-1)
    num_positives = tf.maximum(1.0, tf.reduce_sum(tf.cast(positive_mask, tf.float32)))

    return tf.reduce_sum(focal_loss_value) / num_positives


# --- Главная функция-обертка ---

class DetectorLoss(tf.keras.losses.Loss):
    def __init__(self, config, all_anchors, name="detector_loss"):
        super().__init__(name=name)
        self.all_anchors_tf = tf.constant(all_anchors, dtype=tf.float32)

        self.num_classes = config['num_classes']
        self.cls_loss_weight = config['loss_weights']['classification']
        self.reg_loss_weight = config['loss_weights']['box_regression']
        self.focal_alpha = config.get('focal_loss_alpha', 0.25)
        self.focal_gamma = config.get('focal_loss_gamma', 1.5)  # Обновленное значение по умолчанию

        self.giou_loss_fn = tfa.losses.GIoULoss()

    def _decode_boxes(self, encoded_preds, anchors):
        """Декодирует предсказания (tx,ty,tw,th) обратно в боксы (y1,x1,y2,x2) для GIoU."""
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_w
        anchor_cy = anchors[:, 1] + 0.5 * anchor_h

        tx, ty, tw, th = tf.unstack(encoded_preds, axis=-1)

        pred_cx = tx * anchor_w + anchor_cx
        pred_cy = ty * anchor_h + anchor_cy
        pred_w = tf.exp(tw) * anchor_w
        pred_h = tf.exp(th) * anchor_h

        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h

        return tf.stack([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1)

    def call(self, y_true, y_pred):
        num_levels = len(y_pred) // 2

        y_true_reg_flat = tf.concat([tf.reshape(t, [tf.shape(t)[0], -1, 4]) for t in y_true[:num_levels]], axis=1)
        y_true_cls_flat = tf.concat(
            [tf.reshape(t, [tf.shape(t)[0], -1, self.num_classes]) for t in y_true[num_levels:]], axis=1)
        y_pred_reg_flat = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in y_pred[:num_levels]], axis=1)
        y_pred_cls_flat = tf.concat(
            [tf.reshape(p, [tf.shape(p)[0], -1, self.num_classes]) for p in y_pred[num_levels:]], axis=1)

        cls_loss = focal_loss(y_true_cls_flat, y_pred_cls_flat, self.focal_alpha, self.focal_gamma)

        positive_mask = tf.reduce_any(y_true_cls_flat > 0, axis=-1)
        indices = tf.where(positive_mask)

        if tf.shape(indices)[0] == 0:
            reg_loss = 0.0
        else:
            y_true_reg_pos = tf.gather_nd(y_true_reg_flat, indices)
            y_pred_reg_pos = tf.gather_nd(y_pred_reg_flat, indices)

            anchor_indices = indices[:, 1]
            anchors_pos = tf.gather(self.all_anchors_tf, anchor_indices)

            decoded_pred_boxes = self._decode_boxes(y_pred_reg_pos, anchors_pos)
            decoded_true_boxes = self._decode_boxes(y_true_reg_pos, anchors_pos)

            giou_loss_value = self.giou_loss_fn(decoded_true_boxes, decoded_pred_boxes)

            num_positives = tf.maximum(1.0, tf.cast(tf.shape(indices)[0], tf.float32))
            reg_loss = giou_loss_value / num_positives

        total_loss = (self.cls_loss_weight * cls_loss) + (self.reg_loss_weight * reg_loss)
        return total_loss


# --- Тестовый блок (упрощенный) ---
if __name__ == '__main__':
    print("--- Тестирование DetectorLoss ---")

    BATCH_SIZE, NUM_CLASSES = 2, 2
    NUM_ANCHORS_P3, NUM_ANCHORS_P4, NUM_ANCHORS_P5 = 100, 50, 25
    TOTAL_ANCHORS = NUM_ANCHORS_P3 + NUM_ANCHORS_P4 + NUM_ANCHORS_P5

    test_config = {
        'num_classes': NUM_CLASSES, 'loss_weights': {'classification': 1.0, 'box_regression': 1.5},
        'focal_loss_gamma': 1.5
    }

    y_true_reg_flat = np.random.randn(BATCH_SIZE, TOTAL_ANCHORS, 4).astype(np.float32)
    y_true_cls_flat = np.zeros((BATCH_SIZE, TOTAL_ANCHORS, NUM_CLASSES), dtype=np.float32)
    y_true_cls_flat[0, 5, 0] = 1.0  # Один позитивный пример

    y_pred_reg_flat = np.random.randn(BATCH_SIZE, TOTAL_ANCHORS, 4).astype(np.float32)
    y_pred_cls_flat = np.random.randn(BATCH_SIZE, TOTAL_ANCHORS, NUM_CLASSES).astype(np.float32)

    split_sizes = [NUM_ANCHORS_P3, NUM_ANCHORS_P4, NUM_ANCHORS_P5]
    y_true_reg_list = tf.split(y_true_reg_flat, split_sizes, axis=1)
    y_true_cls_list = tf.split(y_true_cls_flat, split_sizes, axis=1)
    y_pred_reg_list = tf.split(y_pred_reg_flat, split_sizes, axis=1)
    y_pred_cls_list = tf.split(y_pred_cls_flat, split_sizes, axis=1)

    y_true = tuple(y_true_reg_list + y_true_cls_list)
    y_pred = tuple(y_pred_reg_list + y_pred_cls_list)

    all_anchors_np = np.random.rand(TOTAL_ANCHORS, 4).astype(np.float32)
    all_anchors_np[:, 2:] += all_anchors_np[:, :2]  # Убедимся, что x2>x1, y2>y1

    print("\nТест DetectorLoss с focal_loss и GIoU:")
    try:
        loss_fn = DetectorLoss(test_config, all_anchors=all_anchors_np)
        total_loss = loss_fn(y_true, y_pred)
        print(f"  - Total Weighted Loss (GIoU): {total_loss.numpy():.4f}")
        assert not np.isnan(total_loss.numpy()), "Loss не должен быть NaN"
        print("  - [SUCCESS] Тест пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка при вычислении потерь: {e}")
        import traceback;

        traceback.print_exc()