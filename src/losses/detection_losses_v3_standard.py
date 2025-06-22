import tensorflow as tf
import numpy as np


# --- Компоненты Функции Потерь ---

def focal_loss(y_true_cls, y_pred_cls, alpha=0.25, gamma=2.0):
    """
    Вычисляет Focal Loss для задачи классификации.

    Args:
        y_true_cls (tf.Tensor): Эталонные метки классов, форма (batch, num_anchors, num_classes).
                                Содержит one-hot векторы для позитивных/негативных якорей
                                и вектор из -1 для игнорируемых.
        y_pred_cls (tf.Tensor): Предсказанные логиты модели, форма (batch, num_anchors, num_classes).
        alpha (float): Балансирующий параметр alpha.
        gamma (float): Параметр фокусировки gamma.

    Returns:
        tf.Tensor: Скалярное значение потерь.
    """
    # 1. Применяем sigmoid к логитам, чтобы получить вероятности
    y_pred_cls = tf.nn.sigmoid(y_pred_cls)

    # 2. Создаем маску, чтобы исключить игнорируемые якоря (где y_true_cls == -1)
    indices = tf.where(tf.reduce_all(y_true_cls != -1.0, axis=-1))
    y_true_cls = tf.gather_nd(y_true_cls, indices)
    y_pred_cls = tf.gather_nd(y_pred_cls, indices)

    # 3. Вычисляем cross-entropy
    cross_entropy = tf.keras.backend.binary_crossentropy(y_true_cls, y_pred_cls)

    # 4. Вычисляем modulating factor (1 - p_t)^gamma
    p_t = y_true_cls * y_pred_cls + (1 - y_true_cls) * (1 - y_pred_cls)
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    # 5. Вычисляем alpha_t
    alpha_t = y_true_cls * alpha + (1 - y_true_cls) * (1 - alpha)

    # 6. Собираем Focal Loss
    focal_loss_value = alpha_t * modulating_factor * cross_entropy

    # 7. Нормализуем на количество позитивных якорей
    positive_mask = tf.reduce_any(y_true_cls > 0, axis=-1)
    num_positives = tf.reduce_sum(tf.cast(positive_mask, tf.float32))
    num_positives = tf.maximum(1.0, num_positives)

    return tf.reduce_sum(focal_loss_value) / num_positives


def huber_loss(y_true_reg, y_pred_reg, y_true_cls, delta=1.0):
    """
    Вычисляет Huber Loss (Smooth L1) для регрессии рамок.

    Args:
        y_true_reg (tf.Tensor): Эталонные смещения (tx, ty, tw, th), форма (batch, num_anchors, 4).
        y_pred_reg (tf.Tensor): Предсказанные смещения, форма (batch, num_anchors, 4).
        y_true_cls (tf.Tensor): Эталонные метки классов для определения позитивных якорей.
        delta (float): Порог для Huber loss.

    Returns:
        tf.Tensor: Скалярное значение потерь.
    """
    # 1. Находим только позитивные якоря
    positive_mask = tf.reduce_any(y_true_cls > 0, axis=-1)
    indices = tf.where(positive_mask)

    y_true_reg_pos = tf.gather_nd(y_true_reg, indices)
    y_pred_reg_pos = tf.gather_nd(y_pred_reg, indices)

    # 2. Вычисляем Huber loss
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.SUM)
    huber_loss_value = huber(y_true_reg_pos, y_pred_reg_pos)

    # 3. Нормализуем на количество позитивных якорей
    num_positives = tf.maximum(1.0, tf.cast(tf.shape(indices)[0], tf.float32))

    return huber_loss_value / num_positives


# --- Главная функция-обертка ---

class DetectorLoss(tf.keras.losses.Loss):
    def __init__(self, config, name="detector_loss"):
        super().__init__(name=name)
        self.num_classes = config['num_classes']
        self.cls_loss_weight = config['loss_weights']['classification']
        self.reg_loss_weight = config['loss_weights']['box_regression']
        self.focal_alpha = config.get('focal_loss_alpha', 0.25)
        self.focal_gamma = config.get('focal_loss_gamma', 2.0)
        self.huber_delta = config.get('huber_loss_delta', 1.0)

    def call(self, y_true, y_pred):
        """
        Главная функция вычисления потерь.
        Теперь и y_true, и y_pred - это списки из 6 тензоров в одинаковом порядке:
        [reg_p3, reg_p4, reg_p5, cls_p3, cls_p4, cls_p5]
        """
        # Определяем, где в списке заканчивается регрессия и начинается классификация
        num_levels = len(y_pred) // 2

        # --- Собираем "плоские" тензоры из y_true ---
        # Склеиваем все тензоры регрессии (первые `num_levels` штук) в один большой
        y_true_reg_flat = tf.concat(
            [tf.reshape(t, [tf.shape(t)[0], -1, 4]) for t in y_true[:num_levels]],
            axis=1
        )
        # Склеиваем все тензоры классификации (оставшиеся) в один большой
        y_true_cls_flat = tf.concat(
            [tf.reshape(t, [tf.shape(t)[0], -1, self.num_classes]) for t in y_true[num_levels:]],
            axis=1
        )

        # --- Собираем "плоские" тензоры из y_pred ---
        # Делаем то же самое для предсказаний модели
        y_pred_reg_flat = tf.concat(
            [tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in y_pred[:num_levels]],
            axis=1
        )
        y_pred_cls_flat = tf.concat(
            [tf.reshape(p, [tf.shape(p)[0], -1, self.num_classes]) for p in y_pred[num_levels:]],
            axis=1
        )

        # --- Вызываем старые функции потерь с новыми "плоскими" тензорами ---
        # Теперь функции focal_loss и huber_loss получают данные в том формате, который они ожидают
        cls_loss = focal_loss(y_true_cls_flat, y_pred_cls_flat, self.focal_alpha, self.focal_gamma)
        reg_loss = huber_loss(y_true_reg_flat, y_pred_reg_flat, y_true_cls_flat, self.huber_delta)

        # --- Считаем итоговый взвешенный loss ---
        total_loss = (self.cls_loss_weight * cls_loss) + (self.reg_loss_weight * reg_loss)

        return total_loss


# --- Тестовый блок ---
if __name__ == '__main__':
    print("--- Тестирование функций потерь ---")

    # --- Параметры для теста ---
    BATCH_SIZE = 2
    NUM_ANCHORS = 100
    NUM_CLASSES = 2

    test_config = {
        'num_classes': NUM_CLASSES,
        'loss_weights': {'classification': 1.0, 'box_regression': 1.5},
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'huber_loss_delta': 1.0,
    }

    # --- Создание фиктивных данных ---
    y_true_cls = np.zeros((BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES), dtype=np.float32)
    y_true_cls[0, 5, 0] = 1.0
    y_true_cls[0, 10, 1] = 1.0
    y_true_cls[0, 15, :] = -1.0

    y_true_reg = np.zeros((BATCH_SIZE, NUM_ANCHORS, 4), dtype=np.float32)
    y_true_reg[0, 5, :] = [0.1, 0.2, 0.3, 0.4]
    y_true_reg[0, 10, :] = [-0.1, -0.2, -0.3, -0.4]

    y_pred_cls = np.random.randn(BATCH_SIZE, NUM_ANCHORS, NUM_CLASSES).astype(np.float32)
    y_pred_cls[0, 5, 0] = 0.6
    y_pred_cls[0, 5, 1] = -0.4
    y_pred_cls[0, 10, 1] = 0.8
    y_pred_cls[0, 10, 0] = -0.9

    y_pred_reg = np.random.randn(BATCH_SIZE, NUM_ANCHORS, 4).astype(np.float32)
    y_pred_reg[0, 5, :] = [0.11, 0.19, 0.33, 0.38]

    y_true = (tf.constant(y_true_reg), tf.constant(y_true_cls))
    y_pred = (tf.constant(y_pred_reg), tf.constant(y_pred_cls))

    print("\n1. Тест DetectorLoss:")
    try:
        loss_fn = DetectorLoss(test_config)
        total_loss = loss_fn(y_true, y_pred)

        cls_loss_val = focal_loss(y_true[1], y_pred[1], test_config['focal_loss_alpha'],
                                  test_config['focal_loss_gamma'])
        reg_loss_val = huber_loss(y_true[0], y_pred[0], y_true[1], test_config['huber_loss_delta'])

        print(f"  - Classification Loss (Focal): {cls_loss_val.numpy():.4f}")
        print(f"  - Regression Loss (Huber):   {reg_loss_val.numpy():.4f}")
        print(f"  - Total Weighted Loss:         {total_loss.numpy():.4f}")

        expected_total = test_config['loss_weights']['classification'] * cls_loss_val + \
                         test_config['loss_weights']['box_regression'] * reg_loss_val

        print(f"  - Ожидаемый Total Loss:        {expected_total.numpy():.4f}")

        assert total_loss > 0, "Loss должен быть положительным"
        assert np.isclose(total_loss.numpy(), expected_total.numpy()), "Суммарный loss посчитан неверно"
        print("  - [SUCCESS] Тест пройден.")

    except Exception as e:
        print(f"  - [ERROR] Ошибка при вычислении потерь: {e}")
        import traceback

        traceback.print_exc()