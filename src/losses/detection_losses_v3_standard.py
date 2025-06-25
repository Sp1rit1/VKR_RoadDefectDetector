# RoadDefectDetector/src/losses/detection_losses_v3_standard.py

import numpy as np
import tensorflow as tf
# Используем TensorFlow Addons для GIoULoss.
# Убедитесь, что tensorflow-addons установлен (pip install tensorflow-addons).
# pip install tensorflow-addons
import tensorflow_addons as tfa
import math
import logging
import yaml
from pathlib import Path
import sys

# --- Настройка логирования ---
logger = logging.getLogger(__name__)
# Установим уровень только если он не был установлен ранее (для избежания дублирования)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Импорт вспомогательных функций (для тестового блока) ---
# Эти функции импортируются только для использования в if __name__ == '__main__':
# В основном коде функции потерь они не используются напрямую.
# Попытка импортировать generate_all_anchors для тестового блока
# Убедимся, что корень проекта в sys.path
try:
    # Определяем корень проекта
    _current_file_path = Path(__file__).resolve()
    _project_root_path = _current_file_path
    # Идем вверх, пока не найдем папку src или .git (или что-то указывающее на корень)
    while not (_project_root_path / 'src').exists() and _project_root_path != _project_root_path.parent:
        _project_root_path = _project_root_path.parent

    if str(_project_root_path) not in sys.path:
        sys.path.insert(0, str(_project_root_path))
        _added_to_sys_path = True
    else:
        _added_to_sys_path = False

    # Теперь импортируем
    try:
        from src.datasets.data_loader_v3_standard import generate_all_anchors
        logger.debug("generate_all_anchors успешно импортирована для тестового блока.")
    except ImportError:
        logger.warning("Не удалось импортировать generate_all_anchors из src. Тестовый блок может использовать фиктивные данные.")
        generate_all_anchors = None # Установим в None, если импорт не удался

    # Удаляем добавленный путь из sys.path, если мы его добавили
    if _added_to_sys_path:
         sys.path.pop(0)

except Exception as e:
    logger.error(f"Неизвестная ошибка при импорте для тестового блока: {e}")
    generate_all_anchors = None


# --- Компонент Функции Потерь ---

def focal_loss(y_true_cls_flat, y_pred_cls_flat, alpha=0.25, gamma=1.0): # Установлено значение по умолчанию 1.0 для ясности
    """
    Вычисляет Focal Loss для задачи классификации.
    Принимает плоские тензоры. Нормализует на количество позитивных якорей.

    Args:
        y_true_cls_flat (tf.Tensor): Эталонные метки классов, форма (batch, total_anchors, num_classes).
                                     Содержит one-hot векторы для позитивных (1.0), отрицательных (0.0)
                                     и вектор из -1.0 для игнорируемых якорей.
        y_pred_cls_flat (tf.Tensor): Предсказанные логиты модели, форма (batch, total_anchors, num_classes).
        alpha (float): Балансирующий параметр alpha.
        gamma (float): Параметр фокусировки gamma.

    Returns:
        tf.Tensor: Скалярное значение потерь.
    """
    # y_true_cls_flat имеет метки: 1.0 для положительных, 0.0 для отрицательных, -1.0 для игнорируемых.
    # Для Focal Loss нам нужны только неигнорируемые якоря.
    # Маска для неигнорируемых якорей: где НЕ ВСЕ элементы в последней оси равны -1.0.
    non_ignored_mask = tf.reduce_all(y_true_cls_flat != -1.0, axis=-1) # Shape: (batch, total_anchors)

    # Применяем маску для получения тензоров только с неигнорируемыми якорями
    y_true_cls_filtered = tf.boolean_mask(y_true_cls_flat, non_ignored_mask) # Shape: (NumNonIgnoredInBatchTotal, num_classes)
    y_pred_cls_filtered = tf.boolean_mask(y_pred_cls_flat, non_ignored_mask) # Shape: (NumNonIgnoredInBatchTotal, num_classes)


    if tf.shape(y_true_cls_filtered)[0] == 0:
        return 0.0  # Нет неигнорируемых якорей в этом батче, потери 0.

    # Применяем sigmoid к логитам для получения вероятностей (p)
    y_pred_prob_filtered = tf.nn.sigmoid(y_pred_cls_filtered)

    # Вычисляем кросс-энтропию (binary_crossentropy ожидает метки {0, 1})
    # Наши метки для неигнорируемых {0, 1}, что подходит.
    # binary_crossentropy возвращает тензор формы (NumNonIgnoredInBatchTotal, num_classes)
    cross_entropy = tf.keras.backend.binary_crossentropy(y_true_cls_filtered, y_pred_prob_filtered)

    # Вычисляем p_t (вероятность правильного класса)
    # p_t имеет форму (NumNonIgnoredInBatchTotal, num_classes)
    p_t = y_true_cls_filtered * y_pred_prob_filtered + (1.0 - y_true_cls_filtered) * (1.0 - y_pred_prob_filtered)

    # Вычисляем modulating factor (1 - p_t)^gamma
    # modulating_factor имеет форму (NumNonIgnoredInBatchTotal, num_classes)
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    # Вычисляем alpha_t
    # alpha_t имеет форму (NumNonIgnoredInBatchTotal, num_classes)
    alpha_t = y_true_cls_filtered * alpha + (1.0 - y_true_cls_filtered) * (1.0 - alpha)

    # Собираем Focal Loss для каждого элемента (якоря * класс)
    # focal_loss_per_element имеет форму (NumNonIgnoredInBatchTotal, num_classes)
    focal_loss_per_element = alpha_t * modulating_factor * cross_entropy

    # Суммируем Focal Loss по всем неигнорируемым якорям и всем классам
    focal_loss_sum = tf.reduce_sum(focal_loss_per_element)

    # Нормализация: по количеству ПОЗИТИВНЫХ якорей в батче.
    # Позитивные якоря среди неигнорируемых - это те, где хоть один элемент в one-hot векторе > 0.
    # positive_mask_filtered имеет форму (NumNonIgnoredInBatchTotal,)
    positive_mask_filtered = tf.reduce_any(y_true_cls_filtered > 0.0, axis=-1) # Убедимся, что сравниваем с float 0.0
    num_positives_raw = tf.reduce_sum(tf.cast(positive_mask_filtered, tf.float32))

    # Защита от деления на ноль: если нет позитивных якорей, нормализация = 1.0.
    num_positives_for_norm = tf.maximum(1.0, num_positives_raw)

    # Используем tf.math.divide_no_nan для безопасного деления
    normalized_focal_loss = tf.math.divide_no_nan(focal_loss_sum, num_positives_for_norm)

    return normalized_focal_loss


# --- Вспомогательная функция для декодирования боксов (TensorFlow) ---

def _decode_boxes(encoded_preds, anchors):
    """
    Декодирует закодированные предсказания/цели (tx,ty,tw,th) в координаты боксов (y1,x1,y2,x2).
    Работает с тензорами.

    Args:
        encoded_preds (tf.Tensor): Закодированные предсказания/цели (..., 4).
        anchors (tf.Tensor): Соответствующие якоря (..., 4) в формате [x1, y1, x2, y2] нормализованные.
                             Должны иметь ту же батч-размерность и размерность якорей, что и encoded_preds.

    Returns:
        tf.Tensor: Декодированные координаты боксов (..., 4) в формате [y1, x1, y2, x2] нормализованные.
    """
    # Вычисляем ширину, высоту и центры якорей
    # anchors форма (..., 4)
    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = tf.unstack(anchors, axis=-1)
    anchor_w = anchor_x2 - anchor_x1
    anchor_h = anchor_y2 - anchor_y1
    anchor_cx = anchor_x1 + 0.5 * anchor_w
    anchor_cy = anchor_y1 + 0.5 * anchor_h

    # Извлекаем закодированные смещения
    # encoded_preds форма (..., 4)
    tx, ty, tw, th = tf.unstack(encoded_preds, axis=-1)


    # === [ИСПРАВЛЕНО] Добавляем обратное масштабирование смещений ===
    # Делим на те же коэффициенты, на которые умножали в encode_box_targets
    tx /= 0.1
    ty /= 0.1
    tw /= 0.2
    th /= 0.2


    # Декодируем центры и размеры
    # Добавляем маленький эпсилон для стабильности (избежать деления на ноль, хотя anchor_w/h не должны быть нулями)
    epsilon = 1e-7
    anchor_w = tf.maximum(anchor_w, epsilon)
    anchor_h = tf.maximum(anchor_h, epsilon)

    pred_cx = tx * anchor_w + anchor_cx
    pred_cy = ty * anchor_h + anchor_cy

    # === [ВАЖНО] Клиппинг предсказанных tw и th перед экспонентой ===
    # Это предотвращает появление бесконечно больших размеров и NaN.
    # Значение клиппинга должно соответствовать значению клиппинга в encode_box_targets.
    # log(1000/16) ≈ 4.135. Используем это значение.
    bbox_xform_clip = tf.constant(math.log(1000. / 16.), dtype=tf.float32)

    tw = tf.clip_by_value(tw, -bbox_xform_clip, bbox_xform_clip)
    th = tf.clip_by_value(th, -bbox_xform_clip, bbox_xform_clip)

    pred_w = tf.exp(tw) * anchor_w
    pred_h = tf.exp(th) * anchor_h

    # Вычисляем координаты углов декодированных боксов
    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h

    # Собираем декодированные боксы в формате [y1, x1, y2, x2] (требуется NMS и GIoULoss)
    # Убедимся, что координаты в пределах [0, 1] на всякий случай
    decoded_boxes = tf.stack([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1) # Исправлено: формат [y1, x1, y2, x2]
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0.0, 1.0) # Клиппинг к границам изображения

    return decoded_boxes


# --- Главная функция-обертка ---

class DetectorLoss(tf.keras.losses.Loss):
    # ... (__init__ остается без изменений) ...
    def __init__(self, config, all_anchors, name="detector_loss"):
        """
        Инициализирует функцию потерь детектора.
        ... (остается без изменений) ...
        """
        super().__init__(name=name)
        # Проверка, что config - это словарь
        if not isinstance(config, dict):
             raise TypeError(f"DetectorLoss: Argument 'config' must be a dictionary, but got {type(config)}")

        # Преобразуем якоря в TensorFlow константу, если они не Tensor
        # all_anchors должен быть без батч-размерности
        if isinstance(all_anchors, tf.Tensor):
             if len(tf.shape(all_anchors)) > 2: # Убираем батч-размерность, если она случайно есть
                 all_anchors = tf.squeeze(all_anchors, axis=0)
             self.all_anchors_tf = tf.constant(all_anchors, dtype=tf.float32) # Уже константа
        else: # Предполагаем numpy.ndarray
             self.all_anchors_tf = tf.constant(all_anchors, dtype=tf.float32)


        # Получаем параметры из словаря config
        self.num_classes = config['num_classes']
        self.cls_loss_weight = config['loss_weights']['classification']
        self.reg_loss_weight = config['loss_weights']['box_regression']
        self.focal_alpha = config.get('focal_loss_alpha', 0.25)
        self.focal_gamma = config.get('focal_loss_gamma', 1.25) # Используем 1.25 как рекомендовано в аналитике

        # Выбираем Box Loss по конфигу
        self.box_loss_type = config.get('box_loss_type', 'ciou').lower()
        if self.box_loss_type == 'ciou' or self.box_loss_type == 'giou': # Добавим поддержку 'giou' явно
             # GIoULoss в tfa принимает reduction=NONE для получения потерь на каждый бокс
             # GIoULoss работает с форматом [y1, x1, y2, x2]
             # Для GIoULoss.call, первый аргумент - y_true (decoded), второй - y_pred (decoded)
             self.box_loss_fn_per_box = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
             logger.info("Используется Box Loss: GIoU Loss")
        elif self.box_loss_type == 'huber':
             self.huber_delta = config.get('huber_loss_delta', 1.0)
             # HuberLoss в tf.keras.losses также принимает reduction=NONE
             # HuberLoss работает с форматом (tx, ty, tw, th), как и encode_box_targets
             # Поэтому для Huber Loss не нужно декодировать боксы.
             self.box_loss_fn_per_box = tf.keras.losses.Huber(delta=self.huber_delta, reduction=tf.keras.losses.Reduction.NONE)
             logger.info(f"Используется Box Loss: Huber Loss (delta={self.huber_delta})")
        else:
            # Неизвестный тип лосса
            raise ValueError(f"Неизвестный тип Box Loss: {self.box_loss_type}. Поддерживаются 'ciou' и 'huber'.")




    # [ИЗМЕНЕНИЕ] Метод call - теперь принимает плоские y_true
    def call(self, y_true, y_pred):
        """
        Вычисляет суммарную потерю (Classification + Box Regression).

        Args:
            y_true (tuple): Кортеж из двух плоских тензоров y_true:
                           (y_true_reg_flat, y_true_cls_flat).
                           Shapes: (batch, total_anchors, 4) и (batch, total_anchors, num_classes).
                           Эти тензоры уже включают батч-размерность после Dataset.batch().
            y_pred (list): Список тензоров-предсказаний от модели по уровням FPN.
                           Ожидаемый порядок: [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5].
                           Shapes: (batch, H_level, W_level, num_anchors_per_level, tasks).
                           Эти тензоры уже включают батч-размерность.
        """
        # --- [ВАЖНО] Проверка на NaN/Inf в предсказаниях (в режиме графа) ---
        # check_numerics вызовет ошибку во время выполнения графа, если найдет NaN/Inf.
        # Это поможет быстро определить, когда предсказания "взрываются".
        # Проверяем каждый выходной тензор модели
        for i, pred_tensor in enumerate(y_pred):
            tf.debugging.check_numerics(pred_tensor, message=f'DetectorLoss: NaN/Inf in y_pred tensor {i}')

        # 1. Получаем плоские тензоры y_true из входного кортежа
        # Эти тензоры уже батчированы
        y_true_reg_flat, y_true_cls_flat = y_true  # Shapes: (batch, total_anchors, ...)

        # 2. Разбираем список y_pred на регрессию и классификацию и "расплющиваем" их
        # Ожидаемый порядок y_pred: [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5] (чередование)
        # [ИСПРАВЛЕНО] Используем срезы 0::2 и 1::2 для правильного разделения
        num_levels = len(y_pred) // 2  # Количество уровней FPN (должно быть 3)
        reg_list_by_level = y_pred[0::2]  # Берем элементы с четными индексами (0, 2, 4) -> reg P3, reg P4, reg P5
        cls_list_by_level = y_pred[1::2]  # Берем элементы с нечетными индексами (1, 3, 5) -> cls P3, cls P4, cls P5

        # Используем tf.concat для получения плоских предсказаний
        # Каждый тензор уровня (p) уже имеет батч-размерность в y_pred.
        # tf.reshape(p, [tf.shape(p)[0], -1, tasks]) расплющивает spatial и якоря в одну размерность после батча.
        y_pred_reg_flat = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in reg_list_by_level], axis=1)
        y_pred_cls_flat = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, self.num_classes]) for p in cls_list_by_level],
                                    axis=1)

        # === [ВАЖНО] Проверка на соответствие форм y_true и y_pred (в режиме графа) ===
        # Теперь эти assert'ы ДОЛЖНЫ пройти, потому что y_true_flat имеет TotalAnchors из Data Loader,
        # а y_pred_flat после правильной конкатенации тензоров уровня тоже будет иметь TotalAnchors.
        tf.debugging.assert_equal(tf.shape(y_true_reg_flat), tf.shape(y_pred_reg_flat),
                                  message="DetectorLoss: Формы плоских тензоров регрессии y_true и y_pred не совпадают!")
        tf.debugging.assert_equal(tf.shape(y_true_cls_flat), tf.shape(y_pred_cls_flat),
                                  message="DetectorLoss: Формы плоских тензоров классификации y_true и y_pred не совпадают!")

        # 3. Вычисляем Classification Loss (Focal Loss)
        # focal_loss работает с плоскими тензорами и маскирует игнорируемые якоря внутри
        # [ИЗМЕНЕНИЕ] Передаем self.focal_gamma
        cls_loss = focal_loss(y_true_cls_flat, y_pred_cls_flat, self.focal_alpha, self.focal_gamma)

        # 4. Вычисляем Box Regression Loss (GIoU или Huber)
        # Этот лосс применяется ТОЛЬКО к ПОЗИТИВНЫМ якорям.
        # Находим позитивные якоря
        # positive_mask имеет форму (batch, total_anchors). True там, где якорь позитивный.
        # y_true_cls_flat имеет one-hot для позитивов (например, [1, 0] или [0, 1]), нули для негативов, -1 для игнорируемых.
        # tf.reduce_any(y_true_cls_flat > 0.0, axis=-1) корректно находит позитивные якоря.
        positive_mask = tf.reduce_any(y_true_cls_flat > 0.0, axis=-1)  # Shape: (batch, total_anchors)

        # Находим индексы позитивных якорей в батче. indices shape (NumPositivesInBatch, 2), где 0-я колонка - индекс примера в батче, 1-я - плоский индекс якоря.
        indices = tf.where(positive_mask)

        reg_loss = 0.0  # Инициализируем регрессионную потерю нулем

        # Количество позитивных якорей в текущем батче (скалярное значение)
        num_positives_in_batch = tf.cast(tf.shape(indices)[0], tf.float32)

        # Вычисляем регрессионную потерю только если есть позитивные якоря в батче
        if tf.greater(num_positives_in_batch, 0.0):

            # Собираем y_true_reg и y_pred_reg ТОЛЬКО для позитивных якорей
            y_true_reg_pos = tf.gather_nd(y_true_reg_flat, indices)  # Shape: (NumPositivesInBatch, 4)
            y_pred_reg_pos = tf.gather_nd(y_pred_reg_flat, indices)  # Shape: (NumPositivesInBatch, 4)

            # Собираем соответствующие ЯКОРЯ ТОЛЬКО для позитивных якорей
            # indices[:, 1] дает нам плоские индексы якорей (0 до TotalAnchors-1) для каждого позитивного якоря в батче
            anchor_indices_flat = indices[:, 1]
            # Используем tf.gather для выбора нужных якорей из self.all_anchors_tf (который (TotalAnchors, 4))
            # all_anchors_tf не имеет батч-размерности, поэтому gather_nd не нужен, достаточно gather
            anchors_pos = tf.gather(self.all_anchors_tf, anchor_indices_flat)  # Shape: (NumPositivesInBatch, 4)

            # === Вычисляем Box Loss ===
            if self.box_loss_type == 'ciou' or self.box_loss_type == 'giou':  # Добавим поддержку 'giou' явно
                # Для GIoU/DIoU/CIoU Loss нужно декодировать боксы в [y1, x1, y2, x2]
                decoded_true_boxes = _decode_boxes(y_true_reg_pos,
                                                   anchors_pos)  # Shape: (NumPositivesInBatch, 4) [y1,x1,y2,x2]
                decoded_pred_boxes = _decode_boxes(y_pred_reg_pos,
                                                   anchors_pos)  # Shape: (NumPositivesInBatch, 4) [y1,x1,y2,x2]

                # self.box_loss_fn_per_box уже настроен на reduction=NONE
                # Он принимает [y1,x1,y2,x2] формат.
                # Результат loss_per_positive_anchor - это тензор формы (NumPositivesInBatch,)
                loss_per_positive_anchor = self.box_loss_fn_per_box(decoded_true_boxes, decoded_pred_boxes)

                # Суммируем потери по всем позитивным якорям в батче
                sum_box_loss = tf.reduce_sum(loss_per_positive_anchor)

            elif self.box_loss_type == 'huber':
                # Для Huber Loss не нужно декодировать боксы, он работает с (tx, ty, tw, th)
                # self.box_loss_fn_per_box уже настроен на reduction=NONE
                # Он принимает (tx, ty, tw, th) формат.
                # Результат loss_per_positive_anchor - это тензор формы (NumPositivesInBatch,)
                loss_per_positive_anchor = self.box_loss_fn_per_box(y_true_reg_pos, y_pred_reg_pos)

                # Суммируем потери по всем позитивным якорям в батче
                sum_box_loss = tf.reduce_sum(loss_per_positive_anchor)

            else:
                # Этот случай уже должен быть обработан в __init__, но на всякий случай
                raise ValueError(f"Неподдерживаемый тип Box Loss в call: {self.box_loss_type}")

            # Нормализуем суммарную регрессионную потерю по количеству позитивных якорей в батче
            # Используем num_positives_in_batch, вычисленное ранее
            reg_loss = tf.math.divide_no_nan(sum_box_loss, num_positives_in_batch)

        # else: reg_loss остается 0.0, если нет позитивных якорей в батче

        # 5. Суммируем взвешенные потери
        total_loss = (self.cls_loss_weight * cls_loss) + (self.reg_loss_weight * reg_loss)

        # === [ВАЖНО] Проверка на NaN/Inf в итоговой потере (в режиме графа) ===
        tf.debugging.check_numerics(total_loss, message='DetectorLoss: Total loss is NaN/Inf')

        return total_loss

# --- Тестовый блок ---
if __name__ == '__main__':
    logger.info("--- Тестирование DetectorLoss ---")

    # Установим сид для воспроизводимости тестовых данных
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Параметры для теста ---
    BATCH_SIZE = 4 # Используем батч > 1 для лучшего теста
    NUM_CLASSES = 2

    # Параметры, которые должны быть в конфиге для генерации реальных якорей и расчета форм
    # Используем значения по умолчанию, если конфиг не загружен или в нем нет нужных ключей
    input_shape_test = [512, 512, 3]
    fpn_strides_test = [8, 16, 32] # Страйды для P3, P4, P5
    num_anchors_per_level_test = 15 # Количество базовых якорей на ячейку (из конфига)
    anchor_scales_test = [1.0, 1.2599, 1.5874] # Для generate_all_anchors
    anchor_ratios_test = [0.126, 0.362, 1.136, 2.461, 6.093] # Для generate_all_anchors

    # Попытка загрузить реальный конфиг для более точных тестовых параметров
    try:
        _project_root = Path(__file__).parent.parent.parent.resolve()
        config_path = _project_root / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            real_config_for_test = yaml.safe_load(f)

        # Обновляем тестовые параметры из реального конфига, если они там есть
        input_shape_test = real_config_for_test.get('input_shape', input_shape_test)
        # fpn_strides_test = real_config_for_test.get('fpn_strides', fpn_strides_test) # Предполагаем стандартные 8,16,32
        num_anchors_per_level_test = real_config_for_test.get('num_anchors_per_level', num_anchors_per_level_test)
        anchor_scales_test = real_config_for_test.get('anchor_scales', anchor_scales_test)
        anchor_ratios_test = real_config_for_test.get('anchor_ratios', anchor_ratios_test)
        # Проверка на всякий случай, что num_anchors_per_level соответствует scales*ratios
        expected_na_from_sr = len(anchor_scales_test) * len(anchor_ratios_test)
        if num_anchors_per_level_test != expected_na_from_sr:
             logger.warning(f"num_anchors_per_level в конфиге ({num_anchors_per_level_test}) не совпадает с расчетом scales*ratios ({expected_na_from_sr}). Используется значение из конфига ({num_anchors_per_level_test}).")


    except Exception as e:
        logger.warning(f"Ошибка при загрузке конфига для тестовых параметров: {e}. Используются значения по умолчанию.")


    # --- Рассчитываем СОГЛАСОВАННОЕ общее количество якорей и количество по уровням для ТЕСТА ---
    # Этот расчет ОСНОВАН НА ТЕХ ЖЕ ПАРАМЕТРАХ, что используются в Data Loader и Model
    # Он гарантирует, что все расчеты будут согласованы внутри тестового блока.
    level_anchor_counts_test = [] # Количество якорей НА УРОВНЕ
    level_spatial_shapes_test = [] # Форма (H, W) НА УРОВНЕ
    input_shape_h, input_shape_w = input_shape_test[:2]
    for stride in fpn_strides_test:
        fh, fw = input_shape_h // stride, input_shape_w // stride
        level_spatial_shapes_test.append((fh, fw))
        level_anchor_counts_test.append(fh * fw * num_anchors_per_level_test)

    # Общее количество якорей для теста = сумма по уровням. Это главный источник truth для тестовых форм.
    TOTAL_ANCHORS_TEST = sum(level_anchor_counts_test)

    logger.info(f"Тестовые параметры: Input={input_shape_test}, Strides={fpn_strides_test}, Anchors/cell={num_anchors_per_level_test}")
    logger.info(f"Тестовые параметры: Всего якорей={TOTAL_ANCHORS_TEST}, По уровням (counts)={level_anchor_counts_test}, По уровням (spatial)={level_spatial_shapes_test}")

    # Проверка на всякий случай, что общее количество якорей ненулевое
    if TOTAL_ANCHORS_TEST == 0:
        logger.error("Ошибка конфигурации тестовых данных: Общее количество якорей = 0.")


    # --- Генерируем РЕАЛЬНЫЕ якоря для теста (по этим согласованным параметрам) ---
    # Этот all_anchors будет передан в DetectorLoss
    all_anchors_for_loss_test_np = None
    try:
        if 'generate_all_anchors' in globals() and generate_all_anchors is not None:
             # generate_all_anchors ожидает shape, strides, scales, ratios
             all_anchors_for_loss_test_np = generate_all_anchors(
                 input_shape_test,
                 fpn_strides_test,
                 anchor_scales_test, # generate_all_anchors ожидает эти аргументы
                 anchor_ratios_test
             )
             # Проверка на всякий случай, что generate_all_anchors выдала ожидаемое количество
             if all_anchors_for_loss_test_np.shape[0] != TOTAL_ANCHORS_TEST:
                 logger.warning(f"ПРЕДУПРЕЖДЕНИЕ: generate_all_anchors выдала {all_anchors_for_loss_test_np.shape[0]} якорей, но ожидалось {TOTAL_ANCHORS_TEST}. Используется фактическое.")
                 # Если generate_all_anchors выдала другое количество, это потенциальная проблема с самой generate_all_anchors
                 # Для теста просто используем фактическое количество, но логируем предупреждение
                 TOTAL_ANCHORS_TEST = all_anchors_for_loss_test_np.shape[0]
                 # В этом случае, level_anchor_counts_test и level_spatial_shapes_test
                 # могут быть не согласованы с TOTAL_ANCHORS_TEST.
                 # Это может сломать reshape ниже.
                 # Лучше, если generate_all_anchors всегда выдает количество, соответствующее расчету по уровням.
                 # Но для текущего теста, примем фактическое количество, но логика разбиения/reshape будет использовать ОЖИДАЕМЫЕ counts/shapes.
                 # Если они не совпадают, reshape сломается, что будет индикатором проблемы в generate_all_anchors или параметрах конфига.
                 pass # Продолжаем, но с предупреждением.


             logger.info(f"Сгенерировано {TOTAL_ANCHORS_TEST} реальных якорей для теста.")
        else:
             logger.warning("generate_all_anchors недоступна. Используются фиктивные якоря для теста.")
             # Создаем фиктивные якоря, как fallback, с ПРАВИЛЬНЫМ общим количеством
             all_anchors_for_loss_test_np = np.random.rand(TOTAL_ANCHORS_TEST, 4).astype(np.float32)
             all_anchors_for_loss_test_np[:, 2:] += all_anchors_for_loss_test_np[:, :2]
             all_anchors_for_loss_test_np = np.clip(all_anchors_for_loss_test_np, 0.0, 1.0)

    except Exception as e:
        logger.warning(f"Ошибка при генерации якорей в тесте: {e}. Используются фиктивные якоря.")
        # Создаем фиктивные якоря, как fallback, с ПРАВИЛЬНЫМ общим количеством
        all_anchors_for_loss_test_np = np.random.rand(TOTAL_ANCHORS_TEST, 4).astype(np.float32)
        all_anchors_for_loss_test_np[:, 2:] += all_anchors_for_loss_test_np[:, :2]
        all_anchors_for_loss_test_np = np.clip(all_anchors_for_loss_test_np, 0.0, 1.0)


    # --- Создание фиктивных ПЛОСКИХ данных y_true ---
    # Эти данные ИМИТИРУЮТ ВЫХОД Data Loader (плоские)
    # Они ДОЛЖНЫ иметь размерность TOTAL_ANCHORS_TEST
    y_true_reg_flat_np = np.random.randn(BATCH_SIZE, TOTAL_ANCHORS_TEST, 4).astype(np.float32) * 0.1 # Маленькие случайные цели
    # Изначально все негативные (0.0 в one-hot для всех классов)
    y_true_cls_flat_np = np.zeros((BATCH_SIZE, TOTAL_ANCHORS_TEST, NUM_CLASSES), dtype=np.float32)

    # Добавляем несколько позитивных, игнорируемых, негативных якорей в плоских индексах
    # Индексы должны быть МЕНЬШЕ TOTAL_ANCHORS_TEST
    positive_flat_indices_test = []
    ignored_flat_indices_test = []

    # Генерируем тестовые индексы позитивных/игнорируемых, распределяя их по уровням
    # Используем level_anchor_counts_test для определения диапазонов индексов по уровням
    current_total_idx = 0
    for i, level_count in enumerate(level_anchor_counts_test):
         # Добавим несколько тестовых индексов на каждом уровне
         if level_count > 10: # Убедимся, что уровень достаточно большой
              positive_flat_indices_test.append(current_total_idx + int(level_count * 0.1))
              positive_flat_indices_test.append(current_total_idx + int(level_count * 0.5))
              ignored_flat_indices_test.append(current_total_idx + int(level_count * 0.2))
              ignored_flat_indices_test.append(current_total_idx + int(level_count * 0.6))
         elif level_count > 0: # Добавим хотя бы один, если возможно
              positive_flat_indices_test.append(current_total_idx + min(5, level_count - 1))
              ignored_flat_indices_test.append(current_total_idx + min(8, level_count - 1))
         current_total_idx += level_count

    # Убедимся, что тестовые индексы не выходят за пределы TOTAL_ANCHORS_TEST (двойная проверка)
    positive_flat_indices_test = [idx for idx in positive_flat_indices_test if idx < TOTAL_ANCHORS_TEST]
    ignored_flat_indices_test = [idx for idx in ignored_flat_indices_test if idx < TOTAL_ANCHORS_TEST]


    # Заполняем y_true_cls_flat_np метками
    for i in range(BATCH_SIZE):
        # Используем одни и те же плоские индексы для всех батчей в этом тесте
        for idx in positive_flat_indices_test:
             # Назначаем случайный класс для позитивного якоря
             random_class_id = np.random.randint(0, NUM_CLASSES)
             y_true_cls_flat_np[i, idx, random_class_id] = 1.0 # One-hot
             # Установим какие-то цели регрессии для этих позитивов
             y_true_reg_flat_np[i, idx, :] = np.random.randn(4) # Случайные, но ненулевые цели

        for idx in ignored_flat_indices_test:
             y_true_cls_flat_np[i, idx, :] = -1.0 # Метка игнорируемого


    y_true = (tf.constant(y_true_reg_flat_np), tf.constant(y_true_cls_flat_np))


    # --- Создаем СПИСОК фиктивных предсказаний по уровням (ИМИТАЦИЯ ВЫХОДА МОДЕЛИ), используя согласованные размеры ---
    # Это ИМИТИРУЕТ ВЫХОД МОДЕЛИ
    y_pred = []
    num_levels = len(fpn_strides_test)

    for i in range(num_levels):
        batch_size_tf = tf.shape(y_true[0])[0] # Берем батч-размерность из y_true
        H, W = level_spatial_shapes_test[i] # Пространственные размеры для этого уровня
        num_anchors = num_anchors_per_level_test # Якорей на ячейку для этого уровня
        # level_total_count = level_anchor_counts_test[i] # Не нужен для этого способа

        # === Создаем фиктивный тензор для текущего уровня с ПРАВИЛЬНОЙ конечной формой ===
        # Shape: (batch, H, W, num_anchors, tasks)
        reg_tensor_level = tf.random.uniform((batch_size_tf, H, W, num_anchors, 4), dtype=tf.float32) * 0.05
        cls_tensor_level = tf.random.uniform((batch_size_tf, H, W, num_anchors, NUM_CLASSES), dtype=tf.float32) * 0.1

        # Для теста: делаем предсказания для нескольких якорей "хорошими" в каждом батче
        # Выбираем num_positives_this_level случайных ячеек и якорей в них для каждого примера в батче
        num_positives_this_level_per_batch = min(2, H*W*num_anchors) # Максимум 2 "хороших" на уровень на батч для теста
        if num_positives_this_level_per_batch > 0:
             # Выбираем случайные индексы (batch_idx, H, W, num_anchors)
             batch_indices_scatter = np.arange(BATCH_SIZE).repeat(num_positives_this_level_per_batch) # Индексы батчей
             level_indices_H = np.random.randint(0, H, size=BATCH_SIZE * num_positives_this_level_per_batch)
             level_indices_W = np.random.randint(0, W, size=BATCH_SIZE * num_positives_this_level_per_batch)
             level_indices_A = np.random.randint(0, num_anchors, size=BATCH_SIZE * num_positives_this_level_per_batch)

             scatter_indices = np.stack([batch_indices_scatter, level_indices_H, level_indices_W, level_indices_A], axis=1)

             # Создаем фиктивные цели для этих выбранных якорей.
             fake_y_true_reg_targets = np.random.randn(BATCH_SIZE * num_positives_this_level_per_batch, 4) * 0.1
             fake_class_ids = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE * num_positives_this_level_per_batch)
             fake_y_true_cls_targets = np.zeros((BATCH_SIZE * num_positives_this_level_per_batch, NUM_CLASSES), dtype=np.float32)
             fake_y_true_cls_targets[np.arange(BATCH_SIZE * num_positives_this_level_per_batch), fake_class_ids] = 1.0 # One-hot

             # Делаем предсказания близкими к фиктивным целям
             reg_update_values = fake_y_true_reg_targets + np.random.randn(BATCH_SIZE * num_positives_this_level_per_batch, 4) * 0.01
             # Делаем логиты высокими
             cls_update_values = np.random.randn(BATCH_SIZE * num_positives_this_level_per_batch, NUM_CLASSES) * 0.1
             cls_update_values[np.arange(BATCH_SIZE * num_positives_this_level_per_batch), fake_class_ids] = 5.0 # Высокий логит

             # Обновляем тензоры уровня
             reg_tensor_level = tf.tensor_scatter_nd_update(reg_tensor_level, scatter_indices, reg_update_values)
             cls_tensor_level = tf.tensor_scatter_nd_update(cls_tensor_level, scatter_indices, cls_update_values)


        # Добавляем созданные тензоры уровня в список y_pred
        y_pred.append(reg_tensor_level)
        y_pred.append(cls_tensor_level)


    logger.info("\nТест DetectorLoss с focal_loss и CIoU:")
    try:
        # all_anchors_for_loss_test_np - это сгенерированные якоря (np.ndarray)
        # DetectorLoss ожидает np.ndarray или tf.Tensor
        # Передаем test_config как словарь, как и положено.
        test_config_dict = {
            'num_classes': NUM_CLASSES,
            'loss_weights': {'classification': 1.0, 'box_regression': 1.5},
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 1.25, # Используем 1.25 как в финальном конфиге
            'box_loss_type': 'ciou', # Тестируем CIoU
            'huber_loss_delta': 1.0 # Не используется для CIoU
        }

        loss_fn = DetectorLoss(test_config_dict, all_anchors=all_anchors_for_loss_test_np)

        # Вычисляем потерю. y_true - это (y_true_reg_flat, y_true_cls_flat) кортеж. y_pred - это список тензоров по уровням.
        total_loss = loss_fn(y_true, y_pred)

        # --- Выводим компоненты потерь для более детальной проверки ---
        # Вычисление компонентов отдельно для теста (требует доступа к приватным методам или повторения логики из call)
        # Классификационная потеря (используем плоские тензоры)
        # Конкатенируем y_pred_cls по уровням в плоский тензор для сравнения с y_true_cls_flat
        # [ИСПРАВЛЕНО] Используем срезы 1::2 для получения списка cls тензоров
        y_pred_cls_list_for_check = y_pred[1::2] # y_pred[1::2] содержит список cls тензоров по уровням
        y_pred_cls_flat_for_check = tf.concat([tf.reshape(t, [tf.shape(t)[0], -1, NUM_CLASSES]) for t in y_pred_cls_list_for_check], axis=1) # Расплющиваем и конкатенируем

        cls_loss_val = focal_loss(y_true[1], y_pred_cls_flat_for_check, # y_true_cls_flat, y_pred_cls_flat
                                  test_config_dict['focal_loss_alpha'], test_config_dict['focal_loss_gamma'])
        logger.info(f"  - Classification Loss (Focal): {cls_loss_val.numpy():.4f}")

        # Регрессионная потеря (требует извлечения позитивных якорей и декодирования)
        # Получаем позитивные якоря из плоских y_true
        positive_mask_test = tf.reduce_any(y_true[1] > 0.0, axis=-1)
        indices_test = tf.where(positive_mask_test)
        num_positives_test = tf.cast(tf.shape(indices_test)[0], tf.float32)

        if tf.greater(num_positives_test, 0.0):
            # Собираем y_true_reg и y_pred_reg ТОЛЬКО для позитивных якорей (плоские)
            y_true_reg_pos_test = tf.gather_nd(y_true[0], indices_test)
            # Конкатенируем предсказания регрессии по уровням в плоский тензор перед gather_nd
            # [ИСПРАВЛЕНО] Используем срезы 0::2 для получения списка reg тензоров
            y_pred_reg_list_for_gather = y_pred[0::2] # y_pred[0::2] содержит список reg тензоров по уровням
            y_pred_reg_flat_for_gather = tf.concat([tf.reshape(t, [tf.shape(t)[0], -1, 4]) for t in y_pred_reg_list_for_gather], axis=1) # Расплющиваем и конкатенируем
            y_pred_reg_pos_test = tf.gather_nd(y_pred_reg_flat_for_gather, indices_test)

            # Собираем соответствующие ЯКОРЯ ТОЛЬКО для позитивных якорей
            anchor_indices_flat_test = indices_test[:, 1]
            anchors_pos_test = tf.gather(loss_fn.all_anchors_tf, anchor_indices_flat_test) # Собираем якоря

            # Декодируем истинные и предсказанные боксы для позитивных якорей
            decoded_true_boxes_test = _decode_boxes(y_true_reg_pos_test, anchors_pos_test)
            decoded_pred_boxes_test = _decode_boxes(y_pred_reg_pos_test, anchors_pos_test)

            # Вычисляем Box Loss per box
            # Используем box_loss_fn_per_box из экземпляра loss_fn для теста
            loss_per_positive_anchor_test = loss_fn.box_loss_fn_per_box(decoded_true_boxes_test, decoded_pred_boxes_test)

            # Суммируем и нормализуем
            sum_box_loss_test = tf.reduce_sum(loss_per_positive_anchor_test)
            reg_loss_val = tf.math.divide_no_nan(sum_box_loss_test, num_positives_test)
            logger.info(f"  - Regression Loss ({test_config_dict['box_loss_type'].upper()}): {reg_loss_val.numpy():.4f}")

            # Проверяем, что суммарная потеря соответствует взвешенной сумме компонент
            expected_total_loss = test_config_dict['loss_weights']['classification'] * cls_loss_val + test_config_dict['loss_weights']['box_regression'] * reg_loss_val
            logger.info(f"  - Ожидаемая Total Weighted Loss: {expected_total_loss.numpy():.4f}")
            assert np.isclose(total_loss.numpy(), expected_total_loss.numpy(), atol=1e-5), "Суммарная потеря не соответствует взвешенной сумме компонент!"

        else:
             logger.info("  - Regression Loss: Нет позитивных якорей в тестовом батче.")
             # Проверяем, что total_loss = cls_loss * cls_weight
             expected_total_loss = test_config_dict['loss_weights']['classification'] * cls_loss_val
             assert np.isclose(total_loss.numpy(), expected_total_loss.numpy(), atol=1e-5), "Суммарная потеря не соответствует взвешенной сумме компонент (нет позитивов)!"


        logger.info(f"  - Total Weighted Loss: {total_loss.numpy():.4f}")

        assert not np.isnan(total_loss.numpy()), "Total Loss не должен быть NaN"
        # При случайных данных loss > 0
        # assert total_loss.numpy() > 0, "Total Loss должен быть положительным (при случайных данных)"

        logger.info("  - [SUCCESS] Тест пройден.")

    except Exception as e:
        logger.error(f"  - [ERROR] Ошибка при вычислении потерь: {e}")
        import traceback;

        traceback.print_exc()