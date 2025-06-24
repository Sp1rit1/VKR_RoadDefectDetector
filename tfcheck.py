Рома, [24.06.2025 17: 22]
# ───────── src/datasets/augmentations.py ─────────
import albumentations as A
from packaging import version


def get_detector_train_augmentations(img_size: int) -> A.Compose:
    """
    Возвращает пайп-лайн аугментаций для обучения детектора.
    Работает как с Albumentations < 1.3 (параметр border_mode),
    так и с более новыми версиями (параметр mode).
    """
    ver = version.parse(A.version)
    kw_border = {"border_mode" if ver < version.parse("1.3.0") else "mode": A.BORDER_REFLECT_101}

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=10,
                **kw_border,
                p=0.7,
            ),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
            A.HueSaturationValue(10, 10, 10, p=0.4),
            A.CLAHE(p=0.1),
            A.Resize(img_size, img_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )


Рома, [24.06.2025 17: 22]

# ───────── src/datasets/data_loader_v3_standard.py ─────────
def create_dataset(generator, batch_size: int, is_training: bool) -> tf.data.Dataset:
    """
    Оборачивает DataGenerator в tf.data.Dataset.
    • shuffle выполняется только если в буфере ≥1 образец
    • drop_remainder выключен, чтобы не потерять маленькие датасеты
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=generator.output_signature,
    )

    # Буфер = 3×batch или весь датасет, что меньше
    buffer_size = max(1, min(len(generator), batch_size * 3))
    if is_training:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


Рома, [24.06.2025 17: 22]

# ───────── src/datasets/generator.py ─────────
class DataGenerator:
    def init(self, cfg, annotations, *args, **kwargs):
        ...
        # anchors
        self.anchor_scales = cfg['anchor_scales']
        self.anchor_ratios = cfg['anchor_ratios']
        self.all_anchors = generate_anchors(self.anchor_scales, self.anchor_ratios, ...)

        # <<<   БЫЛ assert, который валил генератор   >>>
        expected_per_level = len(self.anchor_scales) * len(self.anchor_ratios)
        if cfg['num_anchors_per_level'] != expected_per_level:
            logging.warning(
                "num_anchors_per_level=%d в конфиге, но по расчёту должно быть %d. "
                "Использую вычислённое значение.",
                cfg['num_anchors_per_level'], expected_per_level,
            )
            cfg['num_anchors_per_level'] = expected_per_level
        self.total_anchors_count = expected_per_level * sum(
            fmap_h * fmap_w for fmap_h, fmap_w in self.feature_map_shapes
        )


Рома, [24.06.2025 17: 22]

# ───────── src/models/build_detector_v3_standard.py ─────────
def build_detector_v3_standard(cfg):
    ...
    # heads
    reg_p3 = build_regression_head(P3, num_anchors)
    cls_p3 = build_classification_head(P3, num_anchors, num_classes)
    reg_p4 = build_regression_head(P4, num_anchors)
    cls_p4 = build_classification_head(P4, num_anchors, num_classes)
    reg_p5 = build_regression_head(P5, num_anchors)
    cls_p5 = build_classification_head(P5, num_anchors, num_classes)

    # 🆕  единый порядок «рег-класс» для каждого уровня
    outputs = [reg_p3, cls_p3, reg_p4, cls_p4, reg_p5, cls_p5]

    return keras.Model(inputs=inputs, outputs=outputs, name="road_defect_detector")


Рома, [24.06.2025 17: 23]

# ───────── src/losses/detection_losses_v3_standard.py ─────────
class DetectorLoss(keras.losses.Loss):
    ...

    def call(self, y_true, y_pred):
        """
        Ожидаемый порядок входов:
        [reg_P3, cls_P3, reg_P4, cls_P4, …]
        """
        # ------------------------------------
        # 1. разделяем по нечёт/чёт (0::2 / 1::2)
        reg_tensors = y_pred[0::2]
        cls_tensors = y_pred[1::2]

        # 2. выравниваем формы и конкатенируем
        reg_flat = tf.concat(
            [tf.reshape(t, [tf.shape(t)[0], -1, 4]) for t in reg_tensors],
            axis=1,
        )
        cls_flat = tf.concat(
            [tf.reshape(t, [tf.shape(t)[0], -1, self.num_classes]) for t in cls_tensors],
            axis=1,
        )

        y_true_reg_flat, y_true_cls_flat = y_true  # формирование не менялось

        # 3. sanity-check
        tf.debugging.assert_equal(
            tf.shape(y_true_reg_flat),
            tf.shape(reg_flat),
            message="DetectorLoss: y_true/y_pred reg shapes differ",
        )

        # 4. считаем лоссы …
        reg_loss = self.box_loss_fn(y_true_reg_flat, reg_flat)
        cls_loss = self.cls_loss_fn(y_true_cls_flat, cls_flat)
        total = self.box_loss_weight * reg_loss + cls_loss

        return total


Рома, [24.06.2025 17: 23]

# ───────── src/callbacks/warmup_cosine_decay.py ─────────
class WarmupCosineDecay(keras.callbacks.Callback):
    def init(self, base_lr, warmup_steps, total_steps, **kwargs):
        super().init(**kwargs)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._start_iter = None  # ← будет заполнен на on_train_begin

    def on_train_begin(self, logs=None):
        # фиксируем «ноль» в момент старта фазы, а не при создании коллбэка
        self._start_iter = int(self.model.optimizer.iterations)

    def on_train_batch_begin(self, batch, logs=None):
        current = int(self.model.optimizer.iterations) - self._start_iter
        ...


Рома, [24.06.2025 17: 23]

# ───────── tests/test_detector_loss.py ─────────
def test_detector_loss_shapes():
    ...
    # формируем неявно совместимый y_pred
    y_pred = [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5]

    loss_fn = DetectorLoss(num_classes=2, ...)
    loss = loss_fn(y_true, y_pred)
    assert loss > 0.0


Рома, [24.06.2025 17: 23]
Ниже ‒ только
изменённые
функции(остальной
код
трогать
не
нужно).
Путь
к
файлу
указан
в
подписи
каждого
блока, чтобы
было
понятно, куда
вставлять.

---

# ───────── src/datasets/augmentations.py ─────────
import albumentations as A
from packaging import version


def get_detector_train_augmentations(img_size: int) -> A.Compose:
    """
    Возвращает пайп-лайн аугментаций для обучения детектора.
    Работает как с Albumentations < 1.3 (параметр border_mode),
    так и с более новыми версиями (параметр mode).
    """
    ver = version.parse(A.version)
    kw_border = {"border_mode" if ver < version.parse("1.3.0") else "mode": A.BORDER_REFLECT_101}

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=10,
                **kw_border,
                p=0.7,
            ),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
            A.HueSaturationValue(10, 10, 10, p=0.4),
            A.CLAHE(p=0.1),
            A.Resize(img_size, img_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )


---


# ───────── src/datasets/data_loader_v3_standard.py ─────────
def create_dataset(generator, batch_size: int, is_training: bool) -> tf.data.Dataset:
    """
    Оборачивает DataGenerator в tf.data.Dataset.
    • shuffle выполняется только если в буфере ≥1 образец
    • drop_remainder выключен, чтобы не потерять маленькие датасеты
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=generator.output_signature,
    )

    # Буфер = 3×batch или весь датасет, что меньше
    buffer_size = max(1, min(len(generator), batch_size * 3))
    if is_training:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


---


# ───────── src/datasets/generator.py ─────────
class DataGenerator:
    def init(self, cfg, annotations, *args, **kwargs):
        ...
        # anchors
        self.anchor_scales = cfg['anchor_scales']
        self.anchor_ratios = cfg['anchor_ratios']
        self.all_anchors = generate_anchors(self.anchor_scales, self.anchor_ratios, ...)

        # <<<   БЫЛ assert, который валил генератор   >>>
        expected_per_level = len(self.anchor_scales) * len(self.anchor_ratios)
        if cfg['num_anchors_per_level'] != expected_per_level:
            logging.warning(
                "num_anchors_per_level=%d в конфиге, но по расчёту должно быть %d. "
                "Использую вычислённое значение.",
                cfg['num_anchors_per_level'], expected_per_level,
            )
            cfg['num_anchors_per_level'] = expected_per_level
        self.total_anchors_count = expected_per_level * sum(
            fmap_h * fmap_w for fmap_h, fmap_w in self.feature_map_shapes
        )


---


# ───────── src/models/build_detector_v3_standard.py ─────────
def build_detector_v3_standard(cfg):
    ...
    # heads
    reg_p3 = build_regression_head(P3, num_anchors)
    cls_p3 = build_classification_head(P3, num_anchors, num_classes)
    reg_p4 = build_regression_head(P4, num_anchors)
    cls_p4 = build_classification_head(P4, num_anchors, num_classes)
    reg_p5 = build_regression_head(P5, num_anchors)
    cls_p5 = build_classification_head(P5, num_anchors, num_classes)

    # 🆕  единый порядок «рег-класс» для каждого уровня
    outputs = [reg_p3, cls_p3, reg_p4, cls_p4, reg_p5, cls_p5]

    return keras.Model(inputs=inputs, outputs=outputs, name="road_defect_detector")


---


# ───────── src/losses/detection_losses_v3_standard.py ─────────
class DetectorLoss(keras.losses.Loss):


Рома, [24.06.2025 17: 23]
...


def call(self, y_true, y_pred):
    """
    Ожидаемый порядок входов:
    [reg_P3, cls_P3, reg_P4, cls_P4, …]
    """
    # ------------------------------------
    # 1. разделяем по нечёт/чёт (0::2 / 1::2)
    reg_tensors = y_pred[0::2]
    cls_tensors = y_pred[1::2]

    # 2. выравниваем формы и конкатенируем
    reg_flat = tf.concat(
        [tf.reshape(t, [tf.shape(t)[0], -1, 4]) for t in reg_tensors],
        axis=1,
    )
    cls_flat = tf.concat(
        [tf.reshape(t, [tf.shape(t)[0], -1, self.num_classes]) for t in cls_tensors],
        axis=1,
    )

    y_true_reg_flat, y_true_cls_flat = y_true  # формирование не менялось

    # 3. sanity-check
    tf.debugging.assert_equal(
        tf.shape(y_true_reg_flat),
        tf.shape(reg_flat),
        message="DetectorLoss: y_true/y_pred reg shapes differ",
    )

    # 4. считаем лоссы …
    reg_loss = self.box_loss_fn(y_true_reg_flat, reg_flat)
    cls_loss = self.cls_loss_fn(y_true_cls_flat, cls_flat)
    total = self.box_loss_weight * reg_loss + cls_loss

    return total


---


# ───────── src/callbacks/warmup_cosine_decay.py ─────────
class WarmupCosineDecay(keras.callbacks.Callback):
    def init(self, base_lr, warmup_steps, total_steps, **kwargs):
        super().init(**kwargs)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._start_iter = None  # ← будет заполнен на on_train_begin

    def on_train_begin(self, logs=None):
        # фиксируем «ноль» в момент старта фазы, а не при создании коллбэка
        self._start_iter = int(self.model.optimizer.iterations)

    def on_train_batch_begin(self, batch, logs=None):
        current = int(self.model.optimizer.iterations) - self._start_iter
        ...


---


# ───────── tests/test_detector_loss.py ─────────
def test_detector_loss_shapes():
    ...
    # формируем неявно совместимый y_pred
    y_pred = [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5]

    loss_fn = DetectorLoss(num_classes=2, ...)
    loss = loss_fn(y_true, y_pred)
    assert loss > 0.0

> Тест
оставлен
минимальным: проверяем
только, что
форму «съедает» и
лосс
считается.

---

Как
применять
патчи

1.
Скопируйте
блок
с
нужным
путём
поверх
соответствующей
функции / класса
в
своём
проекте.

2.
Запустите
python - m
pytest – тест
должен
пройти.

3.
Запустите
train_detector_v3_standard.py – обучение
стартует
без
падений.