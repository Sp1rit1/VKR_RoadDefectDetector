import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Input, Lambda
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import math

logger = logging.getLogger(__name__)
# Установим уровень только если он не был установлен ранее (для избежания дублирования)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DetectorModel(Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        # loss_fn НЕ будет устанавливаться через compile, мы передадим его сами
        self.loss_calculator = None

        # Наши кастомные метрики, которые мы контролируем
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    # [ИСПРАВЛЕНО] compile теперь передает loss в super().compile
    def compile(self, optimizer, loss, **kwargs):
        """
        Компилирует модель с кастомной функцией потерь и метриками.

        Args:
            optimizer: Оптимизатор.
            loss: Наша кастомная функция потерь (например, DetectorLoss).
            **kwargs: Другие аргументы для Keras compile.
        """
        # [ИСПРАВЛЕНО] Мы передаем loss в super().compile, чтобы Keras знал
        # о нем для сериализации. Это не сломает наш кастомный train_step.
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        # Сохраняем нашу функцию потерь в атрибут класса для использования в train_step/test_step
        self.loss_calculator = loss


    @property
    def metrics(self):
        # Keras будет отслеживать только эти метрики
        return [self.loss_tracker, self.val_loss_tracker]

    def train_step(self, data):
        images, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            # ИЗМЕНЕНИЕ: используем наш сохраненный loss_calculator
            loss = self.loss_calculator(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        # Возвращаем словарь с ключом, совпадающим с именем метрики
        return {"loss": self.loss_tracker.result()}

    # [ИЗМЕНЕНО] test_step теперь возвращает val_loss
    def test_step(self, data):
        images, y_true = data
        y_pred = self(images, training=False)
        # ИЗМЕНЕНИЕ: используем наш сохраненный loss_calculator
        loss = self.loss_calculator(y_true, y_pred)

        self.val_loss_tracker.update_state(loss)
        # [ИСПРАВЛЕНО] Возвращаем словарь с ключом 'val_loss', как имя метрики
        return {"loss": self.val_loss_tracker.result()}


    # [НОВЫЙ МЕТОД] reset_metrics для сброса состояния всех метрик
    def reset_metrics(self):
        """
        Сбрасывает состояние всех метрик.
        Вызывается Keras автоматически в начале каждой эпохи и при model.evaluate().
        """
        self.loss_tracker.reset_state()
        self.val_loss_tracker.reset_state()
        # Вызов super().reset_metrics() не обязателен, если мы сбрасываем все метрики вручную
        # super().reset_metrics() # Можно добавить для совместимости, если в базовом классе есть другие метрики

# --- Вспомогательные функции для построения модели ---

def build_retinanet_head(fpn_filters, num_anchors, num_classes, level_name, kernel_regularizer=None):
    """
    Создает общую "голову" RetinaNet и отдельные ветки для классификации и регрессии.

    Args:
        fpn_filters (int): Количество фильтров в FPN.
        num_anchors (int): Количество якорей на ячейку сетки для этого уровня.
        num_classes (int): Количество классов для детекции.
        level_name (str): Имя уровня FPN (напр., 'P3').
        kernel_regularizer (tf.keras.regularizers.Regularizer, optional):
            Регуляризатор для весов сверточных слоев. По умолчанию None.

    Returns:
        tf.keras.Model: Модель головы предсказаний для данного уровня FPN.
    """
    input_tensor = Input(shape=(None, None, fpn_filters), name=f'head_input_{level_name}')

    shared_head = input_tensor
    for i in range(4):
        shared_head = Conv2D(fpn_filters, 3, 1, 'same',
                             kernel_initializer='he_normal',
                             # [ИЗМЕНЕНИЕ] Применяем регуляризатор, если он задан
                             kernel_regularizer=kernel_regularizer,
                             name=f'{level_name}_head_shared_conv_{i}')(shared_head)
        # BatchNormalization по умолчанию trainable=True
        shared_head = BatchNormalization(name=f'{level_name}_head_shared_bn_{i}')(shared_head)
        shared_head = ReLU(name=f'{level_name}_head_shared_relu_{i}')(shared_head)

    # Финальный слой классификации
    cls_head = Conv2D(num_anchors * num_classes, 3, 1, 'same',
                      kernel_initializer='he_normal',
                      # [ИЗМЕНЕНИЕ] Применяем регуляризатор к финальному слою
                      kernel_regularizer=kernel_regularizer,
                      # Инициализация bias'а для Focal Loss
                      bias_initializer=tf.keras.initializers.Constant(math.log((1 - 0.01) / 0.01)),
                      name=f'{level_name}_cls_conv')(shared_head)
    # Lambda для изменения формы на (batch, H, W, num_anchors, num_classes)
    cls_output = Lambda(
        lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_anchors, num_classes]),
        name=f'{level_name}_cls_output')(cls_head)

    # Финальный слой регрессии
    reg_head = Conv2D(num_anchors * 4, 3, 1, 'same',
                      kernel_initializer='he_normal',
                      # [ИЗМЕНЕНИЕ] Применяем регуляризатор к финальному слою
                      kernel_regularizer=kernel_regularizer,
                      name=f'{level_name}_reg_conv')(shared_head)
    # Lambda для изменения формы на (batch, H, W, num_anchors, 4)
    reg_output = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_anchors, 4]),
                        name=f'{level_name}_reg_output')(reg_head)

    head_model = Model(inputs=input_tensor, outputs=[reg_output, cls_output], name=f'prediction_head_{level_name}')
    return head_model


def build_detector_v3_standard(config):
    """
    Строит стандартную модель детектора с EfficientNetB0, FPN и
    ОБЩИМИ (shared) головами предсказаний в стиле RetinaNet.
    Может добавлять L2-регуляризацию к весам голов предсказаний.
    Возвращает кастомную модель с переопределенным train_step.
    """
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    fpn_filters = config['fpn_filters']
    num_anchors = config['num_anchors_per_level'] # Количество якорей на ячейку (одно число)
    freeze_backbone = config.get('freeze_backbone', True)

    # [ИЗМЕНЕНИЕ] Проверяем, задана ли L2-регуляризация в конфиге
    l2_value = config.get('l2_regularization', None)
    if l2_value is not None and l2_value > 0:
        head_regularizer = tf.keras.regularizers.l2(l2_value)
        logger.info(f"Добавлена L2-регуляризация с коэффициентом: {l2_value}")
    else:
        head_regularizer = None
        logger.info("L2-регуляризация НЕ используется.")


    input_tensor = Input(shape=input_shape, name='input_image')

    # Backbone: EfficientNetB0
    # BatchNormalization слои в EfficientNet по умолчанию trainable=True, но при base_model.trainable=False они замораживаются.
    base_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet')
    if freeze_backbone:
        base_model.trainable = False
        print("EfficientNetB0 backbone is FROZEN.")
    else:
        base_model.trainable = True
        print("EfficientNetB0 backbone is TRAINABLE (fine-tuning mode).")

    # Слои из EfficientNet, которые служат источниками признаков для FPN
    # Проверьте model.summary(), чтобы убедиться в правильности имен слоев для вашего input_shape
    c3_layer_name = 'block3b_add' # Stride 8 (shape 64x64 для 512x512)
    c4_layer_name = 'block4c_add' # Stride 16 (shape 32x32 для 512x512)
    c5_layer_name = 'block6d_add' # Stride 32 (shape 16x16 для 512x512)

    try:
        c3_output = base_model.get_layer(c3_layer_name).output
        c4_output = base_model.get_layer(c4_layer_name).output
        c5_output = base_model.get_layer(c5_layer_name).output
        print("Backbone feature outputs identified.")
    except ValueError as e:
        print(f"ERROR: Could not find required backbone layers. Verify layer names match base_model.summary(). Error: {e}")
        raise # Re-raise the exception


    # Neck: Feature Pyramid Network (FPN)
    # Lateral connections (1x1 conv)
    c3_lateral = Conv2D(fpn_filters, 1, 1, 'same', kernel_initializer='he_normal', name='fpn_c3_lateral')(c3_output)
    c4_lateral = Conv2D(fpn_filters, 1, 1, 'same', kernel_initializer='he_normal', name='fpn_c4_lateral')(c4_output)
    c5_lateral = Conv2D(fpn_filters, 1, 1, 'same', kernel_initializer='he_normal', name='fpn_c5_lateral')(c5_output)

    # Top-down pathway (Upsampling + Add)
    p5_top_down = c5_lateral
    p4_top_down = Add(name='fpn_p4_add')([UpSampling2D(2, name='fpn_p5_upsample')(p5_top_down), c4_lateral])
    p3_top_down = Add(name='fpn_p3_add')([UpSampling2D(2, name='fpn_p4_upsample')(p4_top_down), c3_lateral])

    # Output smoothing (3x3 conv)
    p3 = Conv2D(fpn_filters, 3, 1, 'same', kernel_initializer='he_normal', name='fpn_p3_output')(p3_top_down)
    p4 = Conv2D(fpn_filters, 3, 1, 'same', kernel_initializer='he_normal', name='fpn_p4_output')(p4_top_down)
    p5 = Conv2D(fpn_filters, 3, 1, 'same', kernel_initializer='he_normal', name='fpn_p5_output')(p5_top_down)

    # FPN outputs - order matters: P3, P4, P5 (smallest stride first)
    fpn_outputs = [p3, p4, p5]
    print("FPN neck построен.")

    # Prediction Heads (Shared weights across FPN levels)
    # Создаем ОДНУ модель головы и применяем ее к каждому выходу FPN
    # [ИЗМЕНЕНИЕ] Передаем регуляризатор в build_retinanet_head
    # В RetinaNet головы раздельные, но имеют общую архитектуру и ВЕСА (shared weights)
    # То есть, это один и тот же набор сверток, применяемый к разным feature maps.
    # Поэтому мы создаем одну модель головы и вызываем ее для каждого выхода FPN.
    shared_head_model = build_retinanet_head(fpn_filters, num_anchors, num_classes, 'Shared', kernel_regularizer=head_regularizer)

    # Применяем Shared Head к каждому выходу FPN
    reg_p3, cls_p3 = shared_head_model(p3)
    reg_p4, cls_p4 = shared_head_model(p4)
    reg_p5, cls_p5 = shared_head_model(p5)
    print("Общие (shared) головы предсказаний построены и применены.")

    # [ИСПРАВЛЕНО] Собираем выходы в чередующемся порядке, как ожидает DetectorLoss.
    # Порядок: [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5]
    outputs = [reg_p3, cls_p3, reg_p4, cls_p4, reg_p5, cls_p5]

    # Создаем DetectorModel с кастомным train_step
    model = DetectorModel(inputs=input_tensor, outputs=outputs, name=config.get('model_name', 'RetinaNetDetector'))
    print(f"Модель '{model.name}' (с кастомным train_step) успешно собрана.")

    return model



# --- Тестовый блок для проверки сборки ---
if __name__ == '__main__':
    print("--- Тестирование сборки модели ---")

    # Фиктивный конфиг для теста сборки
    test_config = {
        'input_shape': [512, 512, 3],
        'num_classes': 2,
        'fpn_filters': 256,
        'num_anchors_per_level': 15,  # <--- Используем 15 якорей на ячейку
        'freeze_backbone': True,
        'model_name': 'Test_DetectorV3_RetinaNet',
        'l2_regularization': 1e-4 # Тестируем с регуляризацией
    }

    try:
        model = build_detector_v3_standard(test_config)
        # Выводим summary модели, чтобы проверить слои и их trainable статус
        model.summary(line_length=150)

        # Проверяем trainable статус BN слоев в Backbone и Head
        print("\n--- Проверка trainable статуса BatchNormalization слоев ---")
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                 print(f"Слой BN: {layer.name}, trainable: {layer.trainable}")
            if hasattr(layer, 'layers'): # Проверяем вложенные слои (для Backbone и Head модели)
                 for sub_layer in layer.layers:
                      if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                           print(f"  Вложенный слой BN: {sub_layer.name}, trainable: {sub_layer.trainable}")


        # Проверяем формы выходных тензоров
        BATCH_SIZE = 2
        dummy_input = tf.random.normal((BATCH_SIZE, *test_config['input_shape']))
        outputs = model(dummy_input)

        print("\n--- Проверка форм выходных тензоров ---")
        strides = [8, 16, 32] # Страйды для P3, P4, P5
        num_levels = 3
        # Количество якорей на ячейку из test_config
        num_anchors_per_level = test_config['num_anchors_per_level']

        # [ИСПРАВЛЕНО] Проходим по уровням и берем правильные индексы из outputs
        for i in range(num_levels):
            level_name = f"P{i + 3}"
            H, W = test_config['input_shape'][0] // strides[i], test_config['input_shape'][1] // strides[i]

            # Индексы в списке outputs для текущего уровня
            reg_idx = i * 2       # 0, 2, 4
            cls_idx = i * 2 + 1   # 1, 3, 5

            # Проверяем регрессионный выход
            expected_reg_shape = (BATCH_SIZE, H, W, num_anchors_per_level, 4)
            actual_reg_shape = tuple(outputs[reg_idx].shape)
            print(f"Выход reg_{level_name}: {actual_reg_shape}, Ожидалось: {expected_reg_shape}")
            assert actual_reg_shape == expected_reg_shape, f"ОШИБКА ФОРМЫ для reg_{level_name}"

            # Проверяем классификационный выход
            expected_cls_shape = (BATCH_SIZE, H, W, num_anchors_per_level, test_config['num_classes'])
            actual_cls_shape = tuple(outputs[cls_idx].shape)
            print(f"Выход cls_{level_name}: {actual_cls_shape}, Ожидалось: {expected_cls_shape}")
            assert actual_cls_shape == expected_cls_shape, f"ОШИБКА ФОРМЫ для cls_{level_name}"

        print("\n[SUCCESS] Сборка модели и проверка форм выходов прошли успешно!")

    except Exception as e:
        print(f"\n[ERROR] Ошибка при сборке или тестировании модели: {e}")
        import traceback

        traceback.print_exc()