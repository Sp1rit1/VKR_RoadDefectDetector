import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Input, Lambda
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
import math


class DetectorModel(Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        # loss_fn НЕ будет устанавливаться через compile, мы передадим его сами
        self.loss_calculator = None

        # Наши кастомные метрики, которые мы контролируем
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    # ИЗМЕНЕНИЕ: compile теперь принимает наш кастомный loss
    def compile(self, optimizer, loss_fn, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        # Мы сохраняем нашу функцию потерь в атрибут класса
        self.loss_calculator = loss_fn
        # ВАЖНО: мы не передаем 'loss' в super().compile()

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

    def test_step(self, data):
        images, y_true = data
        y_pred = self(images, training=False)
        # ИЗМЕНЕНИЕ: используем наш сохраненный loss_calculator
        loss = self.loss_calculator(y_true, y_pred)

        self.val_loss_tracker.update_state(loss)
        # Возвращаем словарь с ключом, совпадающим с именем метрики
        return {"loss": self.val_loss_tracker.result()}


# --- Вспомогательные функции для построения модели ---

def build_retinanet_head(fpn_filters, num_anchors, num_classes, level_name):
    """
    Создает общую "голову" RetinaNet и отдельные ветки для классификации и регрессии.
    """
    input_tensor = Input(shape=(None, None, fpn_filters), name=f'head_input_{level_name}')

    shared_head = input_tensor
    for i in range(4):
        shared_head = Conv2D(fpn_filters, 3, 1, 'same', kernel_initializer='he_normal',
                             name=f'{level_name}_head_shared_conv_{i}')(shared_head)
        shared_head = BatchNormalization(name=f'{level_name}_head_shared_bn_{i}')(shared_head)
        shared_head = ReLU(name=f'{level_name}_head_shared_relu_{i}')(shared_head)

    cls_head = Conv2D(num_anchors * num_classes, 3, 1, 'same', kernel_initializer='he_normal',
                      bias_initializer=tf.keras.initializers.Constant(math.log((1 - 0.01) / 0.01)),
                      name=f'{level_name}_cls_conv')(shared_head)
    cls_output = Lambda(
        lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_anchors, num_classes]),
        name=f'{level_name}_cls_output')(cls_head)

    reg_head = Conv2D(num_anchors * 4, 3, 1, 'same', kernel_initializer='he_normal', name=f'{level_name}_reg_conv')(
        shared_head)
    reg_output = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_anchors, 4]),
                        name=f'{level_name}_reg_output')(reg_head)

    head_model = Model(inputs=input_tensor, outputs=[reg_output, cls_output], name=f'prediction_head_{level_name}')
    return head_model


def build_detector_v3_standard(config):
    """
    Строит стандартную модель детектора с EfficientNetB0, FPN и
    ОБЩИМИ (shared) головами предсказаний в стиле RetinaNet.
    Возвращает кастомную модель с переопределенным train_step.
    """
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    fpn_filters = config['fpn_filters']
    num_anchors = config['num_anchors_per_level']
    freeze_backbone = config.get('freeze_backbone', True)

    input_tensor = Input(shape=input_shape, name='input_image')

    base_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet')
    if freeze_backbone:
        base_model.trainable = False
        print("EfficientNetB0 backbone is FROZEN.")
    else:
        base_model.trainable = True
        print("EfficientNetB0 backbone is TRAINABLE (fine-tuning mode).")

    c3_layer_name, c4_layer_name, c5_layer_name = 'block3b_add', 'block4c_add', 'block6d_add'

    c3_output = base_model.get_layer(c3_layer_name).output
    c4_output = base_model.get_layer(c4_layer_name).output
    c5_output = base_model.get_layer(c5_layer_name).output

    c3_lateral = Conv2D(fpn_filters, 1, 1, 'same', name='fpn_c3_lateral', kernel_initializer='he_normal')(c3_output)
    c4_lateral = Conv2D(fpn_filters, 1, 1, 'same', name='fpn_c4_lateral', kernel_initializer='he_normal')(c4_output)
    c5_lateral = Conv2D(fpn_filters, 1, 1, 'same', name='fpn_c5_lateral', kernel_initializer='he_normal')(c5_output)

    p5_top_down = c5_lateral
    p4_top_down = Add(name='fpn_p4_add')([UpSampling2D(2, name='fpn_p5_upsample')(p5_top_down), c4_lateral])
    p3_top_down = Add(name='fpn_p3_add')([UpSampling2D(2, name='fpn_p4_upsample')(p4_top_down), c3_lateral])

    p3 = Conv2D(fpn_filters, 3, 1, 'same', name='fpn_p3_output', kernel_initializer='he_normal')(p3_top_down)
    p4 = Conv2D(fpn_filters, 3, 1, 'same', name='fpn_p4_output', kernel_initializer='he_normal')(p4_top_down)
    p5 = Conv2D(fpn_filters, 3, 1, 'same', name='fpn_p5_output', kernel_initializer='he_normal')(p5_top_down)

    fpn_outputs = [p3, p4, p5]
    print("FPN neck построен.")

    head_p3 = build_retinanet_head(fpn_filters, num_anchors, num_classes, 'P3')
    head_p4 = build_retinanet_head(fpn_filters, num_anchors, num_classes, 'P4')
    head_p5 = build_retinanet_head(fpn_filters, num_anchors, num_classes, 'P5')

    reg_p3, cls_p3 = head_p3(p3)
    reg_p4, cls_p4 = head_p4(p4)
    reg_p5, cls_p5 = head_p5(p5)
    print("Общие (shared) головы предсказаний построены.")

    outputs = [reg_p3, reg_p4, reg_p5, cls_p3, cls_p4, cls_p5]

    model = DetectorModel(inputs=input_tensor, outputs=outputs, name=config.get('model_name', 'RetinaNetDetector'))
    print(f"Модель '{model.name}' (с кастомным train_step) успешно собрана.")
    return model


# --- Тестовый блок для проверки сборки ---
if __name__ == '__main__':
    print("--- Тестирование сборки модели ---")

    test_config = {
        'input_shape': [512, 512, 3],
        'num_classes': 2,
        'fpn_filters': 256,
        'num_anchors_per_level': 21,  # <--- Актуальное значение
        'freeze_backbone': True,
        'model_name': 'Test_DetectorV3_RetinaNet'
    }

    try:
        model = build_detector_v3_standard(test_config)
        model.summary(line_length=150)

        BATCH_SIZE = 2
        dummy_input = tf.random.normal((BATCH_SIZE, *test_config['input_shape']))
        outputs = model(dummy_input)

        print("\n--- Проверка форм выходных тензоров ---")
        strides = [8, 16, 32]
        num_levels = 3
        for i in range(num_levels):
            level_name = f"P{i + 3}"
            H, W = test_config['input_shape'][0] // strides[i], test_config['input_shape'][1] // strides[i]

            # Проверяем регрессионный выход
            expected_reg_shape = (BATCH_SIZE, H, W, test_config['num_anchors_per_level'], 4)
            actual_reg_shape = tuple(outputs[i].shape)
            print(f"Выход reg_{level_name}: {actual_reg_shape}, Ожидалось: {expected_reg_shape}")
            assert actual_reg_shape == expected_reg_shape, f"ОШИБКА ФОРМЫ для reg_{level_name}"

            # Проверяем классификационный выход
            expected_cls_shape = (BATCH_SIZE, H, W, test_config['num_anchors_per_level'], test_config['num_classes'])
            actual_cls_shape = tuple(outputs[i + num_levels].shape)
            print(f"Выход cls_{level_name}: {actual_cls_shape}, Ожидалось: {expected_cls_shape}")
            assert actual_cls_shape == expected_cls_shape, f"ОШИБКА ФОРМЫ для cls_{level_name}"

        print("\n[SUCCESS] Сборка модели и проверка форм выходов прошли успешно!")

    except Exception as e:
        print(f"\n[ERROR] Ошибка при сборке или тестировании модели: {e}")
        import traceback

        traceback.print_exc()