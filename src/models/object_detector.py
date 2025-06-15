# src/models/object_detector.py
import tensorflow as tf
import yaml
import os

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
# _base_config_path = os.path.join(_current_dir, '..', 'configs', 'base_config.yaml') # Загрузим, если понадобится

DETECTOR_CONFIG = {}
# BASE_CONFIG = {}

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    # with open(_base_config_path, 'r', encoding='utf-8') as f:
    #     BASE_CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"ПРЕДУПРЕЖДЕНИЕ (object_detector.py): Не удалось загрузить конфиги: {e}. Используются дефолты.")
    DETECTOR_CONFIG.setdefault('backbone_name', 'MobileNetV2')
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('num_classes', 2)
    DETECTOR_CONFIG.setdefault('num_anchors_per_location', 6)  # Обновлено на 6 якорей
    DETECTOR_CONFIG.setdefault('head_conv_filters', 256)
    DETECTOR_CONFIG.setdefault('head_body_depth', 2)  # Глубина "тела" головы (количество Conv блоков)
    DETECTOR_CONFIG.setdefault('head_final_conv_filters',
                               256)  # Фильтры в финальных свертках головы (перед предсказанием)
    DETECTOR_CONFIG.setdefault('leaky_relu_alpha', 0.1)
    DETECTOR_CONFIG.setdefault('l2_regularization', 5e-4)  # Значение для L2 регуляризации

# --- Параметры из Конфига (с дефолтами для безопасности) ---
INPUT_SHAPE_CFG = tuple(DETECTOR_CONFIG.get('input_shape', [416, 416, 3]))
BACKBONE_NAME_CFG = DETECTOR_CONFIG.get('backbone_name', 'MobileNetV2')
# freeze_backbone_initial_train из detector_config теперь не используется здесь напрямую,
# build_object_detector... всегда создает с замороженным backbone по умолчанию.
BACKBONE_LAYER_NAME_IN_MODEL_CFG = DETECTOR_CONFIG.get('backbone_layer_name_in_model', 'Backbone_MobileNetV2')

NUM_CLASSES_CFG = DETECTOR_CONFIG.get('num_classes', 2)
NUM_ANCHORS_CFG = DETECTOR_CONFIG.get('num_anchors_per_location', 6)  # Соответствует 6 якорям

HEAD_BODY_CONV_FILTERS_CFG = DETECTOR_CONFIG.get('head_body_conv_filters', 256)  # Фильтры в "теле" головы
HEAD_BODY_DEPTH_CFG = DETECTOR_CONFIG.get('head_body_depth', 2)  # Количество Conv-блоков в "теле" головы
# HEAD_FINAL_CONV_FILTERS_CFG = DETECTOR_CONFIG.get('head_final_conv_filters', 256) # Фильтры в слоях непосредственно перед выходом (если бы они были)

LEAKY_RELU_ALPHA_CFG = DETECTOR_CONFIG.get('leaky_relu_alpha', 0.1)
L2_REG_FACTOR_CFG = DETECTOR_CONFIG.get('l2_regularization', 5e-4)  # 0.0005 - стандартное значение


def _conv_bn_leaky(x, filters, kernel_size, strides=1, name_prefix=None, l2_reg=L2_REG_FACTOR_CFG,
                   alpha=LEAKY_RELU_ALPHA_CFG):
    """Стандартный сверточный блок: Conv2D -> BatchNormalization -> LeakyReLU с L2 регуляризацией."""
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding='same', use_bias=False,  # BatchNormalization имеет параметр смещения (beta)
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),  # L2 регуляризация
        name=f"{name_prefix}_conv" if name_prefix else None
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn" if name_prefix else None)(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha, name=f"{name_prefix}_leaky" if name_prefix else None)(x)
    return x


def build_object_detector_v1_enhanced():  # Новое имя функции
    """
    Строит улучшенную версию v1 детектора объектов.
    Backbone (MobileNetV2) -> Несколько настраиваемых сверточных слоев ("тело" головы) -> Голова предсказаний.
    Backbone по умолчанию создается ЗАМОРОЖЕННЫМ.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE_CFG, name="input_image_detector_v1_enhanced")

    # 1. Backbone
    if BACKBONE_NAME_CFG == 'MobileNetV2':
        base_model_keras_app = tf.keras.applications.MobileNetV2(
            input_shape=INPUT_SHAPE_CFG,  # Передаем форму входа из конфига
            include_top=False,
            weights='imagenet'
        )
        # Устанавливаем backbone как НЕОБУЧАЕМЫЙ по умолчанию при создании
        base_model_keras_app.trainable = False
        print(
            f"INFO (build_object_detector_v1_enhanced): Backbone '{BACKBONE_NAME_CFG}' инициализирован как ЗАМОРОЖЕННЫЙ.")

        # Выбираем выходной слой из backbone
        # Для MobileNetV2 с входом 416x416, 'block_13_expand_relu' (stride 16) дает карту 26x26x576 (если alpha=1.0)
        # Для MobileNetV2 с alpha=1.0, `block_13_expand_relu` (если считать с 1) это слой с индексом 116 (если include_top=False).
        # Но имена более надежны, если они не меняются.
        try:
            feature_extraction_layer_name = 'block_13_expand_relu'  # Stride 16
            backbone_output_tensor = base_model_keras_app.get_layer(feature_extraction_layer_name).output
            print(
                f"INFO: Используется выход слоя '{feature_extraction_layer_name}' из {BACKBONE_NAME_CFG} (ожидаемый stride 16).")
        except ValueError:
            print(
                f"WARNING: Слой '{feature_extraction_layer_name}' не найден в {BACKBONE_NAME_CFG} для input_shape {INPUT_SHAPE_CFG}.")
            print("         Будет использован последний выходной слой базовой модели (может иметь другой stride).")
            backbone_output_tensor = base_model_keras_app.output

        # Оборачиваем в tf.keras.Model, чтобы имя слоя было консистентным для train_detector.py
        backbone = tf.keras.Model(inputs=base_model_keras_app.input, outputs=backbone_output_tensor,
                                  name=BACKBONE_LAYER_NAME_IN_MODEL_CFG)

    else:
        raise ValueError(f"Неподдерживаемый backbone: {BACKBONE_NAME_CFG}")

    # Применяем backbone к нашим входам
    # Флаг training для backbone будет управляться его свойством .trainable
    # Если backbone.trainable = False, Keras автоматически передаст training=False его слоям.
    x = backbone(inputs)

    # 2. "Тело" Головы (Head Body) - несколько сверточных слоев для обработки признаков из backbone
    # Количество фильтров и глубина настраиваются из конфига.
    current_filters = HEAD_BODY_CONV_FILTERS_CFG
    for i in range(HEAD_BODY_DEPTH_CFG):
        x = _conv_bn_leaky(x, filters=current_filters, kernel_size=3, name_prefix=f"head_body_conv_block{i + 1}")
        if i < HEAD_BODY_DEPTH_CFG - 1:  # Уменьшаем фильтры в промежуточных слоях тела, если их несколько
            if current_filters > 128:  # Не делаем слишком мало фильтров
                current_filters //= 2

    # 3. Финальный Слой Предсказаний (Prediction Layer)
    num_predictions_per_anchor = 4 + 1 + NUM_CLASSES_CFG  # box_xywh, objectness, class_probs
    total_output_filters = NUM_ANCHORS_CFG * num_predictions_per_anchor

    raw_head_output = tf.keras.layers.Conv2D(
        filters=total_output_filters,
        kernel_size=1,  # 1x1 свертка для генерации предсказаний для каждого якоря
        padding='same',
        activation=None,  # Логиты, активации (sigmoid) будут применяться в функции потерь или при инференсе
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG_FACTOR_CFG),
        name="detection_head_raw_predictions"
    )(x)

    # Решейпим выход в (batch_size, grid_height, grid_width, num_anchors, num_predictions_per_anchor)
    # grid_height и grid_width зависят от stride сети. Для stride 16 и входа 416x416 -> 26x26.
    # x.shape[1] и x.shape[2] должны дать нам эти размеры.
    if x.shape[1] is None or x.shape[2] is None:  # Если размеры не определены статически
        grid_h_runtime = tf.shape(x)[1]
        grid_w_runtime = tf.shape(x)[2]
        final_output_shape_runtime = (grid_h_runtime, grid_w_runtime, NUM_ANCHORS_CFG, num_predictions_per_anchor)
        # Для tf.keras.layers.Reshape лучше, если известны статические размеры, если возможно
        # Попробуем использовать известные из конфига, если они там есть и соответствуют stride
        expected_grid_h = INPUT_SHAPE_CFG[0] // 16
        expected_grid_w = INPUT_SHAPE_CFG[1] // 16
        final_output_shape_static = (expected_grid_h, expected_grid_w, NUM_ANCHORS_CFG, num_predictions_per_anchor)
        # print(f"DEBUG: Reshape target shape: {final_output_shape_static}")
        predictions_final_reshaped = tf.keras.layers.Reshape(
            final_output_shape_static, name="detector_predictions_reshaped"
        )(raw_head_output)
    else:  # Размеры известны статически
        final_output_shape_static = (x.shape[1], x.shape[2], NUM_ANCHORS_CFG, num_predictions_per_anchor)
        # print(f"DEBUG: Reshape target shape (static): {final_output_shape_static}")
        predictions_final_reshaped = tf.keras.layers.Reshape(
            final_output_shape_static, name="detector_predictions_reshaped"
        )(raw_head_output)

    model = tf.keras.Model(inputs=inputs, outputs=predictions_final_reshaped,
                           name=f"{DETECTOR_CONFIG.get('model_base_name', 'Detector')}_Enhanced_v1")
    return model


if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Версия v1 Enhanced) ---")
    print(f"Конфигурация входа: {INPUT_SHAPE_CFG}")
    print(f"Backbone: {BACKBONE_NAME_CFG}")
    print(f"Фильтры в теле головы: {HEAD_BODY_CONV_FILTERS_CFG}, Глубина тела головы: {HEAD_BODY_DEPTH_CFG}")
    print(f"Количество классов: {NUM_CLASSES_CFG}, Количество якорей на ячейку: {NUM_ANCHORS_CFG}")
    print(f"L2 Regularization Factor: {L2_REG_FACTOR_CFG}")

    try:
        detector_model_v1e = build_object_detector_v1_enhanced()
        if detector_model_v1e:
            print("\nСтруктура модели детектора (v1 Enhanced):")
            detector_model_v1e.summary(line_length=150)

            print("\nТестовый прогон модели (проверка формы выхода):")
            batch_size_for_test = DETECTOR_CONFIG.get('batch_size', 1)  # Из основного конфига, а не train_params

            dummy_input_shape_test = (batch_size_for_test, INPUT_SHAPE_CFG[0], INPUT_SHAPE_CFG[1], INPUT_SHAPE_CFG[2])
            dummy_input = tf.random.normal(dummy_input_shape_test)
            print(f"  Создан фиктивный входной тензор с формой: {dummy_input.shape}")

            predictions = detector_model_v1e(dummy_input)

            expected_grid_h = INPUT_SHAPE_CFG[0] // 16
            expected_grid_w = INPUT_SHAPE_CFG[1] // 16
            expected_output_features_per_anchor = 4 + 1 + NUM_CLASSES_CFG

            print(f"  Форма предсказаний: {predictions.shape}")
            print(
                f"  Ожидаемая форма: ({batch_size_for_test}, {expected_grid_h}, {expected_grid_w}, {NUM_ANCHORS_CFG}, {expected_output_features_per_anchor})")

            if predictions.shape == (batch_size_for_test, expected_grid_h, expected_grid_w, NUM_ANCHORS_CFG,
                                     expected_output_features_per_anchor):
                print("  Форма выхода СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
            else:
                print("  ОШИБКА: Форма выхода НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
        else:
            print("Модель v1 Enhanced не была создана.")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели детектора v1 Enhanced: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py (v1 Enhanced) завершено ---")