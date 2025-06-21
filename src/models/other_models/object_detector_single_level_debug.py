# src/models/object_detector_single_level_debug.py
import tensorflow as tf
import yaml
import os
from pathlib import Path

# --- Загрузка ОТЛАДОЧНОЙ Конфигурации ---
_current_script_dir = Path(__file__).resolve().parent  # src/models/
_project_root_dir = _current_script_dir.parent.parent  # Корень проекта
_debug_config_path = _project_root_dir / 'src' / 'configs' / 'detector_config_single_level_debug.yaml'

DEBUG_MODEL_CONFIG = {}
try:
    with open(_debug_config_path, 'r', encoding='utf-8') as f:
        DEBUG_MODEL_CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"ОШИБКА (object_detector_single_level_debug.py): Не удалось загрузить {_debug_config_path}: {e}")
    # Минимальные дефолты, чтобы функция могла быть определена
    DEBUG_MODEL_CONFIG.setdefault('fpn_detector_params', {}).setdefault('input_shape', [416, 416, 3])
    DEBUG_MODEL_CONFIG['fpn_detector_params'].setdefault('backbone_name', 'MobileNetV2')
    DEBUG_MODEL_CONFIG['fpn_detector_params'].setdefault('num_classes', 2)
    DEBUG_MODEL_CONFIG['fpn_detector_params'].setdefault('head_config', {'head_depth': 2, 'head_conv_filters': 128,
                                                                         'leaky_relu_alpha': 0.1,
                                                                         'l2_regularization': None})
    DEBUG_MODEL_CONFIG['fpn_detector_params'].setdefault('detector_fpn_anchor_configs',
                                                         {'P4_debug': {'num_anchors_this_level': 1}})
    DEBUG_MODEL_CONFIG.setdefault('unfreeze_backbone', False)  # Важно для начального состояния

_fpn_params_model_debug = DEBUG_MODEL_CONFIG.get('fpn_detector_params', {})
INPUT_SHAPE_MODEL_DBG = tuple(_fpn_params_model_debug.get('input_shape'))
BACKBONE_NAME_MODEL_DBG = _fpn_params_model_debug.get('backbone_name')
NUM_CLASSES_MODEL_DBG = _fpn_params_model_debug.get('num_classes')
HEAD_CONFIG_DBG = _fpn_params_model_debug.get('head_config', {})
HEAD_DEPTH_DBG = HEAD_CONFIG_DBG.get('head_depth')
HEAD_CONV_FILTERS_DBG = HEAD_CONFIG_DBG.get('head_conv_filters')
LEAKY_RELU_ALPHA_DBG = HEAD_CONFIG_DBG.get('leaky_relu_alpha')
L2_REG_VALUE_DBG = HEAD_CONFIG_DBG.get('l2_regularization')
L2_REGULARIZER_DBG = tf.keras.regularizers.l2(L2_REG_VALUE_DBG) if L2_REG_VALUE_DBG else None

# Параметры для единственного уровня P4_debug
_p4_anchor_cfg = _fpn_params_model_debug.get('detector_fpn_anchor_configs', {}).get('P4_debug', {})
NUM_ANCHORS_P4_DBG = _p4_anchor_cfg.get('num_anchors_this_level', 1)

# Используем общий флаг unfreeze_backbone для определения начального состояния заморозки
FREEZE_BACKBONE_INITIALLY_DBG = not DEBUG_MODEL_CONFIG.get('unfreeze_backbone', False)


def conv_bn_leaky_debug(x, filters, kernel_size, strides=1, name_prefix=""):  # Свой conv_bn_leaky для этого файла
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_regularizer=L2_REGULARIZER_DBG, name=f"{name_prefix}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA_DBG, name=f"{name_prefix}_leakyrelu")(x)
    return x


def build_detector_single_level_p4_debug():
    inputs = tf.keras.Input(shape=INPUT_SHAPE_MODEL_DBG, name="input_image_single_level_debug")

    if BACKBONE_NAME_MODEL_DBG == 'MobileNetV2':
        base_model_loader = tf.keras.applications.MobileNetV2(
            input_tensor=inputs, include_top=False, weights='imagenet'
        )
        # Выход для P4 (stride 16)
        feature_map_output_name = 'block_13_expand_relu'
        try:
            feature_map = base_model_loader.get_layer(feature_map_output_name).output
        except ValueError as e_bb:
            print(f"ОШИБКА (debug_model): Не найден слой '{feature_map_output_name}' в MobileNetV2. {e_bb}")
            base_model_loader.summary()  # Выведем summary для помощи
            raise e_bb
    else:
        raise ValueError(f"Неподдерживаемый backbone для отладки: {BACKBONE_NAME_MODEL_DBG}")

    # Создаем feature_extractor, чтобы управлять его trainable свойством
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=feature_map,
                                       name=f"Backbone_{BACKBONE_NAME_MODEL_DBG}_P4_Only")
    feature_extractor.trainable = not FREEZE_BACKBONE_INITIALLY_DBG  # Управляется 'unfreeze_backbone' из конфига
    print(f"INFO (Debug Model): Backbone '{feature_extractor.name}' ЗАМОРОЖЕН: {feature_extractor.trainable == False}")

    # Применяем feature_extractor (он уже содержит input_tensor от base_model_loader)
    # Нет, нужно передать `inputs` нашей основной модели
    x = feature_extractor(inputs)  # Это даст нам карту признаков от нужного слоя backbone

    # "Тело" головы (несколько сверток)
    for i in range(HEAD_DEPTH_DBG):
        x = conv_bn_leaky_debug(x, filters=HEAD_CONV_FILTERS_DBG, kernel_size=3,
                                name_prefix=f"head_P4_body_conv{i + 1}")

    # Финальный предсказывающий слой
    num_predictions_per_anchor = 4 + 1 + NUM_CLASSES_MODEL_DBG  # 4 coords + 1 obj + N classes
    total_output_filters = NUM_ANCHORS_P4_DBG * num_predictions_per_anchor  # NUM_ANCHORS_P4_DBG должен быть 1

    raw_predictions = tf.keras.layers.Conv2D(
        filters=total_output_filters, kernel_size=1, padding='same', activation=None,
        kernel_regularizer=L2_REGULARIZER_DBG, name="head_P4_raw_preds"
    )(x)

    # Reshape выхода
    grid_h_p4_debug = INPUT_SHAPE_MODEL_DBG[0] // 16  # Страйд для P4
    grid_w_p4_debug = INPUT_SHAPE_MODEL_DBG[1] // 16
    output_shape_tuple = (grid_h_p4_debug, grid_w_p4_debug, NUM_ANCHORS_P4_DBG, num_predictions_per_anchor)

    reshaped_predictions = tf.keras.layers.Reshape(output_shape_tuple, name="head_P4_predictions")(raw_predictions)

    model = tf.keras.Model(inputs=inputs, outputs=reshaped_predictions, name="Single_Level_P4_Detector_Debug")
    return model


if __name__ == '__main__':
    print("--- Тестирование object_detector_single_level_debug.py ---")
    model_test = build_detector_single_level_p4_debug()
    model_test.summary(line_length=120)
    dummy_input = tf.random.normal((1, INPUT_SHAPE_MODEL_DBG[0], INPUT_SHAPE_MODEL_DBG[1], INPUT_SHAPE_MODEL_DBG[2]))
    output = model_test(dummy_input)
    print(f"Форма выхода модели: {output.shape}")
    expected_shape = (1, INPUT_SHAPE_MODEL_DBG[0] // 16, INPUT_SHAPE_MODEL_DBG[1] // 16, NUM_ANCHORS_P4_DBG,
                      4 + 1 + NUM_CLASSES_MODEL_DBG)
    print(f"Ожидаемая форма: {expected_shape}")
    assert output.shape.as_list() == list(expected_shape), "Ошибка формы выхода!"
    print("Тест формы выхода пройден.")