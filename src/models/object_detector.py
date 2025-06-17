# src/models/object_detector.py
import tensorflow as tf
import yaml
import os
import numpy as np
import time

# --- Загрузка Конфигурации ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

DETECTOR_CONFIG = {}
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}
except Exception as e:
    print(f"ОШИБКА (object_detector.py): Не удалось загрузить detector_config.yaml: {e}. Используются дефолты.")
    DETECTOR_CONFIG.setdefault('fpn_detector_params', {
        'backbone_name': 'MobileNetV2', 'input_shape': [416, 416, 3], 'num_classes': 2,
        'detector_fpn_levels': ['P3', 'P4', 'P5'],
        'detector_fpn_strides': {'P3': 8, 'P4': 16, 'P5': 32},
        'detector_fpn_anchor_configs': {
            'P3': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3},
            'P4': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3},
            'P5': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3}
        },
        'head_config': {'fpn_filters': 128, 'head_depth': 2, 'head_conv_filters': 128, 'leaky_relu_alpha': 0.1,
                        'l2_regularization': None}
    })
    DETECTOR_CONFIG.setdefault('freeze_backbone', True)  # Общий флаг для начального обучения

# --- Параметры из Конфига для FPN Модели ---
FPN_PARAMS = DETECTOR_CONFIG.get('fpn_detector_params', {})
INPUT_SHAPE_MODEL_CFG = tuple(FPN_PARAMS.get('input_shape', [416, 416, 3]))
BACKBONE_NAME_MODEL_CFG = FPN_PARAMS.get('backbone_name', 'MobileNetV2')
# Используем freeze_backbone из общей секции для начального состояния модели
FREEZE_BACKBONE_INITIAL_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)

FPN_HEAD_CONFIG = FPN_PARAMS.get('head_config', {})
FPN_FILTERS_CFG = FPN_HEAD_CONFIG.get('fpn_filters', 256)  # Увеличим дефолт, если не найден
HEAD_DEPTH_CFG = FPN_HEAD_CONFIG.get('head_depth', 2)
HEAD_CONV_FILTERS_CFG = FPN_HEAD_CONFIG.get('head_conv_filters', 256)  # Увеличим дефолт
LEAKY_RELU_ALPHA_CFG = FPN_HEAD_CONFIG.get('leaky_relu_alpha', 0.1)
L2_REG_VALUE_CFG = FPN_HEAD_CONFIG.get('l2_regularization', None)
L2_REGULARIZER = tf.keras.regularizers.l2(
    L2_REG_VALUE_CFG) if L2_REG_VALUE_CFG is not None and L2_REG_VALUE_CFG > 0 else None

NUM_CLASSES_MODEL_CFG = FPN_PARAMS.get('num_classes', 2)
FPN_ANCHOR_CONFIGS_YAML_MODEL = FPN_PARAMS.get('detector_fpn_anchor_configs', {})
FPN_LEVEL_NAMES_MODEL = FPN_PARAMS.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
FPN_STRIDES_MODEL = FPN_PARAMS.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})

# Получаем количество якорей для каждого уровня ИЗ КОНФИГА
P3_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P3', {}).get('num_anchors_this_level', 3)
P4_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P4', {}).get('num_anchors_this_level', 3)
P5_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P5', {}).get('num_anchors_this_level', 3)


def conv_bn_leaky(x, filters, kernel_size, strides=1, name_prefix=""):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_regularizer=L2_REGULARIZER, name=f"{name_prefix}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA_CFG, name=f"{name_prefix}_leakyrelu")(x)
    return x


# --- ИСПРАВЛЕНА СИГНАТУРА: добавлены known_grid_h, known_grid_w ---
def prediction_head(input_tensor, head_id_str,
                    known_grid_h, known_grid_w,  # Размеры сетки для этого уровня
                    num_anchors_at_this_level, num_classes,
                    head_depth, head_filters):
    """Создает голову предсказаний для одного уровня FPN."""
    x = input_tensor
    for i in range(head_depth):
        x = conv_bn_leaky(x, filters=head_filters, kernel_size=3, name_prefix=f"head_{head_id_str}_body_conv{i + 1}")

    num_predictions_per_anchor = 4 + 1 + num_classes
    raw_predictions = tf.keras.layers.Conv2D(
        filters=num_anchors_at_this_level * num_predictions_per_anchor, kernel_size=1,
        padding='same', activation=None,
        kernel_regularizer=L2_REGULARIZER,
        name=f"head_{head_id_str}_raw_preds"
    )(x)

    # --- ИСПРАВЛЕН RESHAPE: используем известные known_grid_h, known_grid_w ---
    output_shape_tuple = (known_grid_h, known_grid_w, num_anchors_at_this_level, num_predictions_per_anchor)
    reshaped_predictions = tf.keras.layers.Reshape(output_shape_tuple, name=f"head_{head_id_str}_predictions")(
        raw_predictions)
    return reshaped_predictions


def build_object_detector_v2_fpn():  # Имя функции совпадает с твоим резюме
    inputs = tf.keras.Input(shape=INPUT_SHAPE_MODEL_CFG, name="input_image_detector_fpn")

    # 1. Backbone
    if BACKBONE_NAME_MODEL_CFG == 'MobileNetV2':
        # Сначала создаем базовую модель с input_shape
        base_model_loader = tf.keras.applications.MobileNetV2(
            input_shape=INPUT_SHAPE_MODEL_CFG,  # Передаем input_shape сюда
            include_top=False,
            weights='imagenet'
        )
        # Имена слоев для MobileNetV2 (убедись, что они верны для твоего input_shape)
        c3_output_layer_name = 'block_6_expand_relu'  # Stride 8
        c4_output_layer_name = 'block_13_expand_relu'  # Stride 16
        c5_output_layer_name = 'out_relu'  # Stride 32 (последний перед возможным пулингом)
    else:
        raise ValueError(f"Неподдерживаемый backbone для FPN: {BACKBONE_NAME_MODEL_CFG}")

    try:
        c3_out_tensor = base_model_loader.get_layer(c3_output_layer_name).output
        c4_out_tensor = base_model_loader.get_layer(c4_output_layer_name).output
        c5_out_tensor = base_model_loader.get_layer(c5_output_layer_name).output
        print(
            f"INFO: Слои для FPN из {BACKBONE_NAME_MODEL_CFG}: C3='{c3_output_layer_name}', C4='{c4_output_layer_name}', C5='{c5_output_layer_name}'")
    except ValueError as e:
        print(
            f"ОШИБКА: Не удалось найти один из слоев backbone ('{c3_output_layer_name}', '{c4_output_layer_name}', '{c5_output_layer_name}') для {BACKBONE_NAME_MODEL_CFG}.")
        print("Вывод summary базовой модели для проверки доступных имен слоев:")
        base_model_loader.summary(line_length=150)
        raise e

    # Создаем feature_extractor модель, которая будет принимать 'inputs' нашей основной модели
    feature_extractor = tf.keras.Model(inputs=base_model_loader.input,
                                       outputs=[c3_out_tensor, c4_out_tensor, c5_out_tensor],
                                       name=f"Backbone_{BACKBONE_NAME_MODEL_CFG}_FPN_Features")

    # Управление заморозкой backbone
    if FREEZE_BACKBONE_INITIAL_CFG:  # Используем этот флаг из конфига
        feature_extractor.trainable = False  # Замораживаем весь feature_extractor (включая base_model_loader)
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' ЗАМОРОЖЕН.")
    else:
        feature_extractor.trainable = True
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' РАЗМОРОЖЕН.")

    c3_features, c4_features, c5_features = feature_extractor(inputs)

    # FPN - Шея
    p5_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c5_to_p5_in',
                                   kernel_regularizer=L2_REGULARIZER)(c5_features)
    p4_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c4_to_p4_in',
                                   kernel_regularizer=L2_REGULARIZER)(c4_features)
    p3_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c3_to_p3_in',
                                   kernel_regularizer=L2_REGULARIZER)(c3_features)

    p5_up = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p5_upsampled')(p5_in)
    p4_merged = tf.keras.layers.Add(name='fpn_p4_merged')([p5_up, p4_in])

    p4_up = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p4_upsampled')(p4_merged)
    p3_merged = tf.keras.layers.Add(name='fpn_p3_merged')([p4_up, p3_in])

    p5_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p5_output',
                                       kernel_regularizer=L2_REGULARIZER)(p5_in)
    p4_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p4_output',
                                       kernel_regularizer=L2_REGULARIZER)(p4_merged)
    p3_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p3_output',
                                       kernel_regularizer=L2_REGULARIZER)(p3_merged)

    # Prediction Heads
    # Размеры сетки вычисляются на основе страйдов из конфига
    grid_h_p3 = INPUT_SHAPE_MODEL_CFG[0] // FPN_STRIDES_MODEL.get('P3', 8)
    grid_w_p3 = INPUT_SHAPE_MODEL_CFG[1] // FPN_STRIDES_MODEL.get('P3', 8)
    grid_h_p4 = INPUT_SHAPE_MODEL_CFG[0] // FPN_STRIDES_MODEL.get('P4', 16)
    grid_w_p4 = INPUT_SHAPE_MODEL_CFG[1] // FPN_STRIDES_MODEL.get('P4', 16)
    grid_h_p5 = INPUT_SHAPE_MODEL_CFG[0] // FPN_STRIDES_MODEL.get('P5', 32)
    grid_w_p5 = INPUT_SHAPE_MODEL_CFG[1] // FPN_STRIDES_MODEL.get('P5', 32)

    # --- ИСПРАВЛЕН ВЫЗОВ: передаем grid_h, grid_w ---
    predictions_p3 = prediction_head(p3_output, "P3", grid_h_p3, grid_w_p3, P3_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)
    predictions_p4 = prediction_head(p4_output, "P4", grid_h_p4, grid_w_p4, P4_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)
    predictions_p5 = prediction_head(p5_output, "P5", grid_h_p5, grid_w_p5, P5_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)

    model_outputs_list = [predictions_p3, predictions_p4, predictions_p5]

    final_model_name = FPN_PARAMS.get('model_name_prefix', f"{BACKBONE_NAME_MODEL_CFG}_FPN_Detector")
    final_model = tf.keras.Model(inputs=inputs, outputs=model_outputs_list, name=final_model_name)
    return final_model


# --- Блок для тестирования этого файла (if __name__ == '__main__':) ---
if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Версия FPN) ---")
    # Загружаем конфиг еще раз локально для теста, чтобы убедиться, что он читается
    _test_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')
    TEST_DETECTOR_CONFIG_MAIN = {}
    try:
        with open(_test_detector_config_path, 'r', encoding='utf-8') as f:
            TEST_DETECTOR_CONFIG_MAIN = yaml.safe_load(f)
    except Exception as e_test_cfg:
        print(f"Ошибка загрузки detector_config.yaml для теста __main__: {e_test_cfg}")
        # Используем глобальные переменные, если локальная загрузка не удалась
        test_input_shape = INPUT_SHAPE_MODEL_CFG
        test_num_classes = NUM_CLASSES_MODEL_CFG
        test_p3_anchors = P3_ANCHORS_PER_LOC
        test_p4_anchors = P4_ANCHORS_PER_LOC
        test_p5_anchors = P5_ANCHORS_PER_LOC
        test_strides = FPN_STRIDES_MODEL
    else:
        test_fpn_params = TEST_DETECTOR_CONFIG_MAIN.get('fpn_detector_params', {})
        test_input_shape = tuple(test_fpn_params.get('input_shape', [416, 416, 3]))
        test_num_classes = test_fpn_params.get('num_classes', 2)
        test_fpn_anchor_cfgs = test_fpn_params.get('detector_fpn_anchor_configs', {})
        test_p3_anchors = test_fpn_anchor_cfgs.get('P3', {}).get('num_anchors_this_level', 3)
        test_p4_anchors = test_fpn_anchor_cfgs.get('P4', {}).get('num_anchors_this_level', 3)
        test_p5_anchors = test_fpn_anchor_cfgs.get('P5', {}).get('num_anchors_this_level', 3)
        test_strides = test_fpn_params.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})

    print(f"Конфигурация входа для теста __main__: {test_input_shape}")
    print(f"Количество классов: {test_num_classes}")
    print(f"Якоря для P3: {test_p3_anchors}, P4: {test_p4_anchors}, P5: {test_p5_anchors}")

    try:
        # При вызове build_object_detector_v2_fpn() он будет использовать глобальные переменные,
        # которые уже были инициализированы из конфига в начале этого файла.
        detector_model_fpn = build_object_detector_v2_fpn()
        print("\nСтруктура модели детектора с FPN:");
        detector_model_fpn.summary(line_length=150)
        print("\nТестовый прогон модели FPN (проверка форм выходов):")
        batch_size_for_test = 1
        dummy_input_shape_test_fpn = (
        batch_size_for_test, test_input_shape[0], test_input_shape[1], test_input_shape[2])
        dummy_input_fpn = tf.random.normal(dummy_input_shape_test_fpn)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input_fpn.shape}")
        start_time = time.time();
        predictions_list = detector_model_fpn(dummy_input_fpn, training=False);
        end_time = time.time()
        print(f"  Время инференса на фиктивном входе: {end_time - start_time:.4f} сек.")

        if isinstance(predictions_list, list) and len(predictions_list) == 3:
            print(f"\n  Модель вернула список из {len(predictions_list)} тензоров (P3, P4, P5):")

            expected_preds_per_anchor = 4 + 1 + test_num_classes
            level_names_for_test = ["P3", "P4", "P5"]
            all_shapes_match = True

            for i, pred_tensor in enumerate(predictions_list):
                level_key = level_names_for_test[i]
                current_level_stride = test_strides.get(level_key, 8 * (2 ** i))  # Дефолтный страйд, если нет в конфиге
                current_level_anchors = test_fpn_anchor_cfgs.get(level_key, {}).get('num_anchors_this_level',
                                                                                    3)  # Дефолт 3 якоря

                expected_grid_h = test_input_shape[0] // current_level_stride
                expected_grid_w = test_input_shape[1] // current_level_stride
                expected_shape_tuple = (
                batch_size_for_test, expected_grid_h, expected_grid_w, current_level_anchors, expected_preds_per_anchor)

                print(
                    f"    Форма выхода {level_names_for_test[i]}: {pred_tensor.shape}, Ожидаемая: {expected_shape_tuple}")
                if pred_tensor.shape.as_list() != list(expected_shape_tuple):
                    all_shapes_match = False
                    print(f"      ОШИБКА: Форма выхода {level_names_for_test[i]} НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ!")

            if all_shapes_match:
                print("\n  Все формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
            else:
                print("\n  ОШИБКА: НЕ ВСЕ формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
        else:
            print(f"  ОШИБКА: Модель вернула не список из 3х тензоров, а: {type(predictions_list)}")
    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели FPN детектора: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py (FPN) завершено ---")