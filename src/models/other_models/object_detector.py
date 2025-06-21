# src/models/object_detector.py
import time

import tensorflow as tf
import yaml
import os
import numpy as np

# --- Загрузка Конфигурации (как в твоем object1.txt) ---
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
    # Устанавливаем МИНИМАЛЬНО НЕОБХОДИМЫЕ дефолты, чтобы функция могла быть определена
    DETECTOR_CONFIG.setdefault('fpn_detector_params', {
        'backbone_name': 'MobileNetV2', 'input_shape': [416, 416, 3], 'num_classes': 2,
        'detector_fpn_levels': ['P3', 'P4', 'P5'],
        'detector_fpn_strides': {'P3': 8, 'P4': 16, 'P5': 32},
        'detector_fpn_anchor_configs': {  # Обязательно с num_anchors_this_level
            'P3': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3},
            'P4': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3},
            'P5': {'num_anchors_this_level': 3, 'anchors_wh_normalized': [[0.1, 0.1]] * 3}
        },
        'head_config': {'fpn_filters': 128, 'head_depth': 2, 'head_conv_filters': 128, 'leaky_relu_alpha': 0.1,
                        'l2_regularization': None}
    })
    # Этот общий флаг freeze_backbone будет использоваться как FREEZE_BACKBONE_INITIAL_CFG
    DETECTOR_CONFIG.setdefault('freeze_backbone', True)

# --- Параметры из Конфига для FPN Модели ---
FPN_PARAMS = DETECTOR_CONFIG.get('fpn_detector_params', {})  # Основной блок для FPN
INPUT_SHAPE_MODEL_CFG = tuple(FPN_PARAMS.get('input_shape', [416, 416, 3]))
BACKBONE_NAME_MODEL_CFG = FPN_PARAMS.get('backbone_name', 'MobileNetV2')
# Используем общий флаг freeze_backbone из корня DETECTOR_CONFIG для начального состояния
FREEZE_BACKBONE_INITIAL_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)

FPN_HEAD_CONFIG = FPN_PARAMS.get('head_config', {})
FPN_FILTERS_CFG = FPN_HEAD_CONFIG.get('fpn_filters', 256)
HEAD_DEPTH_CFG = FPN_HEAD_CONFIG.get('head_depth', 2)
HEAD_CONV_FILTERS_CFG = FPN_HEAD_CONFIG.get('head_conv_filters', 256)
LEAKY_RELU_ALPHA_CFG = FPN_HEAD_CONFIG.get('leaky_relu_alpha', 0.1)
L2_REG_VALUE_CFG = FPN_HEAD_CONFIG.get('l2_regularization', None)  # Это значение для l2()
L2_REGULARIZER = tf.keras.regularizers.l2(
    L2_REG_VALUE_CFG) if L2_REG_VALUE_CFG is not None and L2_REG_VALUE_CFG > 0 else None

NUM_CLASSES_MODEL_CFG = FPN_PARAMS.get('num_classes', 2)  # Из fpn_detector_params
FPN_ANCHOR_CONFIGS_YAML_MODEL = FPN_PARAMS.get('detector_fpn_anchor_configs', {})
FPN_LEVEL_NAMES_MODEL = FPN_PARAMS.get('detector_fpn_levels', ['P3', 'P4', 'P5'])  # Порядок важен
FPN_STRIDES_MODEL = FPN_PARAMS.get('detector_fpn_strides', {'P3': 8, 'P4': 16, 'P5': 32})

# Получаем количество якорей для каждого уровня ИЗ КОНФИГА
# Используем дефолт 3, если в конфиге что-то не так, чтобы избежать ошибок далее
P3_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P3', {}).get('num_anchors_this_level', 3)
P4_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P4', {}).get('num_anchors_this_level', 3)
P5_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_YAML_MODEL.get('P5', {}).get('num_anchors_this_level', 3)
# Сохраним их в словарь для удобства
NUM_ANCHORS_PER_LEVEL_MAP = {'P3': P3_ANCHORS_PER_LOC, 'P4': P4_ANCHORS_PER_LOC, 'P5': P5_ANCHORS_PER_LOC}


def conv_bn_leaky(x, filters, kernel_size, strides=1, name_prefix=""):
    """Вспомогательный блок: Свертка -> BatchNormalization -> LeakyReLU."""
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,  # BN имеет смещение
        kernel_regularizer=L2_REGULARIZER, name=f"{name_prefix}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA_CFG, name=f"{name_prefix}_leakyrelu")(x)
    return x


def prediction_head(input_tensor, head_id_str,
                    known_grid_h, known_grid_w,  # Размеры сетки для этого уровня (вычисленные из страйда)
                    num_anchors_at_this_level, num_classes,
                    head_depth, head_filters):
    """Создает голову предсказаний для одного уровня FPN."""
    x = input_tensor
    # Тело головы
    for i in range(head_depth):
        current_head_filters = head_filters // (2 ** i)  # Можно уменьшать фильтры к концу
        current_head_filters = max(current_head_filters, 32)  # Не меньше 32
        x = conv_bn_leaky(x, filters=current_head_filters, kernel_size=3,
                          name_prefix=f"head_{head_id_str}_body_conv{i + 1}")

    # Финальная свертка для предсказаний
    num_predictions_per_anchor = 4 + 1 + num_classes  # 4 (box) + 1 (objectness) + C (classes)
    raw_predictions = tf.keras.layers.Conv2D(
        filters=num_anchors_at_this_level * num_predictions_per_anchor,
        kernel_size=1,  # 1x1 свертка для предсказаний
        padding='same',
        activation=None,  # Логиты, активации будут в функции потерь или при инференсе
        kernel_regularizer=L2_REGULARIZER,
        name=f"head_{head_id_str}_raw_preds"
    )(x)

    # Reshape выхода
    # known_grid_h, known_grid_w - это пространственные размеры input_tensor для этой головы
    output_shape_tuple = (known_grid_h, known_grid_w, num_anchors_at_this_level, num_predictions_per_anchor)
    reshaped_predictions = tf.keras.layers.Reshape(output_shape_tuple, name=f"head_{head_id_str}_predictions")(
        raw_predictions)
    return reshaped_predictions


def build_object_detector_v2_fpn(force_freeze_backbone_arg=None):
    """
    Строит модель детектора объектов с FPN.
    force_freeze_backbone_arg: Если True - заморозить backbone, если False - разморозить.
                               Если None - использовать значение из FREEZE_BACKBONE_INITIAL_CFG.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE_MODEL_CFG, name="input_image_detector_fpn")

    # 1. Backbone
    if BACKBONE_NAME_MODEL_CFG == 'MobileNetV2':
        base_model_loader = tf.keras.applications.MobileNetV2(
            input_shape=INPUT_SHAPE_MODEL_CFG,  # Важно передавать input_shape базовой модели
            include_top=False,
            weights='imagenet'
        )
        # Имена слоев для FPN (проверены для MobileNetV2 и входа ~416x416)
        c3_output_layer_name = 'block_6_expand_relu'  # Stride 8 (выход ~52x52)
        c4_output_layer_name = 'block_13_expand_relu'  # Stride 16 (выход ~26x26)
        c5_output_layer_name = 'out_relu'  # Stride 32 (выход ~13x13, последний слой)
    # Добавь сюда elif для других backbone (EfficientNetB0, ResNet50V2) с их именами слоев
    # elif BACKBONE_NAME_MODEL_CFG == 'EfficientNetB0':
    #     base_model_loader = tf.keras.applications.EfficientNetB0(...)
    #     c3_output_layer_name = '...' # Найди нужные имена
    #     c4_output_layer_name = '...'
    #     c5_output_layer_name = '...'
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
        print("Вывод summary базовой модели для проверки доступных имен слоев (если base_model_loader был создан):")
        if 'base_model_loader' in locals(): base_model_loader.summary(line_length=150)
        raise e

    # Создаем feature_extractor модель, которая будет принимать 'inputs' нашей основной модели
    feature_extractor = tf.keras.Model(inputs=base_model_loader.input,  # Используем input базовой модели
                                       outputs=[c3_out_tensor, c4_out_tensor, c5_out_tensor],
                                       name=f"Backbone_{BACKBONE_NAME_MODEL_CFG}_FPN_Features")

    # Управление заморозкой backbone
    current_freeze_decision = FREEZE_BACKBONE_INITIAL_CFG  # По умолчанию из конфига
    if force_freeze_backbone_arg is not None:
        current_freeze_decision = force_freeze_backbone_arg  # Переопределяем аргументом, если он задан

    if current_freeze_decision:
        feature_extractor.trainable = False
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' ЗАМОРОЖЕН (trainable=False).")
    else:
        feature_extractor.trainable = True
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' РАЗМОРОЖЕН (trainable=True).")

    # Применяем feature_extractor к входам нашей основной модели
    C3_features, C4_features, C5_features = feature_extractor(inputs)

    # 2. FPN - Шея
    # Lateral connections (1x1 convs to reduce channels)
    P5_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c5_to_p5_in',
                                   kernel_regularizer=L2_REGULARIZER)(C5_features)
    P4_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c4_to_p4_in',
                                   kernel_regularizer=L2_REGULARIZER)(C4_features)
    P3_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c3_to_p3_in',
                                   kernel_regularizer=L2_REGULARIZER)(C3_features)

    # Top-down pathway
    P5_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p5_upsampled')(
        P5_in)  # P5_in идет на P5_output и на апсемплинг
    P4_merged = tf.keras.layers.Add(name='fpn_p4_merged')([P5_upsampled, P4_in])

    P4_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p4_upsampled')(
        P4_merged)  # P4_merged идет на P4_output и на апсемплинг
    P3_merged = tf.keras.layers.Add(name='fpn_p3_merged')([P4_upsampled, P3_in])

    # Output feature maps of FPN (after 3x3 conv for smoothing)
    P5_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p5_output',
                                       kernel_regularizer=L2_REGULARIZER)(P5_in)
    P4_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p4_output',
                                       kernel_regularizer=L2_REGULARIZER)(P4_merged)
    P3_output = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p3_output',
                                       kernel_regularizer=L2_REGULARIZER)(P3_merged)

    # 3. Prediction Heads
    # Размеры сетки вычисляются на основе страйдов из конфига (или дефолтных)
    # FPN_LEVELS_CONFIG_GLOBAL должен быть доступен из detector_data_loader.py или определен здесь аналогично.
    # Для чистоты, лучше получать эти параметры из FPN_PARAMS (загруженного из конфига)

    fpn_outputs_for_heads = {
        'P3': P3_output,
        'P4': P4_output,
        'P5': P5_output
    }
    model_predictions_list = []

    for level_name in FPN_LEVEL_NAMES_MODEL:  # Итерируемся в правильном порядке P3, P4, P5
        level_stride = FPN_STRIDES_MODEL.get(level_name, 8 * (2 ** FPN_LEVEL_NAMES_MODEL.index(level_name)))
        num_anchors_this_lvl = NUM_ANCHORS_PER_LEVEL_MAP.get(level_name, 3)  # Используем NUM_ANCHORS_PER_LEVEL_MAP

        grid_h_level = INPUT_SHAPE_MODEL_CFG[0] // level_stride
        grid_w_level = INPUT_SHAPE_MODEL_CFG[1] // level_stride

        input_feature_map_for_head = fpn_outputs_for_heads[level_name]

        level_predictions = prediction_head(
            input_feature_map_for_head,
            head_id_str=level_name,
            known_grid_h=grid_h_level,
            known_grid_w=grid_w_level,
            num_anchors_at_this_level=num_anchors_this_lvl,
            num_classes=NUM_CLASSES_MODEL_CFG,
            head_depth=HEAD_DEPTH_CFG,
            head_filters=HEAD_CONV_FILTERS_CFG
        )
        model_predictions_list.append(level_predictions)

    final_model_name = FPN_PARAMS.get('model_name_prefix', f"{BACKBONE_NAME_MODEL_CFG}_FPN_Detector_v2")
    final_model = tf.keras.Model(inputs=inputs, outputs=model_predictions_list, name=final_model_name)
    return final_model


# --- Блок для тестирования этого файла (if __name__ == '__main__':) ---
if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Версия FPN) ---")
    # Загружаем конфиг еще раз локально для теста, чтобы убедиться, что он читается
    # (или используем глобальные переменные, если уверены в их инициализации)
    # Для простоты, будем полагаться на глобальные переменные, инициализированные в начале файла.

    print(f"Используется Backbone: {BACKBONE_NAME_MODEL_CFG}")
    print(f"Конфигурация входа для теста: {INPUT_SHAPE_MODEL_CFG}")
    print(f"Количество классов: {NUM_CLASSES_MODEL_CFG}")
    print(f"Якоря на уровень (из глобального NUM_ANCHORS_PER_LEVEL_MAP):")
    print(
        f"  P3: {NUM_ANCHORS_PER_LEVEL_MAP.get('P3', -1)}, P4: {NUM_ANCHORS_PER_LEVEL_MAP.get('P4', -1)}, P5: {NUM_ANCHORS_PER_LEVEL_MAP.get('P5', -1)}")
    print(
        f"FPN фильтры: {FPN_FILTERS_CFG}, Глубина головы: {HEAD_DEPTH_CFG}, Фильтры в голове: {HEAD_CONV_FILTERS_CFG}")

    try:
        # Тестируем с force_freeze_backbone_arg=True (дефолтное поведение при начальном обучении)
        detector_model_fpn = build_object_detector_v2_fpn(force_freeze_backbone_arg=True)
        print("\nСтруктура модели детектора с FPN (Backbone ЗАМОРОЖЕН):")
        detector_model_fpn.summary(line_length=150)

        print("\nТестовый прогон модели FPN (проверка форм выходов):")
        batch_size_for_test = 1  # Обычно для теста инференса
        dummy_input_shape_test_fpn = (
        batch_size_for_test, INPUT_SHAPE_MODEL_CFG[0], INPUT_SHAPE_MODEL_CFG[1], INPUT_SHAPE_MODEL_CFG[2])
        dummy_input_fpn = tf.random.normal(dummy_input_shape_test_fpn)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input_fpn.shape}")

        start_time = time.time()
        predictions_list = detector_model_fpn(dummy_input_fpn, training=False)  # training=False для инференса
        end_time = time.time()
        print(f"  Время инференса на фиктивном входе: {end_time - start_time:.4f} сек.")

        if isinstance(predictions_list, list) and len(predictions_list) == len(FPN_LEVEL_NAMES_MODEL):
            print(
                f"\n  Модель вернула список из {len(predictions_list)} тензоров ({', '.join(FPN_LEVEL_NAMES_MODEL)}):")

            expected_preds_per_anchor = 4 + 1 + NUM_CLASSES_MODEL_CFG  # box, obj, classes
            all_shapes_match = True

            for i, pred_tensor in enumerate(predictions_list):
                level_key = FPN_LEVEL_NAMES_MODEL[i]
                current_level_stride = FPN_STRIDES_MODEL.get(level_key, 8 * (2 ** i))
                current_level_anchors = NUM_ANCHORS_PER_LEVEL_MAP.get(level_key, 3)

                expected_grid_h = INPUT_SHAPE_MODEL_CFG[0] // current_level_stride
                expected_grid_w = INPUT_SHAPE_MODEL_CFG[1] // current_level_stride
                expected_shape_tuple = (
                batch_size_for_test, expected_grid_h, expected_grid_w, current_level_anchors, expected_preds_per_anchor)

                print(f"    Форма выхода {level_key}: {pred_tensor.shape}, Ожидаемая: {expected_shape_tuple}")
                if pred_tensor.shape.as_list() != list(expected_shape_tuple):
                    all_shapes_match = False
                    print(f"      ОШИБКА: Форма выхода {level_key} НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ!")

            if all_shapes_match:
                print("\n  Все формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
            else:
                print("\n  ОШИБКА: НЕ ВСЕ формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
        else:
            print(
                f"  ОШИБКА: Модель вернула не список из {len(FPN_LEVEL_NAMES_MODEL)} тензоров, а: {type(predictions_list)}")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели FPN детектора: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py (FPN) завершено ---")