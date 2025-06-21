# src/models/object_detector_single_level_v2.py
import tensorflow as tf
import yaml
import os
from pathlib import Path
import time

# --- Настройка sys.path (если этот модуль импортируется из скрипта в корне) ---
# ... (оставляем как было) ...
_current_model_script_dir = Path(__file__).resolve().parent
_src_model_dir = _current_model_script_dir.parent
_project_root_model = _src_model_dir.parent

# --- Загрузка Конфигурации ---
# ... (оставляем как было, с загрузкой DETECTOR_CONFIG_MODEL_V2 и определением
#      INPUT_SHAPE_MODEL_V2, BACKBONE_NAME_MODEL_V2, FORCE_FREEZE_BACKBONE_ON_BUILD_CFG,
#      HEAD_CONV_FILTERS_MODEL_V2, HEAD_DEPTH_MODEL_V2, LEAKY_RELU_ALPHA_MODEL_V2,
#      L2_REG_VALUE_MODEL_V2, L2_REGULARIZER_MODEL_V2, NUM_CLASSES_MODEL_V2,
#      _level_name_model_v2, _level_stride_model_v2, NUM_ANCHORS_SL_MODEL_V2,
#      MODEL_BASE_NAME_FROM_CFG, BACKBONE_NAME_IN_MODEL_CFG
#      ) ...
_detector_config_path_model_v2_primary = _src_model_dir / 'configs' / 'detector_config_single_level_v2.yaml'
_detector_config_path_model_v2_fallback = _src_model_dir / 'configs' / 'detector_config_single_level_debug.yaml'
DETECTOR_CONFIG_MODEL_V2 = {};
_config_to_load_model_v2 = _detector_config_path_model_v2_primary
if not _config_to_load_model_v2.exists(): _config_to_load_model_v2 = _detector_config_path_model_v2_fallback
if os.path.exists(_config_to_load_model_v2):
    try:
        with open(_config_to_load_model_v2, 'r', encoding='utf-8') as f:
            DETECTOR_CONFIG_MODEL_V2 = yaml.safe_load(f)
        if not isinstance(DETECTOR_CONFIG_MODEL_V2, dict) or not DETECTOR_CONFIG_MODEL_V2: DETECTOR_CONFIG_MODEL_V2 = {}
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML (object_detector_sl_v2) в {_config_to_load_model_v2.name}: {e}.")
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ (object_detector_sl_v2): Файл конфигурации {_config_to_load_model_v2.name} не найден.")

_sl_params_model_v2 = DETECTOR_CONFIG_MODEL_V2.get('single_level_detector_params',
                                                   DETECTOR_CONFIG_MODEL_V2.get('fpn_detector_params', {}))
INPUT_SHAPE_MODEL_V2 = tuple(_sl_params_model_v2.get('input_shape', [416, 416, 3]));
BACKBONE_NAME_MODEL_V2 = _sl_params_model_v2.get('backbone_name', 'MobileNetV2')
FORCE_FREEZE_BACKBONE_ON_BUILD_CFG = DETECTOR_CONFIG_MODEL_V2.get('freeze_backbone', True)
_head_cfg_model_v2 = _sl_params_model_v2.get('head_config', DETECTOR_CONFIG_MODEL_V2.get('head_config', {}))
HEAD_CONV_FILTERS_MODEL_V2 = _head_cfg_model_v2.get('head_conv_filters', 256);
HEAD_DEPTH_MODEL_V2 = _head_cfg_model_v2.get('head_depth', 2)
LEAKY_RELU_ALPHA_MODEL_V2 = _head_cfg_model_v2.get('leaky_relu_alpha', 0.1);
L2_REG_VALUE_MODEL_V2 = _head_cfg_model_v2.get('l2_regularization', None)
L2_REGULARIZER_MODEL_V2 = tf.keras.regularizers.l2(
    L2_REG_VALUE_MODEL_V2) if L2_REG_VALUE_MODEL_V2 and L2_REG_VALUE_MODEL_V2 > 0 else None
NUM_CLASSES_MODEL_V2 = _sl_params_model_v2.get('num_classes', 2)
_level_name_sl_model_v2 = \
_sl_params_model_v2.get('detector_levels', _sl_params_model_v2.get('detector_fpn_levels', ['P4_debug']))[0]
_level_stride_sl_model_v2 = _sl_params_model_v2.get('detector_strides', {}).get(_level_name_sl_model_v2,
                                                                                _sl_params_model_v2.get(
                                                                                    'detector_fpn_strides', {}).get(
                                                                                    _level_name_sl_model_v2, 16))
_level_anchor_cfg_yaml_sl_model_v2 = _sl_params_model_v2.get('detector_anchor_configs', {}).get(_level_name_sl_model_v2,
                                                                                                _sl_params_model_v2.get(
                                                                                                    'detector_fpn_anchor_configs',
                                                                                                    {}).get(
                                                                                                    _level_name_sl_model_v2,
                                                                                                    {}))
NUM_ANCHORS_SL_MODEL_V2 = _level_anchor_cfg_yaml_sl_model_v2.get('num_anchors_this_level', 7)
_anchors_wh_list_sl = _level_anchor_cfg_yaml_sl_model_v2.get('anchors_wh_normalized', [])
if _anchors_wh_list_sl and len(_anchors_wh_list_sl) != NUM_ANCHORS_SL_MODEL_V2: NUM_ANCHORS_SL_MODEL_V2 = len(
    _anchors_wh_list_sl)
if NUM_ANCHORS_SL_MODEL_V2 == 0 and not _anchors_wh_list_sl: NUM_ANCHORS_SL_MODEL_V2 = 1
MODEL_BASE_NAME_FROM_CFG = DETECTOR_CONFIG_MODEL_V2.get('model_base_name', 'RoadDefectDetector')
BACKBONE_NAME_IN_MODEL_CFG = DETECTOR_CONFIG_MODEL_V2.get('backbone_layer_name_in_model',
                                                          f"Backbone_{BACKBONE_NAME_MODEL_V2}")

if not DETECTOR_CONFIG_MODEL_V2 or not _sl_params_model_v2: print(
    "ПРЕДУПРЕЖДЕНИЕ (object_detector_sl_v2): АВАРИЙНЫЕ ДЕФОЛТЫ МОДЕЛИ.")  # Сократил для краткости


def conv_bn_leaky_v2(x, filters, kernel_size, strides=1, name_prefix="", l2_reg=L2_REGULARIZER_MODEL_V2):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_regularizer=l2_reg, name=f"{name_prefix}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA_MODEL_V2, name=f"{name_prefix}_leakyrelu")(x)
    return x


def build_object_detector_single_level_v2(
        input_shape_arg=INPUT_SHAPE_MODEL_V2,
        backbone_name_arg=BACKBONE_NAME_MODEL_V2,
        force_initial_freeze_backbone_arg=FORCE_FREEZE_BACKBONE_ON_BUILD_CFG,
        head_depth_arg=HEAD_DEPTH_MODEL_V2,
        head_filters_arg=HEAD_CONV_FILTERS_MODEL_V2,
        num_anchors_arg=NUM_ANCHORS_SL_MODEL_V2,
        num_classes_arg=NUM_CLASSES_MODEL_V2,
        level_stride_arg=_level_stride_sl_model_v2,
        l2_reg_val_arg=L2_REG_VALUE_MODEL_V2
):
    """
    Строит ОДНОУРОВНЕВУЮ модель детектора объектов (Версия 2 - Исправленная).
    """
    inputs = tf.keras.Input(shape=input_shape_arg, name="input_image_detector_sl_v2")
    current_l2_regularizer = tf.keras.regularizers.l2(l2_reg_val_arg) if l2_reg_val_arg and l2_reg_val_arg > 0 else None

    # 1. Backbone
    if backbone_name_arg == 'MobileNetV2':
        # Создаем экземпляр базовой модели MobileNetV2
        # Важно НЕ передавать input_tensor сюда, чтобы base_model была независимой моделью,
        # чей trainable статус мы можем контролировать перед тем, как использовать ее слои.
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape_arg,
            include_top=False,
            weights='imagenet'
            # name=BACKBONE_NAME_IN_MODEL_CFG # Имя можно задать позже для обертки
        )
        # Устанавливаем trainable статус для ВСЕЙ базовой модели
        base_model.trainable = not force_initial_freeze_backbone_arg
        print(
            f"INFO (SL_V2 Model): Базовая модель '{base_model.name}' trainable = {base_model.trainable} (на основе force_initial_freeze_backbone_arg: {force_initial_freeze_backbone_arg}).")

        # Имя слоя для извлечения признаков (stride 16 для MobileNetV2)
        feature_extraction_layer_name = 'block_13_expand_relu'
        try:
            feature_map_from_base_layer = base_model.get_layer(feature_extraction_layer_name)
            print(
                f"INFO (SL_V2 Model): Будет использован слой '{feature_extraction_layer_name}' из {backbone_name_arg}.")
        except ValueError:
            print(
                f"WARNING (SL_V2 Model): Слой '{feature_extraction_layer_name}' не найден. Используется выход всей base_model.")
            feature_map_from_base_layer = base_model  # Используем всю модель как экстрактор

        # Создаем "обертку" вокруг части backbone, чтобы дать ей имя и использовать в нашей модели
        # Вход для этой обертки - это вход всей нашей модели 'inputs'
        # Выход - это выход нужного нам слоя из base_model
        # Это делается путем применения base_model к inputs, а затем извлечения нужного слоя.
        # Более чистый способ - создать новую модель, которая использует слои базовой.

        # Применяем базовую модель (с уже установленным trainable статусом) к нашему входу
        x_backbone_processed = base_model(inputs)  # training флаг будет управляться Keras при model.fit

        # Если feature_extraction_layer_name не последний, нам нужно извлечь именно его выход.
        # Если base_model(inputs) уже выдает выход 'block_13_expand_relu', то все хорошо.
        # Если нет, то нужно создать модель, которая это делает.

        # Чтобы гарантированно получить выход нужного слоя и при этом правильно управлять trainable,
        # мы сначала применяем base_model, а затем, если feature_extraction_layer_name не последний,
        # мы должны были бы создать tf.keras.Model от base_model.input до этого слоя.
        # Но так как base_model УЖЕ принял inputs, x_backbone_processed содержит все выходы.
        # Нам нужно выбрать правильный.

        # Самый надежный способ получить выход нужного слоя из base_model, примененной к inputs:
        temp_model_for_output_extraction = tf.keras.Model(inputs=base_model.input,
                                                          outputs=feature_map_from_base_layer.output)
        x = temp_model_for_output_extraction(base_model.input)  # Получаем тензор-выход
        # Однако, чтобы связать это с нашим `inputs`, мы уже передали `inputs` в `base_model`.
        # Значит, `base_model.get_layer(feature_extraction_layer_name).output` относится к графу,
        # построенному на `inputs`.

        # Правильная цепочка:
        # 1. Создать base_model (MobileNetV2) с input_shape, но БЕЗ input_tensor.
        # 2. Установить base_model.trainable.
        # 3. Создать feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer(...).output, name=BACKBONE_NAME_IN_MODEL_CFG)
        # 4. x = feature_extractor(inputs)

        # Переделываем:
        mobilenet_base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape_arg,
            include_top=False,
            weights='imagenet'
        )
        mobilenet_base.trainable = not force_initial_freeze_backbone_arg
        print(f"INFO (SL_V2 Model): MobileNetV2 trainable = {mobilenet_base.trainable}")

        feature_extraction_layer_name = 'block_13_expand_relu'
        try:
            output_tensor_from_mobilenet = mobilenet_base.get_layer(feature_extraction_layer_name).output
        except ValueError:
            print(
                f"WARNING (SL_V2 Model): Слой '{feature_extraction_layer_name}' не найден. Используется последний выход.")
            output_tensor_from_mobilenet = mobilenet_base.output

        # Создаем функциональную модель для backbone части
        backbone_submodel = tf.keras.Model(
            inputs=mobilenet_base.input,
            outputs=output_tensor_from_mobilenet,
            name=BACKBONE_NAME_IN_MODEL_CFG
        )

        if force_initial_freeze_backbone_arg:
            backbone_submodel.trainable = False
            print(f"INFO (SL_V2 Model): Backbone submodel '{backbone_submodel.name}' ЗАМОРОЖЕН (trainable = False).")
        else:
            backbone_submodel.trainable = True
            print(f"INFO (SL_V2 Model): Backbone submodel '{backbone_submodel.name}' РАЗМОРОЖЕН (trainable = True).")
        # trainable статус backbone_submodel будет унаследован от mobilenet_base

        x = backbone_submodel(inputs)  # Применяем submodel к входу нашей основной модели
        print(f"INFO (SL_V2 Model): Форма выхода backbone submodel ('{backbone_submodel.name}'): {x.shape}")

    else:
        raise ValueError(f"Неподдерживаемый backbone: {backbone_name_arg}")

    # 2. "Тело" детектора
    for i in range(head_depth_arg):
        x = conv_bn_leaky_v2(x, filters=head_filters_arg, kernel_size=3,
                             name_prefix=f"detector_body_sl_d{i + 1}", l2_reg=current_l2_regularizer)

    # 3. "Голова" предсказаний
    num_predictions_per_anchor = 4 + 1 + num_classes_arg
    total_output_filters_head = num_anchors_arg * num_predictions_per_anchor

    print(f"DEBUG MODEL BUILD: num_anchors_arg={num_anchors_arg}, num_classes_arg={num_classes_arg}")
    print(f"DEBUG MODEL BUILD: num_predictions_per_anchor={num_predictions_per_anchor}")
    print(f"DEBUG MODEL BUILD: total_output_filters_head={total_output_filters_head}")  # Должно быть 49

    raw_head_output = tf.keras.layers.Conv2D(
        filters=total_output_filters_head,
        kernel_size=1,
        padding='same',
        activation=None,
        kernel_regularizer=current_l2_regularizer,
        name="detection_head_raw_predictions_sl"
    )(x)

    grid_h_calculated = input_shape_arg[0] // level_stride_arg
    grid_w_calculated = input_shape_arg[1] // level_stride_arg

    output_reshaped_shape = (grid_h_calculated, grid_w_calculated,
                             num_anchors_arg, num_predictions_per_anchor)

    predictions_final = tf.keras.layers.Reshape(
        output_reshaped_shape, name="detector_predictions_sl"
    )(raw_head_output)

    final_model_name = f"{MODEL_BASE_NAME_FROM_CFG}_SingleLevel_V2_Corrected"
    final_model = tf.keras.Model(inputs=inputs, outputs=predictions_final, name=final_model_name)
    return final_model


# --- Блок для тестирования этого файла ---
if __name__ == '__main__':
    print(f"--- Тестирование object_detector_single_level_v2.py (ИСПРАВЛЕННЫЙ) ---")

    print(f"Конфигурация входа (из глобальных переменных модуля): {INPUT_SHAPE_MODEL_V2}")
    print(
        f"Backbone: {BACKBONE_NAME_MODEL_V2}, Заморожен изначально (по конфигу): {FORCE_FREEZE_BACKBONE_ON_BUILD_CFG}")
    print(f"Параметры головы: Глубина={HEAD_DEPTH_MODEL_V2}, Фильтры={HEAD_CONV_FILTERS_MODEL_V2}")
    print(f"Количество классов: {NUM_CLASSES_MODEL_V2}")
    print(f"Количество якорей на ячейку (для уровня '{_level_name_sl_model_v2}'): {NUM_ANCHORS_SL_MODEL_V2}")
    print(f"Страйд для уровня '{_level_name_sl_model_v2}': {_level_stride_sl_model_v2}")
    print(f"L2 Reg Value: {L2_REG_VALUE_MODEL_V2}")

    try:
        detector_model = build_object_detector_single_level_v2()

        print("\nСтруктура модели:")
        # Выведем summary с trainable статусом для каждого слоя
        detector_model.summary(line_length=150, show_trainable=True)

        print("\nТестовый прогон (проверка формы выхода):")
        test_batch_size_main = DETECTOR_CONFIG_MODEL_V2.get('train_params', {}).get('batch_size', 1) \
            if DETECTOR_CONFIG_MODEL_V2 and 'train_params' in DETECTOR_CONFIG_MODEL_V2 \
            else 2  # Дефолт для теста, если train_params не найдены

        dummy_input_shape_test_main = (
            test_batch_size_main,
            INPUT_SHAPE_MODEL_V2[0], INPUT_SHAPE_MODEL_V2[1], INPUT_SHAPE_MODEL_V2[2]
        )
        dummy_input_main = tf.random.normal(dummy_input_shape_test_main)
        print(f"  Фиктивный вход: {dummy_input_main.shape}")

        start_time_inf_main = time.time()
        predictions_main = detector_model(dummy_input_main, training=False)  # training=False для инференса
        end_time_inf_main = time.time()
        print(f"  Время инференса на фиктивном входе: {end_time_inf_main - start_time_inf_main:.4f} сек.")

        print(f"  Форма предсказаний: {predictions_main.shape}")

        expected_gh_main = INPUT_SHAPE_MODEL_V2[0] // _level_stride_sl_model_v2
        expected_gw_main = INPUT_SHAPE_MODEL_V2[1] // _level_stride_sl_model_v2
        expected_features_per_anchor_main = 4 + 1 + NUM_CLASSES_MODEL_V2
        expected_shape_tuple_test_main = (test_batch_size_main, expected_gh_main, expected_gw_main,
                                          NUM_ANCHORS_SL_MODEL_V2, expected_features_per_anchor_main)

        print(f"  Ожидаемая форма: {expected_shape_tuple_test_main}")
        if predictions_main.shape.as_list() == list(expected_shape_tuple_test_main):
            print("  Форма выхода СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
        else:
            print("  ОШИБКА: Форма выхода НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")

    except Exception as e_model_test_main:
        print(f"ОШИБКА при создании или тестировании модели: {e_model_test_main}")
        import traceback

        traceback.print_exc()

    print("\n--- Тестирование object_detector_single_level_v2.py завершено ---")