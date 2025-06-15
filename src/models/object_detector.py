# RoadDefectDetector/src/models/object_detector.py
import tensorflow as tf
import yaml
import os

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
_base_config_path = os.path.join(_current_dir, '..', 'configs', 'base_config.yaml')

DETECTOR_CONFIG = {}
BASE_CONFIG = {}
CONFIG_LOAD_SUCCESS_OBJ_DET = True


def load_config_obj_det(config_path, config_name):
    global CONFIG_LOAD_SUCCESS_OBJ_DET
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_content = yaml.safe_load(f)
        if not isinstance(cfg_content, dict):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Конфиг '{config_name}' ({config_path}) пуст или имеет неверный формат.")
            CONFIG_LOAD_SUCCESS_OBJ_DET = False
            return {}
        print(f"INFO (object_detector.py): Конфиг '{config_name}' успешно загружен.")
        return cfg_content
    except FileNotFoundError:
        print(f"ОШИБКА (object_detector.py): Файл конфига '{config_name}' не найден: {config_path}")
        CONFIG_LOAD_SUCCESS_OBJ_DET = False
        return {}
    except yaml.YAMLError as e_yaml:
        print(f"ОШИБКА YAML (object_detector.py) при чтении '{config_name}': {e_yaml}")
        CONFIG_LOAD_SUCCESS_OBJ_DET = False
        return {}
    except Exception as e_load_cfg:
        print(f"НЕПРЕДВИДЕННАЯ ОШИБКА (object_detector.py) при загрузке '{config_name}': {e_load_cfg}")
        CONFIG_LOAD_SUCCESS_OBJ_DET = False
        return {}


BASE_CONFIG = load_config_obj_det(_base_config_path, "Base Config (for object_detector)")
DETECTOR_CONFIG = load_config_obj_det(_detector_config_path, "Detector Config (for object_detector)")

if not DETECTOR_CONFIG:
    print("КРИТИЧЕСКАЯ ОШИБКА: detector_config.yaml не загружен или пуст. Невозможно построить модель детектора.")
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('num_classes', 2)
    DETECTOR_CONFIG.setdefault('backbone_name', 'MobileNetV2')
    DETECTOR_CONFIG.setdefault('freeze_backbone', True)
    DETECTOR_CONFIG.setdefault('head_conv_filters', 256)
    DETECTOR_CONFIG.setdefault('num_anchors_per_location', 3)
    print("ПРЕДУПРЕЖДЕНИЕ: object_detector.py использует аварийные дефолты из-за проблем с конфигом.")

# --- Параметры из Конфигов ---
if 'input_shape' in DETECTOR_CONFIG and DETECTOR_CONFIG['input_shape']:
    INPUT_SHAPE_CFG = tuple(DETECTOR_CONFIG['input_shape'])
elif 'model_params' in BASE_CONFIG and BASE_CONFIG['model_params'] and 'input_shape' in BASE_CONFIG['model_params'] and \
        BASE_CONFIG['model_params']['input_shape']:
    INPUT_SHAPE_CFG = tuple(BASE_CONFIG['model_params']['input_shape'])
else:
    INPUT_SHAPE_CFG = (416, 416, 3)
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (object_detector.py): input_shape не найден в конфигах, используется дефолт {INPUT_SHAPE_CFG}")

NUM_CLASSES_CFG = DETECTOR_CONFIG.get('num_classes', 2)
BACKBONE_NAME_CFG = DETECTOR_CONFIG.get('backbone_name', 'MobileNetV2')
FREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)
HEAD_CONV_FILTERS_CFG = DETECTOR_CONFIG.get('head_conv_filters', 256)
NUM_ANCHORS_CFG = DETECTOR_CONFIG.get('num_anchors_per_location', 3)
NETWORK_STRIDE_ASSUMED = 16


def build_object_detector_v1():
    """
    Строит модель детектора объектов v1.
    Backbone (MobileNetV2, выход stride 16) -> "Тело" (2 Conv блока) -> "Голова" (2 Conv блока) -> Финальный предсказывающий Conv.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE_CFG, name="input_image_detector")

    # 1. Backbone
    if BACKBONE_NAME_CFG == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=INPUT_SHAPE_CFG,  # Передаем форму входа из конфига
            include_top=False,
            weights='imagenet'
        )
        try:
            feature_extraction_layer_name = 'block_13_expand_relu'  # Слой для stride 16
            backbone_output_tensor = base_model.get_layer(feature_extraction_layer_name).output
            print(
                f"INFO (object_detector): Используется выход слоя '{feature_extraction_layer_name}' из {BACKBONE_NAME_CFG} (ожидаемый stride 16).")
        except ValueError:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ (object_detector): Слой '{feature_extraction_layer_name}' не найден в {BACKBONE_NAME_CFG} для input_shape {INPUT_SHAPE_CFG}.")
            print("                         Будет использован последний выходной слой базовой модели.")
            backbone_output_tensor = base_model.output

        backbone_model_obj = tf.keras.Model(inputs=base_model.input, outputs=backbone_output_tensor,
                                            name=f"Backbone_{BACKBONE_NAME_CFG}")
    else:
        raise ValueError(f"Неподдерживаемый backbone: {BACKBONE_NAME_CFG}")

    if FREEZE_BACKBONE_CFG:
        backbone_model_obj.trainable = False
        print(f"INFO (object_detector): Backbone '{BACKBONE_NAME_CFG}' ЗАМОРОЖЕН (trainable = False).")
    else:
        backbone_model_obj.trainable = True
        print(f"INFO (object_detector): Backbone '{BACKBONE_NAME_CFG}' РАЗМОРОЖЕН (trainable = True).")

    x = backbone_model_obj(inputs, training=(not FREEZE_BACKBONE_CFG))

    # 2. "Тело" детектора
    x = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG, kernel_size=3, padding='same',
                               kernel_initializer='he_normal', use_bias=False, name="detector_body_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="detector_body_bn1")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="detector_body_leakyrelu1")(
        x)  # ИЗМЕНЕНО: alpha -> negative_slope

    x = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG // 2, kernel_size=1, padding='same',
                               kernel_initializer='he_normal', use_bias=False, name="detector_body_conv2_1x1")(x)
    x = tf.keras.layers.BatchNormalization(name="detector_body_bn2")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="detector_body_leakyrelu2")(
        x)  # ИЗМЕНЕНО: alpha -> negative_slope

    head_input_features = x

    # 3. "Голова" предсказаний
    # Блок 1 "головы"
    h = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG, kernel_size=3, padding='same',
                               kernel_initializer='he_normal', use_bias=False, name="detector_head_conv_b1_c1")(
        head_input_features)
    h = tf.keras.layers.BatchNormalization(name="detector_head_bn_b1_c1")(h)
    h = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="detector_head_leaky_b1_c1")(h)  # ИЗМЕНЕНО

    # Блок 2 "головы"
    h = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG, kernel_size=3, padding='same',
                               kernel_initializer='he_normal', use_bias=False, name="detector_head_conv_b2_c1")(h)
    h = tf.keras.layers.BatchNormalization(name="detector_head_bn_b2_c1")(h)
    h = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="detector_head_leaky_b2_c1")(h)  # ИЗМЕНЕНО

    num_predictions_per_anchor = 4 + 1 + NUM_CLASSES_CFG
    total_output_filters = NUM_ANCHORS_CFG * num_predictions_per_anchor

    raw_predictions = tf.keras.layers.Conv2D(
        filters=total_output_filters,
        kernel_size=1,
        padding='same',
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name="detection_head_raw_predictions"
    )(h)

    grid_h = INPUT_SHAPE_CFG[0] // NETWORK_STRIDE_ASSUMED
    grid_w = INPUT_SHAPE_CFG[1] // NETWORK_STRIDE_ASSUMED

    final_output_shape = (grid_h, grid_w, NUM_ANCHORS_CFG, num_predictions_per_anchor)
    output_tensor = tf.keras.layers.Reshape(
        final_output_shape, name="detector_predictions"
    )(raw_predictions)

    model = tf.keras.Model(inputs=inputs, outputs=output_tensor, name=f"{BACKBONE_NAME_CFG}_ObjectDetector_v1")
    return model


if __name__ == '__main__':
    if not CONFIG_LOAD_SUCCESS_OBJ_DET:
        print(
            "\nОШИБКА: Не удалось загрузить файлы конфигурации. Тестирование object_detector.py может быть некорректным.")

    print(f"--- Тестирование object_detector.py (Версия v1 с Backbone, Телом, Головой) ---")
    print(f"Конфигурация входа (из detector_config или base_config): {INPUT_SHAPE_CFG}")
    print(f"Количество классов (из detector_config): {NUM_CLASSES_CFG}")
    print(f"Количество якорей на ячейку (из detector_config): {NUM_ANCHORS_CFG}")
    print(f"Имя Backbone (из detector_config): {BACKBONE_NAME_CFG}")
    print(f"Заморозка Backbone (из detector_config): {FREEZE_BACKBONE_CFG}")
    print(f"Фильтры в свертках головы (из detector_config): {HEAD_CONV_FILTERS_CFG}")

    try:
        detector_model = build_object_detector_v1()
        print("\nСтруктура модели детектора:")
        detector_model.summary(line_length=120)

        print("\nТестовый прогон модели (проверка формы выхода):")
        batch_size_for_test = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 1)
        if batch_size_for_test == 0: batch_size_for_test = 1

        dummy_input_shape_test = (
            batch_size_for_test,
            INPUT_SHAPE_CFG[0],
            INPUT_SHAPE_CFG[1],
            INPUT_SHAPE_CFG[2]
        )
        dummy_input = tf.random.normal(dummy_input_shape_test)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input.shape}")

        predictions = detector_model(dummy_input)

        expected_grid_h = INPUT_SHAPE_CFG[0] // NETWORK_STRIDE_ASSUMED
        expected_grid_w = INPUT_SHAPE_CFG[1] // NETWORK_STRIDE_ASSUMED
        expected_output_features_per_anchor = 4 + 1 + NUM_CLASSES_CFG

        print(f"  Форма предсказаний: {predictions.shape}")
        expected_shape_tuple = (
        batch_size_for_test, expected_grid_h, expected_grid_w, NUM_ANCHORS_CFG, expected_output_features_per_anchor)
        print(f"  Ожидаемая форма: {expected_shape_tuple}")

        actual_shape_no_batch = predictions.shape.as_list()[1:]
        expected_shape_no_batch = list(expected_shape_tuple)[1:]

        if actual_shape_no_batch == expected_shape_no_batch:
            print("  Форма выхода СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
        else:
            print("  ОШИБКА: Форма выхода НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
            print(f"    Получено: {predictions.shape.as_list()}")
            print(f"    Ожидалось: {list(expected_shape_tuple)}")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели детектора: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py завершено ---")