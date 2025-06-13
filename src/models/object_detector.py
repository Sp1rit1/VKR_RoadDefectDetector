# src/models/object_detector.py
import tensorflow as tf
import yaml
import os

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
_base_config_path = os.path.join(_current_dir, '..', 'configs', 'base_config.yaml')

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ОШИБКА: Файл конфигурации детектора или базовый не найден (object_detector.py).")
    # Задаем дефолты, чтобы модуль хотя бы импортировался и можно было запустить if __name__ == '__main__'
    DETECTOR_CONFIG = {'backbone_name': 'MobileNetV2', 'input_shape': [416, 416, 3],
                       'freeze_backbone': True, 'num_classes': 2, 'head_conv_filters': 128,
                       'train_params': {'batch_size': 1}}  # Добавил train_params для теста
    BASE_CONFIG = {'model_params': {'target_height': 416, 'target_width': 416}}
    print("ПРЕДУПРЕЖДЕНИЕ: Файлы конфигурации не найдены, используются значения по умолчанию для object_detector.py.")
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл: {e} (object_detector.py)")
    DETECTOR_CONFIG = {'backbone_name': 'MobileNetV2', 'input_shape': [416, 416, 3],
                       'freeze_backbone': True, 'num_classes': 2, 'head_conv_filters': 128,
                       'train_params': {'batch_size': 1}}
    BASE_CONFIG = {'model_params': {'target_height': 416, 'target_width': 416}}
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка чтения YAML, используются значения по умолчанию (object_detector.py).")

INPUT_SHAPE_CFG = tuple(
    DETECTOR_CONFIG.get('input_shape', BASE_CONFIG.get('model_params', {}).get('input_shape', [416, 416, 3])))
NUM_CLASSES_CFG = DETECTOR_CONFIG.get('num_classes', 2)  # pit, crack
BACKBONE_NAME_CFG = DETECTOR_CONFIG.get('backbone_name', 'MobileNetV2')
FREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)

# Архитектурное решение: Предсказываем фиксированное максимальное количество рамок
MAX_BOXES_PER_IMAGE = DETECTOR_CONFIG.get('max_boxes_per_image', 10)  # Возьмем из конфига или дефолт 10


def build_simple_object_detector():
    """
    Строит очень простую модель детектора объектов.
    Backbone -> GlobalAveragePooling -> Несколько Dense слоев для предсказания MAX_BOXES_PER_IMAGE рамок.
    Каждая рамка: [x_center_norm, y_center_norm, width_norm, height_norm, obj_score, class1_prob, class2_prob]
    (Всего 4 + 1 + NUM_CLASSES_CFG = 7 значений на рамку, если NUM_CLASSES_CFG=2)
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE_CFG, name="input_image_detector")

    # 1. Backbone
    if BACKBONE_NAME_CFG == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE_CFG,
                                                       include_top=False,
                                                       weights='imagenet')
    # Сюда можно добавить другие варианты backbone, если нужно
    else:
        raise ValueError(f"Неподдерживаемый backbone: {BACKBONE_NAME_CFG}")

    if FREEZE_BACKBONE_CFG:
        base_model.trainable = False

    x = base_model(inputs, training=(not FREEZE_BACKBONE_CFG))

    # 2. Уменьшение размерности признаков
    x = tf.keras.layers.GlobalAveragePooling2D(name="detector_global_avg_pool")(x)

    # 3. Полносвязные слои для предсказания всех рамок сразу
    num_outputs_per_box = 4 + 1 + NUM_CLASSES_CFG  # 4 coords, 1 objectness, N_classes
    total_outputs = MAX_BOXES_PER_IMAGE * num_outputs_per_box

    x = tf.keras.layers.Dense(512, activation="relu", name="detector_dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="detector_dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="detector_dense_2")(x)

    raw_predictions = tf.keras.layers.Dense(total_outputs, activation=None, name="detector_raw_predictions")(x)

    # Решейпим выход в (batch_size, MAX_BOXES_PER_IMAGE, num_outputs_per_box)
    outputs = tf.keras.layers.Reshape((MAX_BOXES_PER_IMAGE, num_outputs_per_box),
                                      name="detector_predictions_reshaped")(raw_predictions)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Simple_{BACKBONE_NAME_CFG}_Detector")
    return model


if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Простая модель) ---")
    print(f"INPUT_SHAPE_CFG: {INPUT_SHAPE_CFG}")
    print(f"NUM_CLASSES_CFG: {NUM_CLASSES_CFG}")
    print(f"MAX_BOXES_PER_IMAGE: {MAX_BOXES_PER_IMAGE}")
    try:
        detector_model = build_simple_object_detector()
        print("\nСтруктура модели детектора:")
        detector_model.summary()

        print("\nТестовый прогон модели (проверка формы выхода):")
        # Используем batch_size из detector_config, если есть, или дефолт
        # Убедимся, что train_params существует в DETECTOR_CONFIG
        if 'train_params' in DETECTOR_CONFIG and 'batch_size' in DETECTOR_CONFIG['train_params']:
            batch_size_for_test = DETECTOR_CONFIG['train_params']['batch_size']
        else:  # Если train_params или batch_size нет, используем дефолт (например 1)
            batch_size_for_test = 1
            print(
                "ПРЕДУПРЕЖДЕНИЕ: train_params.batch_size не найден в detector_config.yaml, используется batch_size=1 для теста.")

        dummy_input_shape = (
            batch_size_for_test,
            INPUT_SHAPE_CFG[0],
            INPUT_SHAPE_CFG[1],
            INPUT_SHAPE_CFG[2]
        )
        dummy_input = tf.random.normal(dummy_input_shape)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input.shape}")

        predictions = detector_model(dummy_input)
        expected_output_features = 4 + 1 + NUM_CLASSES_CFG
        print(f"  Форма предсказаний: {predictions.shape}")
        print(f"  Ожидаемая форма: ({batch_size_for_test}, {MAX_BOXES_PER_IMAGE}, {expected_output_features})")
        if predictions.shape == (batch_size_for_test, MAX_BOXES_PER_IMAGE, expected_output_features):
            print("  Форма выхода СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
        else:
            print("  ОШИБКА: Форма выхода НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели детектора: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py завершено ---")