# RoadDefectDetector/src/models/image_classifier.py
import tensorflow as tf
import yaml
import os

# --- Загрузка Конфигурации ---
# _current_dir теперь это src/models/
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Путь к конфигу классификатора
_config_path = os.path.join(_current_dir, '..', 'configs', 'classifier_config.yaml')

try:
    with open(_config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ОШИБКА: Файл конфигурации классификатора не найден: {_config_path}")
    # Предоставляем базовые значения по умолчанию для возможности импорта и тестирования модуля
    CONFIG = {
        'model_name': 'MobileNetV2',
        'input_shape': [224, 224, 3],
        'num_classes': 2,  # 'road', 'not_road'
        'freeze_base_model': True,
        'train_params': {'batch_size': 4}  # Для тестового прогона, если конфиг не найден
    }
    print(f"ПРЕДУПРЕЖДЕНИЕ: {_config_path} не найден. Используются значения по умолчанию для image_classifier.py.")
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл classifier_config.yaml: {e}")
    CONFIG = {
        'model_name': 'MobileNetV2',
        'input_shape': [224, 224, 3],
        'num_classes': 2,
        'freeze_base_model': True,
        'train_params': {'batch_size': 4}
    }
    print(
        f"ПРЕДУПРЕЖДЕНИЕ: Ошибка чтения classifier_config.yaml. Используются значения по умолчанию для image_classifier.py.")


def build_classifier_model():
    """
    Собирает и возвращает модель классификатора изображений.
    Использует параметры из загруженного CONFIG (classifier_config.yaml).
    """
    input_shape_cfg = tuple(CONFIG.get('input_shape', [224, 224, 3]))  # (высота, ширина, каналы)
    num_classes_cfg = CONFIG.get('num_classes', 2)
    base_model_name_cfg = CONFIG.get('model_name', 'MobileNetV2')
    freeze_base_cfg = CONFIG.get('freeze_base_model', True)

    # Выбор базовой pre-trained модели
    if base_model_name_cfg == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape_cfg,
                                                       include_top=False,
                                                       # НЕ включаем верхний классификационный слой ImageNet
                                                       weights='imagenet')  # Загружаем веса, предобученные на ImageNet
    elif base_model_name_cfg == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape_cfg,
                                                          include_top=False,
                                                          weights='imagenet')
    # Сюда можно добавить другие варианты базовых моделей по желанию (VGG16, ResNet50, и т.д.)
    # elif base_model_name_cfg == 'ResNet50V2':
    #     base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape_cfg,
    #                                                   include_top=False,
    #                                                   weights='imagenet')
    else:
        raise ValueError(f"Неподдерживаемое имя базовой модели: '{base_model_name_cfg}' в classifier_config.yaml")

    if freeze_base_cfg:
        base_model.trainable = False  # Замораживаем веса базовой модели для transfer learning
        print(f"Информация: Базовая модель '{base_model_name_cfg}' ЗАМОРОЖЕНА (trainable = False).")
    else:
        base_model.trainable = True  # Или размораживаем для fine-tuning всей сети
        print(f"Информация: Базовая модель '{base_model_name_cfg}' РАЗМОРОЖЕНА (trainable = True).")

    # Создание новой модели ("головы") поверх базовой
    inputs = tf.keras.Input(shape=input_shape_cfg, name="input_image")

    # Передаем training=False для базовой модели, если она заморожена,
    # чтобы слои BatchNormalization работали в режиме инференса.
    # Если base_model.trainable = True (для fine-tuning), то training должен быть True во время обучения.
    # Keras автоматически управляет этим флагом training для слоев во время model.fit() и model.predict().
    x = base_model(inputs, training=False if freeze_base_cfg else True)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    # Слой GlobalAveragePooling2D берет среднее значение по пространственным измерениям
    # для каждой карты признаков, превращая их в одномерный вектор признаков.

    x = tf.keras.layers.Dense(128, activation='relu', name="dense_features")(x)
    # Полносвязный слой для изучения более высокоуровневых комбинаций признаков.
    # 128 нейронов - это примерное значение, можно экспериментировать.

    x = tf.keras.layers.Dropout(0.3, name="dropout_layer")(x)
    # Слой Dropout для регуляризации, помогает предотвратить переобучение,
    # случайно "выключая" некоторые нейроны во время обучения.

    # Выходной слой
    if num_classes_cfg == 2:
        # Для бинарной классификации (road/not_road) можно использовать один выходной нейрон с sigmoid
        # (предсказывает вероятность принадлежности к одному из классов, например, 'road')
        # В этом случае loss должен быть 'binary_crossentropy'.
        # И метки классов должны быть 0 и 1.
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)
    elif num_classes_cfg > 2:
        # Для мультиклассовой классификации (если бы у нас было больше 2х классов)
        outputs = tf.keras.layers.Dense(num_classes_cfg, activation='softmax', name="output_softmax")(x)
    else:  # num_classes_cfg == 1 тоже можно, но тогда это эквивалентно num_classes_cfg == 2 с sigmoid
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model_name_cfg}_ImageClassifier")
    return model


if __name__ == '__main__':
    print(f"--- Тестирование image_classifier.py ---")
    print(f"Используется базовая модель: {CONFIG.get('model_name', 'MobileNetV2')}")
    print(f"Входная форма: {CONFIG.get('input_shape', [224, 224, 3])}")
    print(f"Количество классов: {CONFIG.get('num_classes', 2)}")
    print(f"Заморозка базовой модели: {CONFIG.get('freeze_base_model', True)}")

    try:
        classifier_model = build_classifier_model()
        print("\nСтруктура модели классификатора:")
        classifier_model.summary()

        print("\nТестовый прогон модели (проверка формы выхода):")
        # Получаем batch_size из train_params в CONFIG, если есть, иначе дефолт 4
        batch_size_for_test = CONFIG.get('train_params', {}).get('batch_size', 4)
        dummy_input_shape = (
            batch_size_for_test,
            CONFIG['input_shape'][0],
            CONFIG['input_shape'][1],
            CONFIG['input_shape'][2]
        )
        dummy_input = tf.random.normal(dummy_input_shape)

        print(f"  Создан фиктивный входной тензор с формой: {dummy_input.shape}")
        predictions = classifier_model(dummy_input)  # training=False по умолчанию для model.__call__
        print(
            f"  Форма предсказаний: {predictions.shape}")  # Ожидается (batch_size, 1) для sigmoid или (batch_size, num_classes) для softmax

        if CONFIG['num_classes'] == 2 and predictions.shape[-1] == 1:
            print("  Выходной слой использует sigmoid (ожидаемые значения от 0 до 1).")
        elif predictions.shape[-1] == CONFIG['num_classes']:
            print(
                f"  Выходной слой использует softmax (ожидаемые значения - вероятности для {CONFIG['num_classes']} классов, сумма по последней оси ~1).")
            print(f"  Пример предсказания для первого элемента батча: {predictions[0].numpy()}")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Тестирование image_classifier.py завершено ---")