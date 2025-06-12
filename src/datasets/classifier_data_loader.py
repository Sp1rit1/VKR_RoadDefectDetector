# RoadDefectDetector/src/datasets/classifier_data_loader.py
import tensorflow as tf
import yaml
import os
import glob  # Добавим glob для проверки наличия файлов

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_classifier_config_path = os.path.join(_current_dir, '..', 'configs', 'classifier_config.yaml')

try:
    with open(_classifier_config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ОШИБКА: Файл конфигурации классификатора не найден: {_classifier_config_path}")
    CONFIG = {
        'input_shape': [224, 224, 3],
        'prepared_dataset_path': 'data/Classifier_Dataset_Fallback',
        'train_params': {'batch_size': 1}  # Изменено на 1 для соответствия твоему конфигу
    }
    print(f"ПРЕДУПРЕЖДЕНИЕ: {_classifier_config_path} не найден. Используются значения по умолчанию.")
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл classifier_config.yaml: {e}")
    CONFIG = {
        'input_shape': [224, 224, 3],
        'prepared_dataset_path': 'data/Classifier_Dataset_Fallback',
        'train_params': {'batch_size': 1}
    }
    print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка чтения classifier_config.yaml. Используются значения по умолчанию.")

# --- Параметры из Конфига ---
IMG_HEIGHT = CONFIG.get('input_shape', [224, 224, 3])[0]
IMG_WIDTH = CONFIG.get('input_shape', [224, 224, 3])[1]
BATCH_SIZE = CONFIG.get('train_params', {}).get('batch_size', 1)  # Убедимся, что дефолт 1

_current_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
DATASET_PATH_FROM_CONFIG = CONFIG.get('prepared_dataset_path', 'data/Classifier_Dataset_Default')
PREPARED_DATASET_ABS_PATH = os.path.join(_current_project_root, DATASET_PATH_FROM_CONFIG)


def check_if_dir_has_images(directory_path):
    """Проверяет, есть ли изображения в указанной директории (и ее поддиректориях первого уровня)."""
    if not os.path.isdir(directory_path):
        return False

    # Проверяем наличие поддиректорий (классов)
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    if not subdirs:  # Если нет поддиректорий классов
        return False

    for subdir in subdirs:
        class_dir_path = os.path.join(directory_path, subdir)
        for ext_pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            if glob.glob(os.path.join(class_dir_path, ext_pattern)):
                return True  # Нашли хотя бы одно изображение в одном из классов
    return False


def create_classifier_datasets():
    if not os.path.isdir(PREPARED_DATASET_ABS_PATH):
        print(f"ОШИБКА: Директория подготовленного датасета классификатора не найдена: {PREPARED_DATASET_ABS_PATH}")
        return None, None, None

    train_dir = os.path.join(PREPARED_DATASET_ABS_PATH, 'train')
    val_dir = os.path.join(PREPARED_DATASET_ABS_PATH, 'validation')

    # Проверяем, есть ли изображения в обучающей директории
    if not check_if_dir_has_images(train_dir):
        print(f"ОШИБКА: Директория 'train' ({train_dir}) не содержит изображений или подпапок классов с изображениями.")
        print("Убедитесь, что скрипт 'prepare_classifier_dataset.py' отработал корректно.")
        return None, None, None

    train_dataset = None
    validation_dataset = None
    class_names = None

    try:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode='int'
        )
        class_names = train_dataset.class_names  # Получаем имена классов из обучающего набора
        print(f"Найдены классы для классификатора: {class_names}")

        # Проверяем, есть ли изображения в валидационной директории
        if check_if_dir_has_images(val_dir):
            print(f"Загрузка валидационного датасета из: {val_dir}")
            validation_dataset = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                label_mode='int',
                # Важно: если классы в val могут отличаться от train, это проблема.
                # Но image_dataset_from_directory сам разберется, если папки классов есть.
                # Если в val нет каких-то классов из train, это может быть проблемой для некоторых метрик.
                # Keras ожидает, что набор классов консистентен.
                # Мы можем передать class_names из train_dataset, если нужно строгое соответствие
                # class_names=class_names # Это может вызвать ошибку, если в val нет папки для какого-то класса из train
            )
            if not validation_dataset.class_names:  # Если классы не определились (папки пусты)
                print(
                    f"ПРЕДУПРЕЖДЕНИЕ: Валидационная директория {val_dir} не содержит изображений в подпапках классов.")
                validation_dataset = None  # Считаем, что валидационного датасета нет
            elif set(validation_dataset.class_names) != set(class_names):
                print(
                    f"ПРЕДУПРЕЖДЕНИЕ: Набор классов в train ({class_names}) и validation ({validation_dataset.class_names}) не совпадает!")
                # В этом случае лучше не использовать validation_dataset или исправить данные
                validation_dataset = None


        else:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Валидационная директория {val_dir} не содержит изображений или подпапок классов. Валидационный датасет не будет создан.")
            validation_dataset = None

    except Exception as e:
        print(f"ОШИБКА при создании датасетов с помощью image_dataset_from_directory: {e}")
        return None, None, None

    AUTOTUNE = tf.data.AUTOTUNE

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    if train_dataset:
        train_dataset = train_dataset.map(normalize_img, num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.cache().shuffle(buffer_size=max(100, BATCH_SIZE * 5),
                                                      reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)

    if validation_dataset:
        validation_dataset = validation_dataset.map(normalize_img, num_parallel_calls=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, class_names


if __name__ == '__main__':
    print(f"--- Тестирование classifier_data_loader.py ---")
    print(f"Загрузка данных из (ожидаемый абсолютный путь): {PREPARED_DATASET_ABS_PATH}")
    print(f"Ожидаемый размер изображений: ({IMG_HEIGHT}, {IMG_WIDTH}), Размер батча: {BATCH_SIZE}")

    train_ds, val_ds, class_names_loaded = create_classifier_datasets()

    if train_ds:
        print(f"\nУспешно создан обучающий датасет.")
        print(f"Имена классов (и их порядок ID): {class_names_loaded}")
        if len(class_names_loaded) >= 2:
            print(f"  Метка для класса '{class_names_loaded[0]}' будет 0.")
            print(f"  Метка для класса '{class_names_loaded[1]}' будет 1.")

        print("\nПример первого батча из обучающего датасета:")
        try:
            for images, labels in train_ds.take(1):
                print("  Форма батча изображений:", images.shape)
                print("  Тип данных изображений:", images.dtype)
                print("  Мин/Макс значения пикселей:", tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())
                print("  Форма батча меток:", labels.shape)
                print("  Тип данных меток:", labels.dtype)
                print("  Пример меток в батче:", labels.numpy()[:min(5, BATCH_SIZE)])
        except Exception as e:
            print(f"ОШИБКА при итерации по обучающему датасету: {e}")
    else:
        print("\nНе удалось создать обучающий датасет.")

    if val_ds:
        print(f"\nУспешно создан валидационный датасет.")
        print("\nПример первого батча из валидационного датасета:")
        try:
            for images, labels in val_ds.take(1):
                print("  Форма батча изображений:", images.shape)
                print("  Форма батча меток:", labels.shape)
                print("  Пример меток в батче:", labels.numpy()[:min(5, BATCH_SIZE)])
        except Exception as e:
            print(f"ОШИБКА при итерации по валидационному датасету: {e}")
    else:
        print("\nВалидационный датасет не создан или пуст.")

    print("\n--- Тестирование classifier_data_loader.py завершено ---")