# train_detector.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys

# --- Определяем корень проекта и добавляем src в sys.path для корректных импортов ---
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# --- Импорты из твоих модулей в src ---
from datasets.detector_data_loader import (
    create_detector_tf_dataset,
    MASTER_DATASET_PATH_ABS as CFG_MASTER_DATASET_PATH,  # Путь к мастер-датасету из data_loader'а
    _images_subdir_name_cfg as CFG_IMAGES_SUBDIR,  # Имя подпапки с картинками из data_loader'а
    _annotations_subdir_name_cfg as CFG_ANNOTATIONS_SUBDIR  # Имя подпапки с аннотациями из data_loader'а
)
from models.object_detector import build_object_detector_v1
from losses.detection_losses import compute_detector_loss_v1

# --- Загрузка Конфигураций ---
_base_config_path = os.path.join(_project_root, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root, 'src', 'configs', 'detector_config.yaml')

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS_TRAIN = True

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_TRAIN = False; print(f"ОШИБКА: base_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_TRAIN = False; print(f"ОШИБКА: YAML в base_config.yaml.")

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_TRAIN = False; print(f"ОШИБКА: detector_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_TRAIN = False; print(f"ОШИБКА: YAML в detector_config.yaml.")

if not CONFIG_LOAD_SUCCESS_TRAIN:
    print(
        "ОШИБКА: Не удалось загрузить один или несколько файлов конфигурации. Используются дефолты, что может быть некорректно.")
    # Задаем минимальные дефолты, чтобы скрипт не упал сразу, но это нужно исправить
    DETECTOR_CONFIG.setdefault('train_params', {'batch_size': 1, 'learning_rate': 0.0001, 'epochs_test_overfit': 10})
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    BASE_CONFIG.setdefault('logs_base_dir', 'logs')
    BASE_CONFIG.setdefault('weights_base_dir', 'weights')


def collect_all_data_paths():
    """Собирает пути ко всем изображениям и аннотациям из мастер-датасета."""
    source_subfolders_keys = [
        'source_defective_road_img_parent_subdir',
        'source_normal_road_img_parent_subdir',
        'source_not_road_img_parent_subdir'
    ]
    default_subfolder_names = {
        'source_defective_road_img_parent_subdir': "Defective_Road_Images",
        'source_normal_road_img_parent_subdir': "Normal_Road_Images",
        'source_not_road_img_parent_subdir': "Not_Road_Images"
    }

    all_image_paths = []
    all_xml_paths = []

    print(f"\n--- Сбор путей к данным для детектора ---")
    print(f"Корневая папка мастер-датасета: {CFG_MASTER_DATASET_PATH}")

    for subfolder_key in source_subfolders_keys:
        subfolder_name = BASE_CONFIG.get(subfolder_key, default_subfolder_names.get(subfolder_key))
        if not subfolder_name:  # Если ключ не найден и нет дефолта
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Ключ для подпапки {subfolder_key} не найден в base_config.yaml. Пропускаем.")
            continue

        current_images_dir = os.path.join(CFG_MASTER_DATASET_PATH, subfolder_name, CFG_IMAGES_SUBDIR)
        current_annotations_dir = os.path.join(CFG_MASTER_DATASET_PATH, subfolder_name, CFG_ANNOTATIONS_SUBDIR)

        # print(f"  Проверка директории изображений: {current_images_dir}")
        # print(f"  Проверка директории аннотаций: {current_annotations_dir}")

        if not os.path.isdir(current_images_dir):  # Проверяем только папку с изображениями
            # print(f"    ПРЕДУПРЕЖДЕНИЕ: Директория изображений {current_images_dir} не найдена. Пропускаем категорию '{subfolder_name}'.")
            continue
        if not os.path.isdir(current_annotations_dir):  # Проверяем только папку с изображениями
            # print(f"    ПРЕДУПРЕЖДЕНИЕ: Директория аннотаций {current_annotations_dir} не найдена. Пропускаем категорию '{subfolder_name}'.")
            continue

        valid_extensions = ['.jpg', '.jpeg', '.png']
        image_files_in_subdir = []
        for ext in valid_extensions:
            image_files_in_subdir.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.lower()}")))
            image_files_in_subdir.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.upper()}")))

        # Удаляем дубликаты, если glob нашел и .jpg и .JPG
        image_files_in_subdir = sorted(list(set(image_files_in_subdir)))

        if image_files_in_subdir:
            print(f"  Найдено {len(image_files_in_subdir)} изображений в {current_images_dir}")
            for img_path in image_files_in_subdir:
                base_name, _ = os.path.splitext(os.path.basename(img_path))
                xml_path = os.path.join(current_annotations_dir, base_name + ".xml")
                if os.path.exists(xml_path):
                    all_image_paths.append(img_path)
                    all_xml_paths.append(xml_path)
                # else:
                # print(f"    ПРЕДУПРЕЖДЕНИЕ: XML для {os.path.basename(img_path)} не найден в {current_annotations_dir}")
        # else:
        # print(f"    Изображения не найдены в {current_images_dir}")

    if not all_image_paths:
        print("\nОШИБКА: Не найдено ни одной валидной пары изображение/аннотация во всех категориях.")
        print(
            "Проверьте структуру папок и пути в base_config.yaml (master_dataset_path, source_*_subdirs, images_dir, annotations_dir).")
    else:
        print(f"\nВсего найдено {len(all_image_paths)} пар изображение/аннотация для датасета детектора.")

    return all_image_paths, all_xml_paths


def train_detector_main():
    print("\n--- Обучение Детектора Объектов (Кастомная Модель v1) ---")

    # 1. Сбор всех данных (пока без разделения на train/val для теста на переобучение)
    image_paths, xml_paths = collect_all_data_paths()

    if not image_paths:
        return

    # Для теста на переобучение используем все найденные данные (их должно быть мало)
    # В будущем здесь будет разделение на train/val
    train_image_paths = image_paths
    train_xml_paths = xml_paths
    # val_image_paths, val_xml_paths = [], [] # Пока валидация не используется для этого теста

    print(f"Используется {len(train_image_paths)} изображений для теста на переобучение.")
    if len(train_image_paths) == 0:
        print("Нет данных для обучения. Выход.")
        return
    if len(train_image_paths) > 20:  # Ограничение для теста на переобучение
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Слишком много данных для простого теста на переобучение. Будет использовано не более 20.")
        # Можно добавить логику выбора случайных 20, но пока просто обрежем
        # train_image_paths = train_image_paths[:20]
        # train_xml_paths = train_xml_paths[:20]
        # print(f"Используется {len(train_image_paths)} изображений после ограничения.")

    # 2. Создание датасета
    batch_size_cfg = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 1)
    # Для теста на переобучение на очень малом датасете (3-6 картинок) лучше batch_size=1
    # Если у тебя всего 3 картинки, то batch_size должен быть 1.
    # Если картинок больше, можно попробовать batch_size из конфига.
    # Мы УЖЕ установили BATCH_SIZE = 1 в detector_data_loader.py для теста, так что это должно быть согласовано.
    # Но detector_data_loader.py теперь читает batch_size из конфига.
    # Убедимся, что для этого теста batch_size = 1
    current_test_batch_size = 1
    if len(train_image_paths) < current_test_batch_size:  # Если файлов меньше чем батч
        print(
            f"Количество файлов ({len(train_image_paths)}) меньше чем тестовый batch_size ({current_test_batch_size}). Пропуск обучения.")
        return

    print(f"\nСоздание TensorFlow датасета с batch_size = {current_test_batch_size}...")
    train_dataset_detector = create_detector_tf_dataset(
        train_image_paths,
        train_xml_paths,
        batch_size=current_test_batch_size,  # Используем тестовый batch_size
        # target_height/width/classes берутся из глобальных переменных внутри detector_data_loader
        shuffle=True  # Перемешиваем даже для теста
    )

    if train_dataset_detector is None:
        print("Не удалось создать датасет для детектора. Обучение прервано.")
        return

    # Проверка, что датасет не пуст
    try:
        for _ in train_dataset_detector.take(1): pass
        print("Датасет для детектора успешно создан и содержит данные.")
    except Exception as e_ds_check:
        print(f"ОШИБКА: Датасет для детектора пуст или произошла ошибка при доступе: {e_ds_check}")
        return

    # 3. Создание и компиляция модели
    print("\nСоздание модели детектора (build_object_detector_v1)...")
    model = build_object_detector_v1()  # Используем новую модель
    print("\nСтруктура модели детектора:")
    model.summary(line_length=120)

    learning_rate_cfg = DETECTOR_CONFIG.get('train_params', {}).get('learning_rate', 0.0001)
    print(f"\nКомпиляция модели с learning_rate = {learning_rate_cfg}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_cfg),
                  loss=compute_detector_loss_v1)  # Используем новую функцию потерь

    # 4. Callbacks
    logs_dir_abs = os.path.join(_project_root, BASE_CONFIG.get('logs_base_dir', 'logs'),
                                "detector_fit_v1_overfit_test", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir_abs, histogram_freq=1)

    weights_dir_abs = os.path.join(_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))
    os.makedirs(weights_dir_abs, exist_ok=True)
    checkpoint_filepath = os.path.join(weights_dir_abs,
                                       'detector_v1_overfit_test_epoch_{epoch:02d}.keras')  # Сохраняем .keras

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  # Сохраняем всю модель
        save_freq='epoch'  # Сохраняем после каждой эпохи для теста
    )
    callbacks_list = [tensorboard_callback, model_checkpoint_callback]

    # 5. Запуск обучения (тест на переобучение)
    epochs_to_run = DETECTOR_CONFIG.get('train_params', {}).get('epochs_test_overfit', 100)  # Из конфига
    print(f"\nЗапуск обучения детектора на {epochs_to_run} эпох (тест на переобучение)...")
    print(f"Логи TensorBoard: {logs_dir_abs}")
    print(f"Модели будут сохраняться в {weights_dir_abs} с префиксом detector_v1_overfit_test_epoch_")

    try:
        history = model.fit(
            train_dataset_detector,
            epochs=epochs_to_run,
            callbacks=callbacks_list,
            verbose=1
        )
        print("\n--- Тестовое обучение детектора (v1) завершено ---")

        final_model_path = os.path.join(weights_dir_abs, 'detector_v1_overfit_final.keras')
        model.save(final_model_path)
        print(f"Финальная модель сохранена в: {final_model_path}")

    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit() для детектора: {e_fit}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()