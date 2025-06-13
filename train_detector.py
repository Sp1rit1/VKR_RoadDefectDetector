# train_detector.py (находится в КОРНЕ проекта RoadDefectDetector/)
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
from datasets.detector_data_loader import create_detector_tf_dataset, \
    MASTER_DATASET_PATH_ABS as DETECTOR_DATA_MASTER_PATH
from models.object_detector import build_simple_object_detector
from losses.detection_losses import simple_detector_loss

# --- Загрузка Конфигураций ---
_base_config_path = os.path.join(_project_root, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root, 'src', 'configs', 'detector_config.yaml')

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
except FileNotFoundError as e:
    print(f"ОШИБКА: Не найден один из файлов конфигурации для train_detector.py.")
    print(f"Проверьте пути: \n{os.path.abspath(_base_config_path)}\n{os.path.abspath(_detector_config_path)}")
    print(f"Детали ошибки: {e}")
    exit()
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл конфигурации для train_detector.py.")
    print(f"Детали ошибки: {e}")
    exit()


def train_detector_main():
    print("--- Обучение Детектора Объектов (Тест на ОЧЕНЬ МАЛОМ датасете) ---")

    # 1. Подготовка данных
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
    source_subfolders_to_scan = [
        BASE_CONFIG.get(key, default_subfolder_names[key]) for key in source_subfolders_keys
    ]
    images_subfolder_name = BASE_CONFIG.get('images_dir', "JPEGImages")
    annotations_subfolder_name = BASE_CONFIG.get('annotations_dir', "Annotations")

    print(f"Сканирование мастер-датасета в: {DETECTOR_DATA_MASTER_PATH}")
    print(f"Ожидаемые подпапки с данными: {source_subfolders_to_scan}")
    print(f"Имена папок с изображениями/аннотациями: {images_subfolder_name} / {annotations_subfolder_name}\n")

    all_image_paths = []
    all_xml_paths = []
    found_any_data_category = False

    for subfolder_name in source_subfolders_to_scan:
        current_images_dir = os.path.join(DETECTOR_DATA_MASTER_PATH, subfolder_name, images_subfolder_name)
        current_annotations_dir = os.path.join(DETECTOR_DATA_MASTER_PATH, subfolder_name, annotations_subfolder_name)

        if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
            print(
                f"    ПРЕДУПРЕЖДЕНИЕ: Директория {current_images_dir} или {current_annotations_dir} не найдена. Пропускаем категорию '{subfolder_name}'.")
            continue

            valid_extensions = ['.jpg', '.jpeg', '.png']
        image_files_in_subdir = []
        for ext in valid_extensions:
            image_files_in_subdir.extend(glob.glob(os.path.join(current_images_dir, f"*{ext}")))

        if image_files_in_subdir:
            found_any_data_category = True
            for img_path in image_files_in_subdir:
                base_name, _ = os.path.splitext(os.path.basename(img_path))
                xml_path = os.path.join(current_annotations_dir, base_name + ".xml")
                if os.path.exists(xml_path):
                    all_image_paths.append(img_path)
                    all_xml_paths.append(xml_path)

    if not all_image_paths:  # Изменил проверку, теперь она после цикла
        if not found_any_data_category:
            print(
                f"ОШИБКА: Не найдено изображений ни в одной из ожидаемых подпапок категорий внутри {DETECTOR_DATA_MASTER_PATH}.")
        else:
            print("ОШИБКА: Изображения найдены, но не найдено ни одной СОВПАДАЮЩЕЙ ПАРЫ изображение/аннотация.")
        print("Убедитесь, что для изображений есть соответствующие XML файлы с тем же именем.")
        return

    print(f"Найдено {len(all_image_paths)} пар изображение/аннотация для обучения детектора.")

    # Параметры из конфига
    # Для очень маленького датасета, убедимся, что batch_size = 1
    current_batch_size = 1  # Принудительно ставим 1 для теста на 3х картинках
    print(
        f"ПРИНУДИТЕЛЬНЫЙ BATCH_SIZE = 1 для теста на малом датасете (значение из detector_config.yaml игнорируется для этого теста).")

    epochs_to_run = DETECTOR_CONFIG['train_params'].get('epochs_test_overfit', 50)  # Уменьшим дефолт для быстрого теста
    learning_rate_cfg = DETECTOR_CONFIG['train_params']['learning_rate']

    if not all_image_paths:  # Повторная проверка на всякий случай
        print("Нет файлов для создания датасета.")
        return

    print(f"Используем {len(all_image_paths)} файлов, batch_size={current_batch_size}, эпох для теста={epochs_to_run}")

    # Создаем датасет только из найденных файлов
    train_dataset_detector = create_detector_tf_dataset(
        all_image_paths,  # Список найденных изображений
        all_xml_paths,  # Список найденных XML
        batch_size=current_batch_size,
        target_height_ds=DETECTOR_CONFIG['input_shape'][0],
        target_width_ds=DETECTOR_CONFIG['input_shape'][1],
        shuffle=True  # Перемешиваем, даже если мало данных
    )

    if train_dataset_detector is None:
        print("Не удалось создать датасет для детектора. Обучение прервано.")
        return

    # Проверим, что датасет не пустой
    try:
        for _ in train_dataset_detector.take(1):  # Попытка взять один элемент
            pass
        print("Датасет для детектора успешно создан и содержит данные.")
    except tf.errors.OutOfRangeError:
        print("ОШИБКА: Датасет для детектора пуст после создания. Проверьте detector_data_loader.py и исходные файлы.")
        return
    except Exception as e_ds_check:
        print(f"ОШИБКА при проверке датасета детектора: {e_ds_check}")
        return

    model = build_simple_object_detector()
    print("\nСтруктура модели детектора:")
    model.summary(line_length=120)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_cfg),
                  loss=simple_detector_loss)

    logs_dir_abs = os.path.join(_project_root, BASE_CONFIG.get('logs_base_dir', 'logs'),
                                "detector_fit_tiny_test", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir_abs, histogram_freq=1)

    weights_dir_abs = os.path.join(_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))
    os.makedirs(weights_dir_abs, exist_ok=True)
    checkpoint_filepath = os.path.join(weights_dir_abs, 'detector_simple_tiny_overfit_test.keras')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_freq='epoch'
    )
    callbacks_list = [tensorboard_callback, model_checkpoint_callback]

    print(f"\nЗапуск обучения детектора на {epochs_to_run} эпох (тест на переобучение на ОЧЕНЬ МАЛОМ датасете)...")
    print(f"Начальная скорость обучения: {learning_rate_cfg}")
    print(f"Логи TensorBoard будут сохраняться в: {logs_dir_abs}")
    print(f"Модель будет сохраняться в: {checkpoint_filepath} после каждой эпохи.")

    try:
        history = model.fit(
            train_dataset_detector,
            epochs=epochs_to_run,
            callbacks=callbacks_list,
            verbose=1
        )
        print("\n--- Тестовое обучение детектора завершено ---")

        final_model_path = os.path.join(weights_dir_abs, 'detector_simple_tiny_overfit_final.keras')
        model.save(final_model_path)
        print(f"Финальная модель сохранена в: {final_model_path}")

    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit() для детектора: {e_fit}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Убедись, что у тебя есть 3 файла изображений и 3 XML аннотации
    # в соответствующих подпапках внутри data/Master_Dataset/
    # Например:
    # data/Master_Dataset/Defective_Road_Images/JPEGImages/defect_example.jpg
    # data/Master_Dataset/Defective_Road_Images/Annotations/defect_example.xml (с разметкой pit или crack)
    # data/Master_Dataset/Normal_Road_Images/JPEGImages/normal_example.jpg
    # data/Master_Dataset/Normal_Road_Images/Annotations/normal_example.xml (без <object>)
    # data/Master_Dataset/Not_Road_Images/JPEGImages/notroad_example.jpg
    # data/Master_Dataset/Not_Road_Images/Annotations/notroad_example.xml (без <object>)
    #
    # И что detector_config.yaml содержит:
    # train_params:
    #   batch_size: 1 # ВАЖНО для такого маленького датасета
    #   epochs_test_overfit: 50 # или 100, для теста
    #   learning_rate: 0.0001
    train_detector_main()