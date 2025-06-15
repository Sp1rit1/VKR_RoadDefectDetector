# RoadDefectDetector/train_detector.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time  # <--- Добавляем импорт time

# --- Определяем корень проекта и добавляем src в sys.path для корректных импортов ---
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# --- Импорты из твоих модулей в src ---
from datasets.detector_data_loader import (
    create_detector_tf_dataset,
    # Глобальные переменные из detector_data_loader.py теперь не импортируются напрямую,
    # так как все необходимые параметры будут браться из конфигов.
)
from models.object_detector import build_object_detector_v1
from losses.detection_losses import compute_detector_loss_v1
from utils.callbacks import EpochTimeLogger  # Импортируем наш коллбэк

# --- Загрузка Конфигураций ---
_base_config_path = os.path.join(_project_root, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root, 'src', 'configs', 'detector_config.yaml')

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS_TRAIN_DET = True

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN_DET = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): base_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): YAML в base_config.yaml.")

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN_DET = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): detector_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): YAML в detector_config.yaml.")

if not CONFIG_LOAD_SUCCESS_TRAIN_DET:
    print("ОШИБКА: Не удалось загрузить один или несколько файлов конфигурации для train_detector.py. Выход.")
    # Задаем минимальные дефолты, чтобы скрипт мог хотя бы импортироваться другими модулями,
    # но обучение не запустится корректно.
    DETECTOR_CONFIG.setdefault('train_params', {'batch_size': 1, 'learning_rate': 0.0001, 'epochs': 10})
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('classes', ['pit', 'crack'])
    DETECTOR_CONFIG.setdefault('use_augmentation', False)
    BASE_CONFIG.setdefault('logs_base_dir', 'logs')
    BASE_CONFIG.setdefault('weights_base_dir', 'weights')
    BASE_CONFIG.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    # exit() # Можно раскомментировать, чтобы прервать выполнение, если конфиги критичны

# --- Параметры из Конфигов ---
# Пути к разделенному датасету детектора
_detector_dataset_ready_path_rel = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS = os.path.join(_project_root, _detector_dataset_ready_path_rel)

IMAGES_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')

# Параметры обучения
TRAIN_PARAMS_DET = DETECTOR_CONFIG.get('train_params', {})
BATCH_SIZE_DET = TRAIN_PARAMS_DET.get('batch_size', 2)  # Увеличил дефолт до 2
EPOCHS_DET = TRAIN_PARAMS_DET.get('epochs', 50)
LEARNING_RATE_DET = TRAIN_PARAMS_DET.get('learning_rate', 0.0001)
USE_AUGMENTATION_TRAIN = DETECTOR_CONFIG.get('use_augmentation', False)

# Параметры для логов и весов
LOGS_BASE_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('logs_base_dir', 'logs'))
WEIGHTS_BASE_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))


# Параметры модели для create_detector_tf_dataset (берутся из detector_data_loader при импорте)
# Нам нужно передать их явно или убедиться, что data_loader их правильно использует из своих глобальных переменных
# Для create_detector_tf_dataset параметры target_height, target_width, classes_list и т.д.
# уже установлены как глобальные переменные в detector_data_loader.py при его импорте,
# они читаются из тех же конфигов.


def collect_split_data_paths(split_dir_abs_path, images_subdir, annotations_subdir):
    """Собирает пути к изображениям и аннотациям для указанного разделения (train/val)."""
    image_paths = []
    xml_paths = []

    current_images_dir = os.path.join(split_dir_abs_path, images_subdir)
    current_annotations_dir = os.path.join(split_dir_abs_path, annotations_subdir)

    if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ: Директория изображений ({current_images_dir}) или аннотаций ({current_annotations_dir}) не найдена для этого split. Возвращаем пустые списки.")
        return image_paths, xml_paths  # Возвращаем пустые списки, чтобы не было ошибки дальше

    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files_in_split = []
    for ext in valid_extensions:
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.lower()}")))
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.upper()}")))

    image_files_in_split = sorted(list(set(image_files_in_split)))

    for img_path in image_files_in_split:
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        xml_path = os.path.join(current_annotations_dir, base_name + ".xml")
        if os.path.exists(xml_path):
            image_paths.append(img_path)
            xml_paths.append(xml_path)
        else:
            print(
                f"    ПРЕДУПРЕЖДЕНИЕ (collect_split): XML для {os.path.basename(img_path)} не найден в {current_annotations_dir}. Изображение пропущено для этого split.")

    return image_paths, xml_paths


def train_detector_main():
    if not CONFIG_LOAD_SUCCESS_TRAIN_DET:
        print("Критическая ошибка: не удалось загрузить файлы конфигурации. Обучение детектора прервано.")
        return

    print("\n--- Обучение Детектора Объектов (Кастомная Модель v1 с Train/Val) ---")

    # --- ИЗМЕНЕНИЕ: Смешанная точность (опционально, но может ускорить) ---
    # Раскомментируй, если твой GPU поддерживает (NVIDIA Volta, Turing, Ampere и новее)
    # from tensorflow.keras import mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # print("INFO: Включена смешанная точность (mixed_float16).")
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # 1. Сбор путей к данным из разделенных папок
    train_split_dir = os.path.join(DETECTOR_DATASET_READY_ABS, "train")
    val_split_dir = os.path.join(DETECTOR_DATASET_READY_ABS, "validation")

    print(f"\nСбор обучающих данных из: {train_split_dir}")
    train_image_paths, train_xml_paths = collect_split_data_paths(train_split_dir, IMAGES_SUBDIR_NAME_DET,
                                                                  ANNOTATIONS_SUBDIR_NAME_DET)

    print(f"\nСбор валидационных данных из: {val_split_dir}")
    val_image_paths, val_xml_paths = collect_split_data_paths(val_split_dir, IMAGES_SUBDIR_NAME_DET,
                                                              ANNOTATIONS_SUBDIR_NAME_DET)

    if not train_image_paths:
        print("ОШИБКА: Обучающие данные не найдены. Убедитесь, что 'create_data_splits.py' был успешно запущен.")
        print(
            f"Ожидаемая директория: {train_split_dir}/{IMAGES_SUBDIR_NAME_DET} и {train_split_dir}/{ANNOTATIONS_SUBDIR_NAME_DET}")
        return

    print(f"\nНайдено для обучения: {len(train_image_paths)} изображений.")
    if val_image_paths:
        print(f"Найдено для валидации: {len(val_image_paths)} изображений.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены или пусты. Обучение будет без валидации на лету.")

    # 2. Создание датасетов
    print(f"\nСоздание TensorFlow датасетов...")
    print(f"  Параметры для датасета: Batch Size={BATCH_SIZE_DET}, Аугментация для train={USE_AUGMENTATION_TRAIN}")

    train_dataset_detector = create_detector_tf_dataset(
        train_image_paths,
        train_xml_paths,
        batch_size=BATCH_SIZE_DET,
        shuffle=True,
        augment=USE_AUGMENTATION_TRAIN
    )

    validation_dataset_detector = None
    if val_image_paths:
        validation_dataset_detector = create_detector_tf_dataset(
            val_image_paths,
            val_xml_paths,
            batch_size=BATCH_SIZE_DET,
            shuffle=False,
            augment=False
        )

    if train_dataset_detector is None:
        print("Не удалось создать обучающий датасет для детектора. Обучение прервано.")
        return

    try:  # Проверка, что датасеты не пусты
        for _ in train_dataset_detector.take(1): pass
        print("Обучающий датасет для детектора успешно создан и содержит данные.")
        if validation_dataset_detector:
            for _ in validation_dataset_detector.take(1): pass
            print("Валидационный датасет для детектора успешно создан и содержит данные.")
    except Exception as e_ds_check:
        print(f"ОШИБКА: Датасет для детектора пуст или произошла ошибка при доступе: {e_ds_check}")
        return

    # 3. Создание и компиляция модели
    print("\nСоздание модели детектора (build_object_detector_v1)...")
    model = build_object_detector_v1()
    print("\nСтруктура модели детектора:")
    model.summary(line_length=120)  # Сделаем вывод summary более широким

    print(f"\nКомпиляция модели с learning_rate = {LEARNING_RATE_DET}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DET),
                  loss=compute_detector_loss_v1)

    # 4. Callbacks
    log_dir = os.path.join(LOGS_BASE_DIR_ABS, "detector_fit_v1_full", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    os.makedirs(WEIGHTS_BASE_DIR_ABS, exist_ok=True)

    # --- ИЗМЕНЕНИЕ: Добавляем наш EpochTimeLogger ---
    epoch_time_logger_callback = EpochTimeLogger()
    callbacks_list = [tensorboard_callback, epoch_time_logger_callback]
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    checkpoint_filepath_best = None  # Инициализируем
    if validation_dataset_detector:
        checkpoint_filepath_best = os.path.join(WEIGHTS_BASE_DIR_ABS, 'detector_v1_best_val_loss.keras')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_best,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)
        print(f"Лучшая модель будет сохраняться в: {checkpoint_filepath_best} (по val_loss)")

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=DETECTOR_CONFIG.get('train_params', {}).get('early_stopping_patience', 15),
            # Берем из конфига или дефолт
            verbose=1,
            restore_best_weights=True)
        callbacks_list.append(early_stopping_callback)

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=DETECTOR_CONFIG.get('train_params', {}).get('lr_factor', 0.2),
            patience=DETECTOR_CONFIG.get('train_params', {}).get('lr_patience', 5),
            verbose=1,
            min_lr=DETECTOR_CONFIG.get('train_params', {}).get('min_lr', 1e-7))
        callbacks_list.append(reduce_lr_callback)
    else:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен. ModelCheckpoint(save_best_only) и EarlyStopping по val_loss не будут эффективно использованы.")
        print("                 Будет сохранена только финальная модель после всех эпох.")

    # 5. Запуск обучения
    print(f"\nЗапуск обучения детектора на {EPOCHS_DET} эпох...")
    print(f"  Обучающая выборка: {len(train_image_paths)} изображений, Аугментация: {USE_AUGMENTATION_TRAIN}")
    if val_image_paths:
        print(f"  Валидационная выборка: {len(val_image_paths)} изображений")
    print(f"  Batch Size: {BATCH_SIZE_DET}")
    print(f"  Логи TensorBoard: {log_dir}")

    # --- ИЗМЕНЕНИЕ: Замеряем общее время обучения ---
    training_start_time = time.time()
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    try:
        history = model.fit(
            train_dataset_detector,
            epochs=EPOCHS_DET,
            validation_data=validation_dataset_detector,
            callbacks=callbacks_list,
            verbose=1  # verbose=1 для стандартного Keras прогресс-бара + нашего времени
            # verbose=2 если будешь использовать EpochTimeLoggerPretty
        )

        # --- ИЗМЕНЕНИЕ: Вычисляем и выводим общее время обучения ---
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print(
            f"\n--- Общее время обучения детектора: {total_training_time // 3600:.0f} ч {(total_training_time % 3600) // 60:.0f} мин {total_training_time % 60:.2f} сек ---")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        print("\n--- Обучение детектора (v1) завершено ---")

        final_model_save_path = os.path.join(WEIGHTS_BASE_DIR_ABS, 'detector_v1_final_after_full_train.keras')
        model.save(final_model_save_path)
        print(
            f"Финальная модель (после всех эпох или EarlyStopping с restore_best_weights) сохранена в: {final_model_save_path}")

        if validation_dataset_detector and checkpoint_filepath_best and os.path.exists(checkpoint_filepath_best):
            print(f"Лучшая модель по val_loss также сохранена в: {checkpoint_filepath_best}")
        elif validation_dataset_detector and checkpoint_filepath_best:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Ожидался файл лучшей модели {checkpoint_filepath_best}, но он не найден. Возможно, обучение было прервано до первого сохранения лучшей модели.")


    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit() для детектора: {e_fit}")
        import traceback
        traceback.print_exc()
        # --- ИЗМЕНЕНИЕ: Вывод времени даже если была ошибка ---
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print(
            f"\n--- Обучение прервано. Затраченное время: {total_training_time // 3600:.0f} ч {(total_training_time % 3600) // 60:.0f} мин {total_training_time % 60:.2f} сек ---")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---


if __name__ == '__main__':
    if not CONFIG_LOAD_SUCCESS_TRAIN_DET:
        print("\nОбучение не может быть запущено из-за ошибок загрузки конфигурации.")
    else:
        train_detector_main()