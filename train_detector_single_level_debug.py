# train_detector_single_level_debug.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# --- Определяем корень проекта и добавляем src в sys.path ---
_project_root_sdl_train = Path(__file__).resolve().parent  # Корень проекта, где лежит этот скрипт
_src_path_sdl_train = _project_root_sdl_train / 'src'
if str(_src_path_sdl_train) not in sys.path:
    sys.path.insert(0, str(_src_path_sdl_train))
if str(_project_root_sdl_train) not in sys.path:
    sys.path.insert(0, str(_project_root_sdl_train))

# --- Импорты из ОТЛАДОЧНЫХ (_single_level_debug) модулей ---
try:
    from datasets.detector_data_loader_single_level_debug import (
        create_detector_single_level_tf_dataset,
        DEBUG_SDL_CONFIG_GLOBAL,  # Загруженный отладочный конфиг
        BASE_CONFIG_SDL_GLOBAL,  # Загруженный базовый конфиг
        TARGET_IMG_HEIGHT_SDL_G, TARGET_IMG_WIDTH_SDL_G,  # Для информации
        FPN_LEVEL_NAME_DEBUG_SDL_G  # Имя отладочного уровня
    )
    from models.object_detector_single_level_debug import build_detector_single_level_p4_debug
    from losses.detection_losses_single_level_debug import compute_detector_loss_single_level_debug

    print("INFO (train_sdl): Отладочные компоненты (data_loader, model, loss) успешно импортированы.")
except ImportError as e_imp_sdl_train:
    print(f"КРИТИЧЕСКАЯ ОШИБКА (train_sdl): Не удалось импортировать отладочные компоненты: {e_imp_sdl_train}")
    import traceback

    traceback.print_exc()
    exit()

# --- Параметры из ОТЛАДОЧНОГО Конфига (DEBUG_SDL_CONFIG_GLOBAL) ---
# Он уже загружен в detector_data_loader_single_level_debug.py, мы его импортировали.
# Используем его для параметров обучения, так как это обучение ОТЛАДОЧНОЙ модели.

# Параметры для датасета (пути к разделенному датасету)
_detector_dataset_ready_path_rel_sdl = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_SDL = _project_root_sdl_train / _detector_dataset_ready_path_rel_sdl
IMAGES_SUBDIR_NAME_SDL_TRAIN = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('annotations_dir', 'Annotations')

# Параметры обучения из отладочного конфига
MODEL_BASE_NAME_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('fpn_detector_params', {}).get('model_name_prefix',
                                                                                       "RoadDefectDetector_Debug_P4_Only")
BATCH_SIZE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('batch_size',
                                                   2)  # Берем batch_size из корневого уровня отладочного конфига
EPOCHS_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('epochs_for_debug', 50)  # Используем epochs_for_debug
LEARNING_RATE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('initial_learning_rate', 0.0001)
USE_AUGMENTATION_TRAIN_SDL = DEBUG_SDL_CONFIG_GLOBAL.get('use_augmentation',
                                                         False)  # Используем флаг из отладочного конфига

# Параметры Callbacks (можно взять из основного detector_config или определить здесь)
# Для простоты, возьмем основные из отладочного конфига, если есть, или дефолты
EARLY_STOPPING_PATIENCE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('early_stopping_patience', 15)
REDUCE_LR_PATIENCE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('reduce_lr_patience', 5)
REDUCE_LR_FACTOR_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('reduce_lr_factor', 0.2)
MIN_LR_ON_PLATEAU_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('min_lr_on_plateau', 1e-7)

# Пути для логов и весов (используем из основного base_config, но с префиксом)
LOGS_BASE_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('logs_base_dir', 'logs')
WEIGHTS_BASE_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('weights_base_dir', 'weights')
GRAPHS_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('graphs_dir', 'graphs')
os.makedirs(GRAPHS_DIR_ABS_SDL, exist_ok=True)
os.makedirs(WEIGHTS_BASE_DIR_ABS_SDL, exist_ok=True)
os.makedirs(LOGS_BASE_DIR_ABS_SDL, exist_ok=True)


# --- Вспомогательные Функции ---
def collect_split_data_paths_sdl(split_dir_abs_path, images_subdir, annotations_subdir):
    # ... (Эта функция такая же, как в основном train_detector.py, можно скопировать)
    image_paths = []
    xml_paths = []
    current_images_dir = os.path.join(split_dir_abs_path, images_subdir)
    current_annotations_dir = os.path.join(split_dir_abs_path, annotations_subdir)
    if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ (sdl_train): Директория {current_images_dir} или {current_annotations_dir} не найдена для split'а. Пропускаем.")
        return image_paths, xml_paths
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
    return image_paths, xml_paths


class EpochTimeLoggerSDL(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f" - Время эпохи: {time.time() - self.epoch_start_time:.2f} сек")


def plot_training_history_sdl(history, save_path_plot, run_suffix=""):
    # ... (Эта функция такая же, как в основном train_detector.py)
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss']:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    title = f'История обучения отладочного детектора ({run_suffix})' if run_suffix else 'История обучения отладочного детектора'
    plt.title(title);
    plt.ylabel('Loss');
    plt.xlabel('Epoch')
    plt.legend(loc='upper right');
    plt.grid(True);
    plt.tight_layout()
    try:
        plt.savefig(save_path_plot); print(f"График истории обучения сохранен в: {save_path_plot}")
    except Exception as e:
        print(f"ОШИБКА: Не удалось сохранить график обучения: {e}")
    plt.close()


# --- Основная функция обучения для ОТЛАДОЧНОЙ ОДНОУРОВНЕВОЙ МОДЕЛИ ---
def train_single_level_detector():
    print(f"\n--- Обучение ОТЛАДОЧНОЙ Одноуровневой Модели ({MODEL_BASE_NAME_SDL_TRAIN}) ---")
    training_run_description_sdl = "single_level_P4_debug_full_train"

    # 1. Подготовка данных (используем разделенный Detector_Dataset_Ready)
    train_split_dir_sdl = DETECTOR_DATASET_READY_ABS_SDL / "train"
    val_split_dir_sdl = DETECTOR_DATASET_READY_ABS_SDL / "validation"

    print(f"\nСбор обучающих данных из: {train_split_dir_sdl}")
    train_image_paths_sdl, train_xml_paths_sdl = collect_split_data_paths_sdl(
        str(train_split_dir_sdl), IMAGES_SUBDIR_NAME_SDL_TRAIN, ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN
    )
    print(f"\nСбор валидационных данных из: {val_split_dir_sdl}")
    val_image_paths_sdl, val_xml_paths_sdl = collect_split_data_paths_sdl(
        str(val_split_dir_sdl), IMAGES_SUBDIR_NAME_SDL_TRAIN, ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN
    )

    if not train_image_paths_sdl: print("ОШИБКА: Обучающие данные не найдены для отладочной модели."); return
    print(f"\nНайдено для обучения (отладочная модель): {len(train_image_paths_sdl)}.")
    if val_image_paths_sdl:
        print(f"Найдено для валидации (отладочная модель): {len(val_image_paths_sdl)}.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены для отладочной модели.")

    print(f"\nСоздание TensorFlow датасетов для отладочной модели...")
    print(f"  Параметры: Batch Size={BATCH_SIZE_SDL_TRAIN}, Аугментация для train={USE_AUGMENTATION_TRAIN_SDL}")
    train_dataset_sdl = create_detector_single_level_tf_dataset(
        train_image_paths_sdl, train_xml_paths_sdl, batch_size_arg=BATCH_SIZE_SDL_TRAIN,
        shuffle_arg=True, augment_arg=USE_AUGMENTATION_TRAIN_SDL  # Используем флаг из отладочного конфига
    )
    validation_dataset_sdl = None
    if val_image_paths_sdl:
        validation_dataset_sdl = create_detector_single_level_tf_dataset(
            val_image_paths_sdl, val_xml_paths_sdl, batch_size_arg=BATCH_SIZE_SDL_TRAIN,
            shuffle_arg=False, augment_arg=False  # Аугментация на валидации всегда False
        )
    if train_dataset_sdl is None: print("Не удалось создать обучающий датасет для отладочной модели. Выход."); return

    # 2. Создание и компиляция модели
    print("\nСоздание отладочной одноуровневой модели (build_detector_single_level_p4_debug)...")
    model_sdl = build_detector_single_level_p4_debug()  # Эта функция читает свой отладочный конфиг для параметров
    print("\nСтруктура отладочной модели:")
    model_sdl.summary(line_length=120)

    print(f"\nКомпиляция отладочной модели с learning_rate = {LEARNING_RATE_SDL_TRAIN}...")
    model_sdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SDL_TRAIN),
                      loss=lambda yt, yp: compute_detector_loss_single_level_debug(yt, yp, return_details=True))  # Отладочная функция потерь

    # 3. Callbacks
    timestamp_sdl = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name_sdl = f"detector_{MODEL_BASE_NAME_SDL_TRAIN}_{timestamp_sdl}"  # Используем префикс из отладочного конфига
    log_dir_sdl = LOGS_BASE_DIR_ABS_SDL / log_dir_name_sdl
    tensorboard_callback_sdl = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir_sdl), histogram_freq=1)
    callbacks_list_sdl = [tensorboard_callback_sdl, EpochTimeLoggerSDL()]
    best_model_filename_sdl = f'{MODEL_BASE_NAME_SDL_TRAIN}_best.keras'  # Фиксированное имя для лучшей
    checkpoint_filepath_best_sdl = WEIGHTS_BASE_DIR_ABS_SDL / best_model_filename_sdl
    if validation_dataset_sdl:
        model_checkpoint_sdl = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath_best_sdl), save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        callbacks_list_sdl.append(model_checkpoint_sdl)

        early_stopping_sdl = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_SDL_TRAIN, verbose=1, restore_best_weights=True)
        callbacks_list_sdl.append(early_stopping_sdl)

        reduce_lr_sdl = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=REDUCE_LR_FACTOR_SDL_TRAIN, patience=REDUCE_LR_PATIENCE_SDL_TRAIN,
            verbose=1, min_lr=MIN_LR_ON_PLATEAU_SDL_TRAIN)
        callbacks_list_sdl.append(reduce_lr_sdl)
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен для отладочной модели.")

    # 4. Запуск Обучения
    print(f"\nЗапуск обучения отладочной модели на {EPOCHS_SDL_TRAIN} эпох...")
    # ... (остальные print'ы как в основном train_detector.py)
    print(f"  Логи TensorBoard: {log_dir_sdl}")

    try:
        history_sdl = model_sdl.fit(
            train_dataset_sdl,
            epochs=EPOCHS_SDL_TRAIN,
            validation_data=validation_dataset_sdl,
            callbacks=callbacks_list_sdl,
            verbose=1
            # initial_epoch не нужен, если мы всегда начинаем с нуля для этого скрипта
        )
        print(f"\n--- Обучение отладочной модели завершено ---")

        final_model_filename_sdl = f'{MODEL_BASE_NAME_SDL_TRAIN}_final_epoch{history_sdl.epoch[-1] + 1}_{timestamp_sdl}.keras'
        final_model_save_path_sdl = WEIGHTS_BASE_DIR_ABS_SDL / final_model_filename_sdl
        model_sdl.save(final_model_save_path_sdl)
        print(f"Финальная отладочная модель сохранена в: {final_model_save_path_sdl}")
        if validation_dataset_sdl and checkpoint_filepath_best_sdl.exists():
            print(f"Лучшая отладочная модель по val_loss также сохранена/обновлена в: {checkpoint_filepath_best_sdl}")

        history_plot_filename_sdl = f'detector_history_{MODEL_BASE_NAME_SDL_TRAIN}_{timestamp_sdl}.png'
        history_plot_save_path_sdl = GRAPHS_DIR_ABS_SDL / history_plot_filename_sdl
        plot_training_history_sdl(history_sdl, str(history_plot_save_path_sdl), run_suffix=training_run_description_sdl)

    except Exception as e_fit_sdl:  # Переменная здесь e_fit_sdl
        print(f"ОШИБКА во время model.fit() для отладочной модели: {e_fit_sdl}");  # И здесь должна быть e_fit_sdl
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    train_single_level_detector()