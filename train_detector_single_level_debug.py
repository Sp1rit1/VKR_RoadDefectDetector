# train_detector_single_level_debug.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time
import matplotlib.pyplot as plt  # Оставляем для графика истории обучения в конце
from pathlib import Path
import random

# --- 1. ОПРЕДЕЛЯЕМ КОРЕНЬ ПРОЕКТА И ДОБАВЛЯЕМ SRC В sys.path ---
_project_root_sdl_train = Path(__file__).resolve().parent
_src_path_sdl_train = _project_root_sdl_train / 'src'
if str(_src_path_sdl_train) not in sys.path:
    sys.path.insert(0, str(_src_path_sdl_train))
if str(_project_root_sdl_train) not in sys.path:
    sys.path.insert(0, str(_project_root_sdl_train))

# --- 2. ИМПОРТЫ ИЗ ТВОИХ МОДУЛЕЙ В SRC ---
try:
    from datasets.detector_data_loader_single_level_debug import (
        create_detector_single_level_tf_dataset,
        DEBUG_SDL_CONFIG_GLOBAL,  # Отладочный конфиг целиком
        BASE_CONFIG_SDL_GLOBAL,  # Базовый конфиг, загруженный в отладочном загрузчике
        # Остальные специфичные переменные из отладочного загрузчика нам здесь не так важны,
        # так как параметры обучения будут браться из DEBUG_SDL_CONFIG_GLOBAL напрямую
    )
    from models.object_detector_single_level_debug import build_detector_single_level_p4_debug
    from losses.detection_losses_single_level_debug import (
        compute_detector_loss_single_level_debug,  # Основная функция с return_details
        wrapped_detector_loss_for_compile  # Обертка для model.compile()
    )

    print("INFO (train_sdl): Отладочные компоненты (data_loader, model, loss) успешно импортированы.")
except ImportError as e_imp_sdl_train:
    print(f"КРИТИЧЕСКАЯ ОШИБКА (train_sdl): Не удалось импортировать один из отладочных компонентов: {e_imp_sdl_train}")
    import traceback

    traceback.print_exc()
    exit()
except Exception as e_generic_imp:  # Ловим другие возможные ошибки при импорте
    print(f"КРИТИЧЕСКАЯ ОШИБКА (train_sdl): Непредвиденная ошибка при импорте отладочных компонентов: {e_generic_imp}")
    import traceback

    traceback.print_exc()
    exit()

# --- 3. ПАРАМЕТРЫ ИЗ ОТЛАДОЧНОГО КОНФИГА (DEBUG_SDL_CONFIG_GLOBAL) ---
# Он уже загружен и импортирован из detector_data_loader_single_level_debug.py
MODEL_BASE_NAME_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('fpn_detector_params', {}).get('model_name_prefix',
                                                                                       "RoadDefectDetector_Debug_P4_Only")
BATCH_SIZE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('batch_size', 2)
EPOCHS_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('epochs_for_debug', 50)
LEARNING_RATE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('initial_learning_rate', 0.0001)
# Флаг аугментации для этого загрузчика берется из ЕГО отладочного конфига
# и учитывает доступность функции аугментации
# Переменная USE_AUGMENTATION_SDL_LOADER_FLAG_G должна быть импортирована из detector_data_loader_single_level_debug
try:
    from datasets.detector_data_loader_single_level_debug import USE_AUGMENTATION_SDL_LOADER_FLAG_G
except ImportError:  # На случай, если она там не определена глобально
    USE_AUGMENTATION_SDL_LOADER_FLAG_G = DEBUG_SDL_CONFIG_GLOBAL.get('use_augmentation', False)
    print("ПРЕДУПРЕЖДЕНИЕ (train_sdl): Не удалось импортировать USE_AUGMENTATION_SDL_LOADER_FLAG_G, берем из конфига.")

EARLY_STOPPING_PATIENCE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('early_stopping_patience', 15)
REDUCE_LR_PATIENCE_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('reduce_lr_patience', 5)
REDUCE_LR_FACTOR_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('reduce_lr_factor', 0.2)
MIN_LR_ON_PLATEAU_SDL_TRAIN = DEBUG_SDL_CONFIG_GLOBAL.get('min_lr_on_plateau', 1e-7)

DEBUG_CALLBACK_LOG_FREQ_SDL = DEBUG_SDL_CONFIG_GLOBAL.get('debug_callback_log_freq', 1)

# Пути к данным и для сохранения (используем BASE_CONFIG_SDL_GLOBAL, импортированный из отладочного data_loader)
_detector_dataset_ready_path_rel_sdl = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_SDL = _project_root_sdl_train / _detector_dataset_ready_path_rel_sdl
IMAGES_SUBDIR_NAME_SDL_TRAIN = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN = BASE_CONFIG_SDL_GLOBAL.get('dataset', {}).get('annotations_dir', 'Annotations')

LOGS_BASE_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('logs_base_dir', 'logs')
WEIGHTS_BASE_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('weights_base_dir', 'weights')
GRAPHS_DIR_ABS_SDL = _project_root_sdl_train / BASE_CONFIG_SDL_GLOBAL.get('graphs_dir', 'graphs')
for p in [LOGS_BASE_DIR_ABS_SDL, WEIGHTS_BASE_DIR_ABS_SDL, GRAPHS_DIR_ABS_SDL]:
    p.mkdir(parents=True, exist_ok=True)


# --- 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def collect_split_data_paths_sdl(split_dir_abs_path_str, images_subdir_name, annotations_subdir_name):
    image_paths = []
    xml_paths = []
    current_images_dir = os.path.join(split_dir_abs_path_str, images_subdir_name)
    current_annotations_dir = os.path.join(split_dir_abs_path_str, annotations_subdir_name)
    if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ (sdl_train): Директория {current_images_dir} или {current_annotations_dir} не найдена для split'а.")
        return image_paths, xml_paths
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files_in_split = []
    for ext in valid_extensions:
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.lower()}")))
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.upper()}")))
    image_files_in_split = sorted(list(set(image_files_in_split)))
    for img_path_str in image_files_in_split:
        base_name, _ = os.path.splitext(os.path.basename(img_path_str))
        xml_path_str = os.path.join(current_annotations_dir, base_name + ".xml")
        if os.path.exists(xml_path_str):
            image_paths.append(img_path_str)
            xml_paths.append(xml_path_str)
    return image_paths, xml_paths


class EpochTimeLoggerSDL(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch_duration_sec'] = time.time() - self.epoch_start_time
        print(f" - Время эпохи: {logs['epoch_duration_sec']:.2f} сек")  # Keras verbose=1 тоже выводит время


def plot_training_history_sdl(history, save_path_plot_str, run_suffix=""):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss'] is not None:  # Проверка на None
        plt.plot(history.history['val_loss'], label='Validation Loss')
    title = f'История обучения отладочного детектора ({run_suffix})' if run_suffix else 'История обучения отладочного детектора'
    plt.title(title);
    plt.ylabel('Loss');
    plt.xlabel('Epoch')
    plt.legend(loc='upper right');
    plt.grid(True);
    plt.tight_layout()
    try:
        plt.savefig(save_path_plot_str);
        print(f"График истории обучения сохранен в: {save_path_plot_str}")
    except Exception as e:
        print(f"ОШИБКА: Не удалось сохранить график обучения ({save_path_plot_str}): {e}")
    plt.close()


# --- 5. КАСТОМНЫЙ КОЛЛБЭК ДЛЯ ДЕТАЛЬНОГО ЛОГИРОВАНИЯ (БЕЗ ВИЗУАЛИЗАЦИИ КАРТИНОК) ---
class SingleLevelDetailedLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_sample_dataset, log_freq=1, loss_function_for_details=None):
        super().__init__()
        self.validation_sample_dataset = validation_sample_dataset  # Ожидаем батчированный датасет
        self.log_freq = max(1, int(log_freq))
        self.loss_fn_details = loss_function_for_details

        if self.validation_sample_dataset is None:
            print(
                "ПРЕДУПРЕЖДЕНИЕ (LossLogger): Валидационные данные не предоставлены, детальный лог потерь будет пропущен.")
        elif not self.loss_fn_details:
            print("ПРЕДУПРЕЖДЕНИЕ (LossLogger): Функция потерь для деталей не предоставлена.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}  # Keras передает сюда словарь с метриками (loss, val_loss и т.д.)
        if (epoch + 1) % self.log_freq != 0 or self.validation_sample_dataset is None or not self.loss_fn_details:
            return

        print(f"\n--- Детальный Лог Потерь в Конце Эпохи {epoch + 1} (Отладочная Модель) ---")

        all_loss_components_accumulated = {}
        num_batches_for_log = 0

        # Берем несколько батчей из валидационного датасета для усреднения потерь
        # Убедимся, что validation_sample_dataset можно итерировать несколько раз или .repeat()
        # Для простоты, возьмем .take(N), где N - небольшое число
        for images_val_batch, y_true_val_batch in self.validation_sample_dataset.take(5):  # Например, 5 батчей
            y_pred_val_batch = self.model.predict_on_batch(images_val_batch)  # model доступен как self.model

            # Вызываем функцию потерь с return_details=True
            loss_details_dict_batch = self.loss_fn_details(y_true_val_batch, y_pred_val_batch, return_details=True)

            if isinstance(loss_details_dict_batch, dict):
                for k, v_tensor in loss_details_dict_batch.items():
                    v_numpy_mean_for_batch = tf.reduce_mean(v_tensor).numpy()  # Усредняем по текущему батчу
                    if k not in all_loss_components_accumulated:
                        all_loss_components_accumulated[k] = []
                    all_loss_components_accumulated[k].append(v_numpy_mean_for_batch)
                num_batches_for_log += 1
            else:  # Если вернулся только скаляр (не должно быть, если return_details=True)
                if 'total_loss_from_fn' not in all_loss_components_accumulated:
                    all_loss_components_accumulated['total_loss_from_fn'] = []
                all_loss_components_accumulated['total_loss_from_fn'].append(loss_details_dict_batch.numpy())
                num_batches_for_log += 1

        if num_batches_for_log > 0:
            print("  Детализированные Потери на Валидационной Выборке (усредненные по нескольким батчам):")
            for k, v_list_component in all_loss_components_accumulated.items():
                avg_v_component = np.mean(v_list_component)
                logs[f'val_detailed_{k}'] = avg_v_component  # Добавляем в логи Keras для TensorBoard
                print(f"    val_detailed_{k}: {avg_v_component:.4f}")
        else:
            print("  Не удалось обработать батчи для расчета детализированных потерь.")
        print(f"--- Конец Детального Лога Потерь Эпохи {epoch + 1} ---")


# --- 6. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ---
def train_single_level_detector():
    print(f"\n--- Обучение ОТЛАДОЧНОЙ Одноуровневой Модели ({MODEL_BASE_NAME_SDL_TRAIN}) ---")
    training_run_description_sdl = "single_level_P4_debug_full_train_loss_logger"  # Обновим имя запуска

    # 1. Подготовка данных
    train_split_dir_sdl_path = DETECTOR_DATASET_READY_ABS_SDL / "train"
    val_split_dir_sdl_path = DETECTOR_DATASET_READY_ABS_SDL / "validation"
    train_image_paths_sdl, train_xml_paths_sdl = collect_split_data_paths_sdl(
        str(train_split_dir_sdl_path), IMAGES_SUBDIR_NAME_SDL_TRAIN, ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN)
    val_image_paths_sdl, val_xml_paths_sdl = collect_split_data_paths_sdl(
        str(val_split_dir_sdl_path), IMAGES_SUBDIR_NAME_SDL_TRAIN, ANNOTATIONS_SUBDIR_NAME_SDL_TRAIN)

    if not train_image_paths_sdl: print("ОШИБКА: Обучающие данные не найдены."); return
    print(f"\nНайдено для обучения: {len(train_image_paths_sdl)}.")
    if val_image_paths_sdl:
        print(f"Найдено для валидации: {len(val_image_paths_sdl)}.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены.")

    print(f"\nСоздание TensorFlow датасетов...")
    print(f"  Параметры: Batch Size={BATCH_SIZE_SDL_TRAIN}, Аугментация для train={USE_AUGMENTATION_SDL_LOADER_FLAG_G}")
    train_dataset_sdl = create_detector_single_level_tf_dataset(
        train_image_paths_sdl, train_xml_paths_sdl, batch_size_arg=BATCH_SIZE_SDL_TRAIN,
        shuffle_arg=True, augment_arg=USE_AUGMENTATION_SDL_LOADER_FLAG_G
    )
    validation_dataset_sdl = None
    if val_image_paths_sdl:
        validation_dataset_sdl = create_detector_single_level_tf_dataset(
            val_image_paths_sdl, val_xml_paths_sdl, batch_size_arg=BATCH_SIZE_SDL_TRAIN,
            shuffle_arg=False, augment_arg=False
        )
    if train_dataset_sdl is None: print("Не удалось создать обучающий датасет. Выход."); return

    # Проверка, что датасеты не пусты
    try:
        for _ in train_dataset_sdl.take(1): pass
        print("Обучающий датасет для отладочной модели успешно создан и содержит данные.")
        if validation_dataset_sdl:
            for _ in validation_dataset_sdl.take(1): pass
            print("Валидационный датасет для отладочной модели успешно создан и содержит данные.")
    except tf.errors.OutOfRangeError:
        print("ОШИБКА: Один из датасетов (train или val) оказался пустым после попытки взять первый батч.");
        return
    except Exception as e_ds_check_sdl:
        print(f"ОШИБКА: Датасет для отладочной модели пуст или произошла ошибка при доступе: {e_ds_check_sdl}");
        return

    # 2. Создание и компиляция модели
    print("\nСоздание отладочной одноуровневой модели (build_detector_single_level_p4_debug)...")
    model_sdl = build_detector_single_level_p4_debug()
    # print("\nСтруктура отладочной модели:") # Можно раскомментировать для вывода summary
    # model_sdl.summary(line_length=120)

    print(f"\nКомпиляция отладочной модели с learning_rate = {LEARNING_RATE_SDL_TRAIN}...")
    model_sdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SDL_TRAIN),
                      loss=wrapped_detector_loss_for_compile)  # Используем обертку для компиляции

    # 3. Callbacks
    timestamp_sdl = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name_sdl = f"detector_{MODEL_BASE_NAME_SDL_TRAIN}_{training_run_description_sdl}_{timestamp_sdl}"
    log_dir_sdl = LOGS_BASE_DIR_ABS_SDL / log_dir_name_sdl
    tensorboard_callback_sdl = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir_sdl), histogram_freq=1)

    callbacks_list_sdl = [tensorboard_callback_sdl, EpochTimeLoggerSDL()]

    if validation_dataset_sdl:
        # Добавляем наш новый логгер детализированных потерь
        detailed_loss_logger_sdl = SingleLevelDetailedLossLogger(
            validation_sample_dataset=validation_dataset_sdl,  # Передаем батчированный val_ds
            log_freq=DEBUG_CALLBACK_LOG_FREQ_SDL,
            loss_function_for_details=compute_detector_loss_single_level_debug  # Основная функция
        )
        callbacks_list_sdl.append(detailed_loss_logger_sdl)

        best_model_filename_sdl = f'{MODEL_BASE_NAME_SDL_TRAIN}_{training_run_description_sdl}_best.keras'
        checkpoint_filepath_best_sdl = WEIGHTS_BASE_DIR_ABS_SDL / best_model_filename_sdl
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
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен. Детальное логирование потерь и сохранение лучшей модели по val_loss будут пропущены.")

    # 4. Запуск Обучения
    print(f"\nЗапуск обучения отладочной модели на {EPOCHS_SDL_TRAIN} эпох...")
    print(f"  Логи TensorBoard: {log_dir_sdl}")
    try:
        history_sdl = model_sdl.fit(
            train_dataset_sdl,
            epochs=EPOCHS_SDL_TRAIN,
            validation_data=validation_dataset_sdl,
            callbacks=callbacks_list_sdl,
            verbose=1  # Включим стандартный вывод Keras для эпох
        )
        print(f"\n--- Обучение отладочной модели завершено ---")

        final_model_filename_sdl = f'{MODEL_BASE_NAME_SDL_TRAIN}_{training_run_description_sdl}_final_epoch{history_sdl.epoch[-1] + 1}_{timestamp_sdl}.keras'
        final_model_save_path_sdl = WEIGHTS_BASE_DIR_ABS_SDL / final_model_filename_sdl
        model_sdl.save(final_model_save_path_sdl)
        print(f"Финальная отладочная модель сохранена в: {final_model_save_path_sdl}")

        if validation_dataset_sdl and checkpoint_filepath_best_sdl.exists():
            print(f"Лучшая отладочная модель по val_loss также сохранена/обновлена в: {checkpoint_filepath_best_sdl}")

        history_plot_filename_sdl = f'detector_history_{MODEL_BASE_NAME_SDL_TRAIN}_{training_run_description_sdl}_{timestamp_sdl}.png'
        history_plot_save_path_sdl = GRAPHS_DIR_ABS_SDL / history_plot_filename_sdl
        plot_training_history_sdl(history_sdl, str(history_plot_save_path_sdl), run_suffix=training_run_description_sdl)

    except Exception as e_fit_sdl:
        print(f"ОШИБКА во время model.fit() для отладочной модели: {e_fit_sdl}");
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    train_single_level_detector()