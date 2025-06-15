# RoadDefectDetector/train_detector.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time
import matplotlib.pyplot as plt  # Для сохранения графика
from pathlib import Path  # Для удобной работы с путями

# --- Определяем корень проекта и добавляем src в sys.path для корректных импортов ---
_project_root = Path(__file__).resolve().parent  # Если train_detector.py в корне
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# --- Импорты из твоих модулей в src ---
try:
    from datasets.detector_data_loader import create_detector_tf_dataset
    from models.object_detector import build_object_detector_v1
    from losses.detection_losses import compute_detector_loss_v1

    # from utils.callbacks import EpochTimeLogger # Если будешь использовать из отдельного файла
    print("INFO: Необходимые модули из src успешно импортированы.")
except ImportError as e:
    print(f"ОШИБКА ИМПОРТА (train_detector.py): Не удалось импортировать один из модулей из src: {e}")
    exit(1)
except Exception as e_gen_imp:
    print(f"ОШИБКА ИМПОРТА (train_detector.py): Общая ошибка при импорте: {e_gen_imp}")
    exit(1)

# --- Загрузка Конфигураций ---
_base_config_path_obj = _src_path / 'configs' / 'base_config.yaml'
_detector_config_path_obj = _src_path / 'configs' / 'detector_config.yaml'

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS_TRAIN_DET = True


def load_config_train_det(config_path_obj, config_name_str):
    global CONFIG_LOAD_SUCCESS_TRAIN_DET
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg_content = yaml.safe_load(f)
        if not isinstance(cfg_content, dict) or not cfg_content:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Конфиг '{config_name_str}' ({config_path_obj}) пуст или имеет неверный формат.")
            CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
            return {}
        print(f"INFO: Конфиг '{config_name_str}' успешно загружен.")
        return cfg_content
    except FileNotFoundError:
        print(f"ОШИБКА: Файл конфига '{config_name_str}' не найден: {config_path_obj}")
        CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
        return {}
    except yaml.YAMLError as e_yaml:
        print(f"ОШИБКА YAML при чтении '{config_name_str}' ({config_path_obj}): {e_yaml}")
        CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
        return {}
    except Exception as e_load_cfg:
        print(f"НЕПРЕДВИДЕННАЯ ОШИБКА при загрузке '{config_name_str}' ({config_path_obj}): {e_load_cfg}")
        CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
        return {}


print("\n--- Загрузка конфигурационных файлов для train_detector.py ---")
BASE_CONFIG = load_config_train_det(_base_config_path_obj, "Base Config")
DETECTOR_CONFIG = load_config_train_det(_detector_config_path_obj, "Detector Config")

if not CONFIG_LOAD_SUCCESS_TRAIN_DET or not BASE_CONFIG or not DETECTOR_CONFIG:  # Добавил проверку на пустые конфиги
    print(
        "\nОШИБКА: Не удалось загрузить один или несколько критичных файлов конфигурации. Обучение детектора прервано.")
    exit(1)

# --- Параметры из Конфигов ---
_detector_dataset_ready_path_rel = DETECTOR_CONFIG.get("prepared_data_path", "data/Detector_Dataset_Ready")
DETECTOR_DATASET_READY_ABS = (_project_root / _detector_dataset_ready_path_rel).resolve()

IMAGES_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')

TRAIN_PARAMS_DET = DETECTOR_CONFIG.get('train_params', {})
BATCH_SIZE_DET = TRAIN_PARAMS_DET.get('batch_size', 4)
EPOCHS_DET = TRAIN_PARAMS_DET.get('epochs', 100)
LEARNING_RATE_DET = TRAIN_PARAMS_DET.get('learning_rate', 0.0001)
USE_AUGMENTATION_TRAIN = DETECTOR_CONFIG.get('use_augmentation', True)

EARLY_STOPPING_PATIENCE_DET = TRAIN_PARAMS_DET.get('early_stopping_patience', 20)
LR_FACTOR_DET = TRAIN_PARAMS_DET.get('lr_factor', 0.2)
LR_PATIENCE_DET = TRAIN_PARAMS_DET.get('lr_patience', 7)
MIN_LR_DET = TRAIN_PARAMS_DET.get('min_lr', 1e-6)

LOGS_BASE_DIR_ABS = (_project_root / BASE_CONFIG.get('logs_base_dir', 'logs')).resolve()
WEIGHTS_BASE_DIR_ABS = (_project_root / BASE_CONFIG.get('weights_base_dir', 'weights')).resolve()


class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Keras verbose=1 уже выводит время, этот коллбэк может быть не нужен
        # или использоваться при verbose=0 или verbose=2 для кастомного лога
        pass
        # epoch_duration = time.time() - self.epoch_start_time
        # print(f"Epoch {epoch+1} duration: {epoch_duration:.2f}s")


def collect_split_data_paths(split_dir_path_obj, images_subdir_str, annotations_subdir_str):
    image_paths = [];
    xml_paths = []
    current_images_dir = split_dir_path_obj / images_subdir_str
    current_annotations_dir = split_dir_path_obj / annotations_subdir_str

    if not current_images_dir.is_dir() or not current_annotations_dir.is_dir():
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ: Директория изображений ({current_images_dir}) или аннотаций ({current_annotations_dir}) не найдена для split '{split_dir_path_obj.name}'.")
        return image_paths, xml_paths

    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files_in_split = []
    for ext in valid_extensions:
        image_files_in_split.extend(list(current_images_dir.glob(f"*{ext.lower()}")))
        image_files_in_split.extend(list(current_images_dir.glob(f"*{ext.upper()}")))
    image_files_in_split = sorted(list(set(image_files_in_split)))

    for img_path_obj_loop in image_files_in_split:
        base_name = img_path_obj_loop.stem
        xml_path_obj_loop = current_annotations_dir / (base_name + ".xml")
        if xml_path_obj_loop.exists():
            image_paths.append(str(img_path_obj_loop))
            xml_paths.append(str(xml_path_obj_loop))
        else:
            print(
                f"    ПРЕДУПРЕЖДЕНИЕ (collect_split): XML для {img_path_obj_loop.name} не найден. Изображение пропущено.")
    return image_paths, xml_paths


def train_detector_main():
    if not CONFIG_LOAD_SUCCESS_TRAIN_DET or not BASE_CONFIG or not DETECTOR_CONFIG:
        print("Критическая ошибка: Не удалось загрузить файлы конфигурации. Обучение детектора прервано.")
        return

    print("\n--- Обучение Детектора Объектов (Кастомная Модель v1 с Train/Val) ---")

    # --- Смешанная точность (опционально) ---
    # try:
    #     from tensorflow.keras import mixed_precision
    #     policy = mixed_precision.Policy('mixed_float16')
    #     mixed_precision.set_global_policy(policy)
    #     print("INFO: Включена смешанная точность (mixed_float16).")
    # except Exception as e_mp:
    #     print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при установке смешанной точности: {e_mp}")

    # 1. Сбор путей к данным
    train_split_dir_obj = DETECTOR_DATASET_READY_ABS / "train"
    val_split_dir_obj = DETECTOR_DATASET_READY_ABS / "validation"

    print(f"\nСбор обучающих данных из: {train_split_dir_obj}")
    train_image_paths, train_xml_paths = collect_split_data_paths(train_split_dir_obj, IMAGES_SUBDIR_NAME_DET,
                                                                  ANNOTATIONS_SUBDIR_NAME_DET)
    print(f"\nСбор валидационных данных из: {val_split_dir_obj}")
    val_image_paths, val_xml_paths = collect_split_data_paths(val_split_dir_obj, IMAGES_SUBDIR_NAME_DET,
                                                              ANNOTATIONS_SUBDIR_NAME_DET)

    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(f"\nНайдено для обучения: {len(train_image_paths)} изображений.")
    if val_image_paths:
        print(f"Найдено для валидации: {len(val_image_paths)} изображений.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены. Обучение без валидации на лету.")

    # 2. Создание датасетов
    print(f"\nСоздание TensorFlow датасетов...")
    print(f"  Параметры: Batch Size={BATCH_SIZE_DET}, Аугментация (train)={USE_AUGMENTATION_TRAIN}")
    train_dataset_detector = create_detector_tf_dataset(train_image_paths, train_xml_paths, batch_size=BATCH_SIZE_DET,
                                                        shuffle=True, augment=USE_AUGMENTATION_TRAIN)
    validation_dataset_detector = None
    if val_image_paths:
        validation_dataset_detector = create_detector_tf_dataset(val_image_paths, val_xml_paths,
                                                                 batch_size=BATCH_SIZE_DET, shuffle=False,
                                                                 augment=False)

    if train_dataset_detector is None: print("Не удалось создать обучающий датасет."); return
    try:
        for _ in train_dataset_detector.take(1): pass; print("Обучающий датасет успешно создан.")
        if validation_dataset_detector:
            for _ in validation_dataset_detector.take(1): pass; print("Валидационный датасет успешно создан.")
    except Exception as e_ds_check:
        print(f"ОШИБКА: Датасет пуст или ошибка доступа: {e_ds_check}"); return

    # 3. Создание и компиляция модели
    print("\nСоздание модели детектора (build_object_detector_v1)...")
    model = build_object_detector_v1()
    print("\nСтруктура модели детектора:")
    model.summary(line_length=120)
    print(f"\nКомпиляция модели с learning_rate = {LEARNING_RATE_DET}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DET), loss=compute_detector_loss_v1)

    # 4. Callbacks
    current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_path_obj = LOGS_BASE_DIR_ABS / "detector_fit_v1_full" / current_time_str
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir_path_obj), histogram_freq=1)
    WEIGHTS_BASE_DIR_ABS.mkdir(parents=True, exist_ok=True)

    epoch_time_logger_callback = EpochTimeLogger()
    callbacks_list = [tensorboard_callback]  # Убрал EpochTimeLogger, т.к. verbose=1 уже выводит время
    # Если хочешь его, раскомментируй и добавь в список,
    # и в model.fit() поставь verbose=2 или verbose=0

    checkpoint_filepath_best_obj = None
    if validation_dataset_detector:
        checkpoint_filename_best = f'detector_v1_best_val_loss_{current_time_str}.keras'
        checkpoint_filepath_best_obj = WEIGHTS_BASE_DIR_ABS / checkpoint_filename_best
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath_best_obj), save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        callbacks_list.append(model_checkpoint_callback)
        print(f"Лучшая модель будет сохраняться в: {checkpoint_filepath_best_obj} (по val_loss)")
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_DET, verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stopping_callback)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=LR_FACTOR_DET, patience=LR_PATIENCE_DET, verbose=1, min_lr=MIN_LR_DET)
        callbacks_list.append(reduce_lr_callback)
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен.")

    # 5. Запуск обучения
    print(f"\nЗапуск обучения детектора на {EPOCHS_DET} эпох...")
    print(f"  Обучающая выборка: {len(train_image_paths)} изображений, Аугментация: {USE_AUGMENTATION_TRAIN}")
    if val_image_paths: print(f"  Валидационная выборка: {len(val_image_paths)} изображений")
    print(f"  Batch Size: {BATCH_SIZE_DET}")
    print(f"  Логи TensorBoard: {log_dir_path_obj}")

    training_start_time = time.time()
    history = None  # Инициализируем history
    try:
        history = model.fit(
            train_dataset_detector,
            epochs=EPOCHS_DET,
            validation_data=validation_dataset_detector,
            callbacks=callbacks_list,
            verbose=1
        )
    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit() для детектора: {e_fit}")
        import traceback
        traceback.print_exc()
    finally:  # Блок finally выполнится даже если была ошибка в try
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        time_message_prefix = "Общее время обучения детектора" if history else "Обучение прервано. Затраченное время"
        print(
            f"\n--- {time_message_prefix}: {total_training_time // 3600:.0f} ч {(total_training_time % 3600) // 60:.0f} мин {total_training_time % 60:.2f} сек ---")

        if history:  # Если обучение прошло хотя бы одну эпоху
            print("\n--- Обучение детектора (v1) завершено (или остановлено EarlyStopping) ---")

            # Сохранение графика обучения
            if history.history:  # Убедимся, что история существует и не пуста
                print("\nСохранение графика обучения...")
                graphs_dir_obj = _project_root / "graphs"
                graphs_dir_obj.mkdir(parents=True, exist_ok=True)
                plot_filename = f"detector_training_history_{current_time_str}.png"
                plot_save_path_obj = graphs_dir_obj / plot_filename

                plt.figure(figsize=(12, 6))
                if 'loss' in history.history:
                    plt.plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='Validation Loss')

                plt.title('История Обучения Детектора (Loss)')
                plt.ylabel('Loss');
                plt.xlabel('Эпоха')
                if 'loss' in history.history or ('val_loss' in history.history and validation_dataset_detector):
                    plt.legend(loc='upper right')
                plt.grid(True, linestyle='--', alpha=0.7)
                try:
                    plt.savefig(str(plot_save_path_obj))
                    print(f"График истории обучения сохранен в: {plot_save_path_obj}")
                except Exception as e_plot:
                    print(f"ОШИБКА при сохранении графика: {e_plot}")
                plt.close()
            else:
                print("ПРЕДУПРЕЖДЕНИЕ: history.history пуст, график обучения не будет сохранен.")

            # Сохранение финальной модели (может быть той же, что и лучшая, если EarlyStopping сработал с restore_best_weights)
            final_model_filename = f'detector_v1_final_full_{current_time_str}.keras'
            final_model_save_path_obj = WEIGHTS_BASE_DIR_ABS / final_model_filename
            try:
                model.save(str(final_model_save_path_obj))
                print(f"Финальная модель сохранена в: {final_model_save_path_obj}")
            except Exception as e_save_model:
                print(f"ОШИБКА при сохранении финальной модели: {e_save_model}")

            if validation_dataset_detector and checkpoint_filepath_best_obj and checkpoint_filepath_best_obj.exists():
                print(f"Лучшая модель по val_loss также сохранена в: {checkpoint_filepath_best_obj}")
            elif validation_dataset_detector and checkpoint_filepath_best_obj:  # Если путь был сформирован, но файла нет
                print(
                    f"ПРЕДУПРЕЖДЕНИЕ: Ожидался файл лучшей модели {checkpoint_filepath_best_obj}, но он не найден (возможно, val_loss не улучшался).")
        else:  # Если history is None (например, model.fit() упал до первой эпохи)
            print("\nОбучение не было успешно запущено, финальная модель не сохранена.")


if __name__ == '__main__':
    if not CONFIG_LOAD_SUCCESS_TRAIN_DET or not BASE_CONFIG or not DETECTOR_CONFIG:
        print("\nОбучение не может быть запущено из-за критических ошибок загрузки конфигурации.")
    else:
        train_detector_main()