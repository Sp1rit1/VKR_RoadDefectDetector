# train_detector.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time  # Для замера времени эпохи
import matplotlib.pyplot as plt  # Для сохранения графика

# --- Определяем корень проекта и добавляем src в sys.path для корректных импортов ---
_project_root = os.path.dirname(os.path.abspath(__file__))  # Это корень проекта, где лежит train_detector.py
_src_path = os.path.join(_project_root, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# --- Импорты из твоих модулей в src ---
from datasets.detector_data_loader import create_detector_tf_dataset
from models.object_detector import \
    build_object_detector_v1_enhanced  # Предполагается, что эта функция теперь всегда создает модель с ЗАМОРОЖЕННЫМ backbone по умолчанию
from losses.detection_losses import compute_detector_loss_v1  # Убедись, что имя функции правильное

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
    print(f"ОШИБКА (train_detector.py): base_config.yaml не найден: {_base_config_path}")
except yaml.YAMLError as e:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): YAML в base_config.yaml: {e}")

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN_DET = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): detector_config.yaml не найден: {_detector_config_path}")
except yaml.YAMLError as e:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False;
    print(f"ОШИБКА (train_detector.py): YAML в detector_config.yaml: {e}")

if not CONFIG_LOAD_SUCCESS_TRAIN_DET:
    print("ОШИБКА: Не удалось загрузить один или несколько файлов конфигурации. Выход.")
    exit()

# --- Параметры из Конфигов (с дефолтами для безопасности) ---
# Для датасета
IMAGES_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel = "data/Detector_Dataset_Ready"  # Ожидаемый путь к папке с train/val
DETECTOR_DATASET_READY_ABS = os.path.join(_project_root, _detector_dataset_ready_path_rel)

# Для модели
MODEL_BASE_NAME_CFG = DETECTOR_CONFIG.get('model_base_name', 'RoadDefectDetector_DefaultName')
BACKBONE_LAYER_NAME_IN_MODEL_CFG = DETECTOR_CONFIG.get('backbone_layer_name_in_model', 'Backbone_MobileNetV2')
# Параметры для build_object_detector_v1 (он сам их возьмет из своего экземпляра DETECTOR_CONFIG)

# Режим обучения и параметры
CONTINUE_FROM_CHECKPOINT_CFG = DETECTOR_CONFIG.get('continue_from_checkpoint', False)
PATH_TO_CHECKPOINT_REL_CFG = DETECTOR_CONFIG.get('path_to_checkpoint', None)
UNFREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('unfreeze_backbone', False)
UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG = DETECTOR_CONFIG.get('unfreeze_backbone_layers_from_end', 0)

BATCH_SIZE_CFG = DETECTOR_CONFIG.get('batch_size', 2)  # Общий batch_size, используется при создании датасета
EPOCHS_CFG = DETECTOR_CONFIG.get('epochs', 50)
INITIAL_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('initial_learning_rate', 0.0001)
FINETUNE_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('finetune_learning_rate', 1e-5)

EARLY_STOPPING_PATIENCE_CFG = DETECTOR_CONFIG.get('early_stopping_patience', 20)
REDUCE_LR_PATIENCE_CFG = DETECTOR_CONFIG.get('reduce_lr_patience', 7)
REDUCE_LR_FACTOR_CFG = DETECTOR_CONFIG.get('reduce_lr_factor', 0.2)
MIN_LR_ON_PLATEAU_CFG = DETECTOR_CONFIG.get('min_lr_on_plateau', 1e-6)
USE_AUGMENTATION_CFG = DETECTOR_CONFIG.get('use_augmentation', True)

# Параметры для логов и весов
LOGS_BASE_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('logs_base_dir', 'logs'))
WEIGHTS_BASE_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))
GRAPHS_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('graphs_dir', 'graphs'))  # Для сохранения графиков
os.makedirs(GRAPHS_DIR_ABS, exist_ok=True)
os.makedirs(WEIGHTS_BASE_DIR_ABS, exist_ok=True)


# --- Вспомогательные Функции ---
def collect_split_data_paths(split_dir_abs_path, images_subdir, annotations_subdir):
    image_paths = []
    xml_paths = []
    current_images_dir = os.path.join(split_dir_abs_path, images_subdir)
    current_annotations_dir = os.path.join(split_dir_abs_path, annotations_subdir)
    if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
        print(
            f"  ПРЕДУПРЕЖДЕНИЕ: Директория {current_images_dir} или {current_annotations_dir} не найдена для split'а. Пропускаем.")
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
        # else:
        # print(f"    ПРЕДУПРЕЖДЕНИЕ (collect_split): XML для {os.path.basename(img_path)} не найден в {current_annotations_dir}")
    return image_paths, xml_paths


class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        # Вывод номера эпохи теперь делается Keras при verbose=1 или 2
        # print(f"Epoch {epoch+1}/{self.params['epochs']} - Начало", end="")

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        # Keras выводит время на шаг, мы добавим общее время эпохи в лог
        # TensorFlow уже логирует время эпохи сам при verbose=1 или 2
        # logs['epoch_duration_sec'] = epoch_duration # Можно добавить в логи для TensorBoard
        print(f" - Время эпохи: {epoch_duration:.2f} сек")


def plot_training_history(history, save_path_plot, run_suffix=""):
    """Сохраняет график истории обучения."""
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss']:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    title = f'История обучения детектора ({run_suffix})' if run_suffix else 'История обучения детектора'
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(save_path_plot)
        print(f"График истории обучения сохранен в: {save_path_plot}")
    except Exception as e:
        print(f"ОШИБКА: Не удалось сохранить график обучения: {e}")
    plt.close()


# --- Основная функция обучения ---
def train_detector_main():
    print(f"\n--- Запуск Обучения Детектора Объектов ---")

    # 1. Определяем режим и параметры на основе конфига
    training_run_description = "initial_frozen_bb"  # Суффикс для имен файлов и логов
    current_lr = INITIAL_LEARNING_RATE_CFG
    current_epochs = EPOCHS_CFG
    current_use_augmentation = USE_AUGMENTATION_CFG
    initial_epoch_for_fit = 0  # Для продолжения счета эпох в model.fit()
    model_to_load_path = None

    if PATH_TO_CHECKPOINT_REL_CFG:
        model_to_load_path = os.path.join(WEIGHTS_BASE_DIR_ABS, PATH_TO_CHECKPOINT_REL_CFG)

    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path and os.path.exists(model_to_load_path):
        print(f"Режим: Продолжение обучения / Fine-tuning с чекпоинта: {model_to_load_path}")
        if UNFREEZE_BACKBONE_CFG:
            training_run_description = "finetune_bb"
            current_lr = FINETUNE_LEARNING_RATE_CFG
            # current_use_augmentation = DETECTOR_CONFIG.get('use_augmentation_on_finetune', False) # Если есть отдельный флаг
            print(f"  Backbone будет разморожен. Learning rate: {current_lr}")
        else:
            training_run_description = "continued_frozen_bb"
            print(f"  Backbone останется/будет заморожен. Learning rate: {current_lr}")
        # initial_epoch_for_fit можно попытаться извлечь из имени файла или логов, если нужно
    elif UNFREEZE_BACKBONE_CFG:  # Обучение с нуля, но backbone сразу разморожен
        training_run_description = "initial_unfrozen_bb"
        current_lr = FINETUNE_LEARNING_RATE_CFG  # Используем LR для fine-tuning'а, так как это более деликатный процесс
        print(f"Режим: Начальное обучение с РАЗМОРОЖЕННЫМ backbone. Learning rate: {current_lr}")
    else:  # initial_train с замороженным backbone (стандартный)
        print(f"Режим: Начальное обучение с ЗАМОРОЖЕННЫМ backbone. Learning rate: {current_lr}")
        # path_to_model_for_continuation здесь не используется, модель создается с нуля

    print(f"Описание текущего запуска: {training_run_description}")

    # 2. Подготовка данных
    train_split_dir = os.path.join(DETECTOR_DATASET_READY_ABS, "train")
    val_split_dir = os.path.join(DETECTOR_DATASET_READY_ABS, "validation")
    train_image_paths, train_xml_paths = collect_split_data_paths(train_split_dir, IMAGES_SUBDIR_NAME_DET,
                                                                  ANNOTATIONS_SUBDIR_NAME_DET)
    val_image_paths, val_xml_paths = collect_split_data_paths(val_split_dir, IMAGES_SUBDIR_NAME_DET,
                                                              ANNOTATIONS_SUBDIR_NAME_DET)

    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(f"\nНайдено для обучения: {len(train_image_paths)}.")
    if val_image_paths:
        print(f"Найдено для валидации: {len(val_image_paths)}.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены.")

    print(f"\nСоздание TensorFlow датасетов...")
    print(f"  Параметры: Batch Size={BATCH_SIZE_CFG}, Аугментация для train={current_use_augmentation}")
    train_dataset = create_detector_tf_dataset(
        train_image_paths, train_xml_paths, batch_size=BATCH_SIZE_CFG,
        shuffle=True, augment=current_use_augmentation
    )
    validation_dataset = None
    if val_image_paths:
        validation_dataset = create_detector_tf_dataset(
            val_image_paths, val_xml_paths, batch_size=BATCH_SIZE_CFG,
            shuffle=False, augment=False
        )
    if train_dataset is None: print("Не удалось создать обучающий датасет. Выход."); return
    # ... (проверки на пустоту датасетов) ...

    # 3. Создание или Загрузка Модели
    model = None
    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path and os.path.exists(model_to_load_path):
        print(f"\nЗагрузка существующей модели из: {model_to_load_path}")
        try:
            model = tf.keras.models.load_model(
                model_to_load_path,
                custom_objects={'compute_detector_loss_v1': compute_detector_loss_v1},
                compile=False  # Загружаем без компиляции
            )
            print("Модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}. Будет создана новая модель.")
            model = None

    if model is None:
        if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Файл '{model_to_load_path}' не найден или ошибка загрузки.")
        print("\nСоздание новой модели детектора (build_object_detector_v1)...")
        # build_object_detector_v1 должен сам использовать freeze_backbone_initial_train из конфига,
        # или мы предполагаем, что он всегда создает с замороженным.
        model = build_object_detector_v1_enhanced()
        # Если это не "continue_train", то начальная заморозка backbone уже произошла в build_object_detector_v1
        # (предполагая, что freeze_backbone_initial_train в detector_config управляет этим)

    # Применяем UNFREEZE_BACKBONE_CFG, если это указано
    if UNFREEZE_BACKBONE_CFG:
        print(f"\nПрименение флага unfreeze_backbone: True")
        backbone_layer = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)
        if backbone_layer:
            backbone_layer.trainable = True
            num_layers_to_finetune = UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG
            if num_layers_to_finetune > 0 and hasattr(backbone_layer, 'layers') and \
                    len(backbone_layer.layers) > num_layers_to_finetune:
                print(f"  Размораживаются только последние {num_layers_to_finetune} слоев backbone...")
                for layer_idx, layer in enumerate(backbone_layer.layers):
                    if layer_idx < len(backbone_layer.layers) - num_layers_to_finetune:
                        layer.trainable = False
                    else:
                        layer.trainable = True  # Убедимся, что они разморожены
            else:
                print(f"  Backbone '{backbone_layer.name}' ПОЛНОСТЬЮ РАЗМОРОЖЕН.")
            current_lr = FINETUNE_LEARNING_RATE_CFG  # Для любого сценария с разморозкой используем finetune_lr
        else:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Слой Backbone '{BACKBONE_LAYER_NAME_IN_MODEL_CFG}' не найден. Разморозка не применена.")
    elif model and not CONTINUE_FROM_CHECKPOINT_CFG:  # Если модель новая и unfreeze_backbone=false
        # Убедимся, что backbone заморожен, как и ожидается от build_object_detector_v1
        backbone_layer = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)
        if backbone_layer:
            backbone_layer.trainable = False  # Явно замораживаем, если build_object_detector_v1 это не сделал
            print(f"INFO: Backbone '{backbone_layer.name}' ЗАМОРОЖЕН (начальное обучение).")

    print("\nСтруктура модели (финальная перед компиляцией):")
    model.summary(line_length=120)
    print(f"\nКомпиляция модели с learning_rate = {current_lr}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                  loss=compute_detector_loss_v1)

    # 4. Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"detector_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}"
    log_dir = os.path.join(LOGS_BASE_DIR_ABS, log_dir_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorboard_callback, EpochTimeLogger()]

    best_model_filename = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_best.keras'
    checkpoint_filepath_best = os.path.join(WEIGHTS_BASE_DIR_ABS, best_model_filename)

    if validation_dataset:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_best, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        callbacks_list.append(model_checkpoint_callback)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_CFG, verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stopping_callback)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=REDUCE_LR_FACTOR_CFG, patience=REDUCE_LR_PATIENCE_CFG,
            verbose=1, min_lr=MIN_LR_ON_PLATEAU_CFG)
        callbacks_list.append(reduce_lr_callback)
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен.")

    # 5. Запуск Обучения
    print(f"\nЗапуск обучения детектора на {EPOCHS_CFG} эпох (initial_epoch для model.fit() = {initial_epoch_for_fit})...")
    print(f"  Обучающая выборка: {len(train_image_paths)} изображений, Аугментация: {current_use_augmentation}")
    if val_image_paths: print(f"  Валидационная выборка: {len(val_image_paths)} изображений.")
    print(f"  Batch Size (для датасета): {BATCH_SIZE_CFG}")  # Тот, что используется в create_detector_tf_dataset
    print(f"  Логи TensorBoard: {log_dir}")

    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS_CFG, # Это общее количество эпох для ЭТОГО сеанса fit
            validation_data=validation_dataset,
            callbacks=callbacks_list,
            verbose=1,
            initial_epoch=initial_epoch_for_fit # <--- ИСПРАВЛЕНО
        )
        print(f"\n--- Обучение детектора (режим: {training_run_description}) завершено ---")

        final_model_filename = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_final_epoch{history.epoch[-1] + 1}_{timestamp}.keras'
        final_model_save_path = os.path.join(WEIGHTS_BASE_DIR_ABS, final_model_filename)
        model.save(final_model_save_path)
        print(f"Финальная модель (на момент остановки) сохранена в: {final_model_save_path}")

        if validation_dataset and os.path.exists(checkpoint_filepath_best):
            print(f"Лучшая модель по val_loss также сохранена в: {checkpoint_filepath_best}")

        history_plot_filename = f'detector_history_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}.png'
        history_plot_save_path = os.path.join(GRAPHS_DIR_ABS, history_plot_filename)
        plot_training_history(history, history_plot_save_path, run_suffix=training_run_description)

    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit() для детектора: {e_fit}");
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()