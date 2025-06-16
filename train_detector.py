# train_detector.py
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path  # Используем pathlib для работы с путями

# --- Определяем корень проекта и добавляем src в sys.path ---
_project_root = Path(__file__).resolve().parent  # Корень проекта, где лежит train_detector.py
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# --- Импорты из твоих модулей в src ---
from datasets.detector_data_loader import create_detector_tf_dataset, USE_AUGMENTATION_CFG
from models.object_detector import build_object_detector_v1_enhanced
from losses.detection_losses import compute_detector_loss_v1

# --- Загрузка Конфигураций ---
_base_config_path = _src_path / 'configs' / 'base_config.yaml'
_detector_config_path = _src_path / 'configs' / 'detector_config.yaml'

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
IMAGES_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS = _project_root / _detector_dataset_ready_path_rel

MODEL_BASE_NAME_CFG = DETECTOR_CONFIG.get('model_base_name', 'RoadDefectDetector_DefaultName')
BACKBONE_LAYER_NAME_IN_MODEL_CFG = DETECTOR_CONFIG.get('backbone_layer_name_in_model', 'Backbone_MobileNetV2')

CONTINUE_FROM_CHECKPOINT_CFG = DETECTOR_CONFIG.get('continue_from_checkpoint', False)
PATH_TO_CHECKPOINT_REL_CFG = DETECTOR_CONFIG.get('path_to_checkpoint', None)  # Имя файла модели в weights/
UNFREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('unfreeze_backbone', False)
UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG = DETECTOR_CONFIG.get('unfreeze_backbone_layers_from_end', 0)
FINETUNE_KEEP_BN_FROZEN = DETECTOR_CONFIG.get('finetune_keep_bn_frozen', True)  # Новый параметр

BATCH_SIZE_CFG = DETECTOR_CONFIG.get('batch_size', 2)
EPOCHS_CFG = DETECTOR_CONFIG.get('epochs', 50)
INITIAL_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('initial_learning_rate', 0.0001)
FINETUNE_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('finetune_learning_rate', 1e-5)

EARLY_STOPPING_PATIENCE_CFG = DETECTOR_CONFIG.get('early_stopping_patience', 15)  # Увеличим немного дефолт
REDUCE_LR_PATIENCE_CFG = DETECTOR_CONFIG.get('reduce_lr_patience', 5)  # Уменьшим для более быстрой реакции
REDUCE_LR_FACTOR_CFG = DETECTOR_CONFIG.get('reduce_lr_factor', 0.2)
MIN_LR_ON_PLATEAU_CFG = DETECTOR_CONFIG.get('min_lr_on_plateau', 1e-7)
USE_AUGMENTATION_TRAIN_CFG = DETECTOR_CONFIG.get('use_augmentation', True)

LOGS_BASE_DIR_ABS = _project_root / BASE_CONFIG.get('logs_base_dir', 'logs')
WEIGHTS_BASE_DIR_ABS = _project_root / BASE_CONFIG.get('weights_base_dir', 'weights')
GRAPHS_DIR_ABS = _project_root / BASE_CONFIG.get('graphs_dir', 'graphs')
os.makedirs(GRAPHS_DIR_ABS, exist_ok=True)
os.makedirs(WEIGHTS_BASE_DIR_ABS, exist_ok=True)
os.makedirs(LOGS_BASE_DIR_ABS, exist_ok=True)


# --- Вспомогательные Функции ---
# collect_split_data_paths (без изменений, как в твоей последней версии)
def collect_split_data_paths(split_dir_abs_path, images_subdir, annotations_subdir):
    # ... (твой код collect_split_data_paths) ...
    image_paths = []
    xml_paths = []
    current_images_dir = os.path.join(split_dir_abs_path, images_subdir)
    current_annotations_dir = os.path.join(split_dir_abs_path, annotations_subdir)
    if not os.path.isdir(current_images_dir) or not os.path.isdir(current_annotations_dir):
        return image_paths, xml_paths  # Возвращаем пустые списки, если директории нет
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


# EpochTimeLogger (без изменений)
class EpochTimeLogger(tf.keras.callbacks.Callback):
    # ... (твой код EpochTimeLogger) ...
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f" - Время эпохи: {epoch_duration:.2f} сек")


# plot_training_history (без изменений)
def plot_training_history(history, save_path_plot, run_suffix=""):
    # ... (твой код plot_training_history) ...
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
    print(f"\n--- Запуск Обучения Детектора Объектов (Версия с Улучшенным Управлением Fine-tuning'ом) ---")

    # 1. Определение режима и параметров
    training_run_description = "initial_frozen_bb"
    current_lr = INITIAL_LEARNING_RATE_CFG
    current_use_augmentation = USE_AUGMENTATION_CFG  # По умолчанию берем из конфига
    model_to_load_path_abs = None
    perform_fine_tuning_on_backbone = False  # Флаг, указывающий, нужно ли размораживать backbone

    if PATH_TO_CHECKPOINT_REL_CFG:
        model_to_load_path_abs = WEIGHTS_BASE_DIR_ABS / PATH_TO_CHECKPOINT_REL_CFG

    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path_abs and model_to_load_path_abs.exists():
        print(f"Режим: Продолжение обучения / Fine-tuning с чекпоинта: {model_to_load_path_abs}")
        if UNFREEZE_BACKBONE_CFG:
            training_run_description = "finetune_bb"
            current_lr = FINETUNE_LEARNING_RATE_CFG
            perform_fine_tuning_on_backbone = True  # Устанавливаем флаг
            # current_use_augmentation остается из конфига (USE_AUGMENTATION_CFG)
            print(f"  Backbone будет разморожен (частично или полностью). Learning rate: {current_lr}")
            print(f"  Аугментация для fine-tuning'а: {current_use_augmentation}")
        else:  # Продолжение обучения с замороженным backbone
            training_run_description = "continued_frozen_bb"
            # LR остается INITIAL_LEARNING_RATE_CFG или должен быть меньше?
            # Обычно для продолжения берут текущий LR из оптимизатора модели или немного меньше.
            # Пока оставим INITIAL_LEARNING_RATE_CFG, но это место для возможной корректировки.
            print(f"  Backbone останется замороженным. Learning rate: {current_lr}")
            print(f"  Аугментация: {current_use_augmentation}")
    elif UNFREEZE_BACKBONE_CFG:  # Обучение с нуля, но backbone сразу разморожен (не рекомендуется)
        training_run_description = "initial_unfrozen_bb_WARNING"  # Добавим WARNING
        current_lr = FINETUNE_LEARNING_RATE_CFG
        perform_fine_tuning_on_backbone = True
        print(f"ПРЕДУПРЕЖДЕНИЕ: Начальное обучение с РАЗМОРОЖЕННЫМ backbone. Это может быть нестабильно.")
        print(f"  Learning rate: {current_lr}, Аугментация: {current_use_augmentation}")
    else:  # Стандартное начальное обучение с замороженным backbone
        print(f"Режим: Начальное обучение с ЗАМОРОЖЕННЫМ backbone.")
        print(f"  Learning rate: {current_lr}, Аугментация: {current_use_augmentation}")
        # perform_fine_tuning_on_backbone остается False

    print(f"Итоговое описание запуска: {training_run_description}")

    # 2. Подготовка данных
    # ... (код сбора train/val путей остается таким же) ...
    train_split_dir = DETECTOR_DATASET_READY_ABS / "train"
    val_split_dir = DETECTOR_DATASET_READY_ABS / "validation"
    train_image_paths, train_xml_paths = collect_split_data_paths(str(train_split_dir), IMAGES_SUBDIR_NAME_DET,
                                                                  ANNOTATIONS_SUBDIR_NAME_DET)
    val_image_paths, val_xml_paths = collect_split_data_paths(str(val_split_dir), IMAGES_SUBDIR_NAME_DET,
                                                              ANNOTATIONS_SUBDIR_NAME_DET)
    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(f"\nНайдено для обучения: {len(train_image_paths)}.")
    if val_image_paths: print(f"Найдено для валидации: {len(val_image_paths)}.")

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

    # 3. Создание или Загрузка Модели
    model = None
    initial_epoch = 0  # Для model.fit, если продолжаем обучение

    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path_abs and model_to_load_path_abs.exists():
        print(f"\nЗагрузка существующей модели из: {model_to_load_path_abs}")
        try:
            model = tf.keras.models.load_model(
                str(model_to_load_path_abs),  # Path объекту нужен str()
                custom_objects={'compute_detector_loss_v1': compute_detector_loss_v1},
                compile=False  # Загружаем без компиляции, чтобы потом применить новый LR или разморозку
            )
            print("Модель успешно загружена.")
            # Здесь можно было бы попытаться извлечь последнюю эпоху из имени файла, но это усложнит.
            # Оставим initial_epoch = 0, TensorBoard все равно покажет общую картину по шагам.
        except Exception as e:
            print(f"Ошибка при загрузке модели из чекпоинта: {e}. Будет создана новая модель.")
            model = None

    if model is None:
        if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path_abs:  # Если чекпоинт был указан, но не загрузился
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить модель из '{model_to_load_path_abs}'.")
        print("\nСоздание новой модели детектора (build_object_detector_v1_enhanced)...")
        model = build_object_detector_v1_enhanced()  # Эта функция должна создавать backbone замороженным по умолчанию
        # Если это не "продолжение", то начальная заморозка уже должна быть в build_object_detector_v1_enhanced
        # на основе DETECTOR_CONFIG['freeze_backbone'] (который сейчас true).

    # Применяем разморозку, ЕСЛИ это указано и модель существует
    if model and perform_fine_tuning_on_backbone:  # perform_fine_tuning_on_backbone устанавливается выше
        print(f"\nПрименение параметров Fine-tuning'а к Backbone...")
        try:
            backbone_model_internal = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)

            # Сначала делаем весь backbone обучаемым
            backbone_model_internal.trainable = True

            num_layers_to_finetune_from_end = UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG

            if num_layers_to_finetune_from_end > 0:  # Если указано частичное размораживание
                print(
                    f"  Размораживаются ПОСЛЕДНИЕ {num_layers_to_finetune_from_end} слоев в Backbone '{backbone_model_internal.name}'.")
                # Замораживаем все слои, КРОМЕ последних N
                for i, layer in enumerate(backbone_model_internal.layers):
                    if i < len(backbone_model_internal.layers) - num_layers_to_finetune_from_end:
                        layer.trainable = False
                    else:
                        # Для BN слоев в размораживаемой части, если хотим их оставить замороженными
                        if FINETUNE_KEEP_BN_FROZEN and isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.trainable = False
                            # print(f"    Слой BN '{layer.name}' оставлен замороженным при fine-tuning'е.")
                        else:
                            layer.trainable = True  # Убеждаемся, что эти слои обучаемы
            else:  # num_layers_to_finetune_from_end == 0 или не указан => размораживаем все
                print(f"  Backbone '{backbone_model_internal.name}' ПОЛНОСТЬЮ РАЗМОРОЖЕН (все слои trainable=True).")
                if FINETUNE_KEEP_BN_FROZEN:  # Но BN слои все равно замораживаем, если указано
                    for layer_in_backbone in backbone_model_internal.layers:
                        if isinstance(layer_in_backbone, tf.keras.layers.BatchNormalization):
                            layer_in_backbone.trainable = False
                    print("    (При этом BatchNormalization слои в Backbone оставлены замороженными согласно конфигу)")

        except ValueError:  # Если слой backbone не найден
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Слой Backbone '{BACKBONE_LAYER_NAME_IN_MODEL_CFG}' не найден в загруженной/созданной модели. Разморозка не применена.")
        except Exception as e_unfreeze:
            print(f"ОШИБКА при попытке разморозить backbone: {e_unfreeze}")
    elif model and not perform_fine_tuning_on_backbone:  # Начальное обучение или продолжение с замороженным
        backbone_model_internal = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)
        if backbone_model_internal:
            backbone_model_internal.trainable = False  # Убедимся, что он заморожен
            print(f"INFO: Backbone '{backbone_model_internal.name}' используется как ЗАМОРОЖЕННЫЙ.")

    print("\nСтруктура модели (финальная перед компиляцией):")
    model.summary(line_length=120)  # Выводим summary ПОСЛЕ всех манипуляций с trainable

    print(f"\nКомпиляция модели с learning_rate = {current_lr}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                  loss=compute_detector_loss_v1)

    # 4. Callbacks
    # ... (код коллбэков такой же, как в твоей последней версии, но имя файла для лучшей модели будет динамическим)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"detector_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}"
    log_dir = LOGS_BASE_DIR_ABS / log_dir_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    callbacks_list = [tensorboard_callback, EpochTimeLogger()]

    # Имя для лучшей модели будет фиксированным, чтобы мы всегда знали, где лежит лучшая модель ЭТОГО ТИПА запуска
    best_model_filename = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_best.keras'
    checkpoint_filepath_best = WEIGHTS_BASE_DIR_ABS / best_model_filename

    if validation_dataset:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath_best), save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        callbacks_list.append(model_checkpoint_callback)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_CFG, verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stopping_callback)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=REDUCE_LR_FACTOR_CFG, patience=REDUCE_LR_PATIENCE_CFG,
            verbose=1, min_lr=MIN_LR_ON_PLATEAU_CFG)
        callbacks_list.append(reduce_lr_callback)

    # 5. Запуск Обучения
    # ... (код запуска model.fit() и сохранения финальной модели и графика такой же) ...
    print(
        f"\nЗапуск обучения детектора на {EPOCHS_CFG} эпох (initial_epoch = {initial_epoch})...")  # Используем initial_epoch
    # ... (остальные print'ы)
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS_CFG,
            validation_data=validation_dataset,
            callbacks=callbacks_list,
            verbose=1,
            initial_epoch=initial_epoch
        )
        # ... (код сохранения финальной модели и графика) ...
        final_model_filename = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_final_epoch{history.epoch[-1] + 1}_{timestamp}.keras'
        final_model_save_path = WEIGHTS_BASE_DIR_ABS / final_model_filename
        model.save(final_model_save_path)
        print(f"Финальная модель сохранена в: {final_model_save_path}")
        if validation_dataset and checkpoint_filepath_best.exists():
            print(f"Лучшая модель по val_loss также сохранена/обновлена в: {checkpoint_filepath_best}")
        history_plot_filename = f'detector_history_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}.png'
        history_plot_save_path = GRAPHS_DIR_ABS / history_plot_filename
        plot_training_history(history, str(history_plot_save_path), run_suffix=training_run_description)

    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit(): {e_fit}");
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()