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
from pathlib import Path

# --- Определяем корень проекта и добавляем src в sys.path ---
_project_root = Path(__file__).resolve().parent
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# --- Импорты из твоих модулей в src ---
from datasets.detector_data_loader import create_detector_tf_dataset
from models.object_detector import build_object_detector_v2_fpn
from losses.detection_losses import compute_detector_loss_v2_fpn

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
    print("ОШИБКА: Не удалось загрузить один или несколько файлов конфигурации. Выход.");
    exit()


# --- Вспомогательные Функции и Классы ---
def collect_split_data_paths(split_dir_abs_path_str, images_subdir_name, annotations_subdir_name):
    image_paths, xml_paths = [], []
    split_dir_abs_path = Path(split_dir_abs_path_str)
    current_images_dir = split_dir_abs_path / images_subdir_name
    current_annotations_dir = split_dir_abs_path / annotations_subdir_name
    if not current_images_dir.is_dir() or not current_annotations_dir.is_dir():
        return image_paths, xml_paths
    valid_extensions = ['.jpg', '.jpeg', '.png']
    img_files = []
    for ext in valid_extensions:
        img_files.extend(list(current_images_dir.glob(f"*{ext.lower()}")))
        img_files.extend(list(current_images_dir.glob(f"*{ext.upper()}")))
    img_files = sorted(list(set(img_files)))
    for img_p_obj in img_files:
        xml_p_obj = current_annotations_dir / (img_p_obj.stem + ".xml")
        if xml_p_obj.exists():
            image_paths.append(str(img_p_obj))
            xml_paths.append(str(xml_p_obj))
    return image_paths, xml_paths


def plot_training_history(history, save_path_plot_obj, run_desc=""):
    plt.figure(figsize=(12, 6))
    loss_exists = 'loss' in history.history and history.history['loss']
    val_loss_exists = 'val_loss' in history.history and history.history['val_loss']

    if loss_exists:
        epochs_range_train = range(len(history.history['loss']))
        plt.plot(epochs_range_train, history.history['loss'], label='Train Loss')
    if val_loss_exists:
        epochs_range_val = range(len(history.history['val_loss']))
        plt.plot(epochs_range_val, history.history['val_loss'], label='Validation Loss')

    title_str = f'История обучения детектора ({run_desc})' if run_desc else 'История обучения детектора'
    plt.title(title_str);
    plt.ylabel('Loss');
    plt.xlabel('Epoch')

    if loss_exists or val_loss_exists:
        plt.legend(loc='upper right')

    plt.grid(True);
    plt.tight_layout()
    try:
        plt.savefig(str(save_path_plot_obj));
        print(f"График истории обучения сохранен в: {save_path_plot_obj}")
    except Exception as e:
        print(f"ОШИБКА сохранения графика: {e}")
    plt.close()


# Коллбэк для логирования LR в TensorBoard и вывода в консоль
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Попытка получить LR перед началом эпохи (может быть полезно, если он меняется в on_epoch_begin других коллбэков)
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if callable(lr):  # Если это LearningRateSchedule
                lr_val = lr(self.model.optimizer.iterations).numpy() if hasattr(self.model.optimizer,
                                                                                'iterations') and self.model.optimizer.iterations is not None else lr(
                    epoch).numpy()  # Приблизительно
            elif isinstance(lr, tf.Variable):
                lr_val = lr.numpy()
            else:
                lr_val = float(lr)  # Старые оптимизаторы
            # print(f"Epoch {epoch+1}: Learning Rate = {lr_val:.2e}") # Можно раскомментировать для вывода перед эпохой
            if logs is not None:  # Добавляем в логи TensorBoard
                logs['learning_rate'] = lr_val

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if callable(lr):
                lr_val = lr(self.model.optimizer.iterations).numpy() if hasattr(self.model.optimizer,
                                                                                'iterations') and self.model.optimizer.iterations is not None else lr(
                    epoch + 1).numpy()  # +1 для LR следующей эпохи
            elif isinstance(lr, tf.Variable):
                lr_val = lr.numpy()
            else:
                lr_val = float(lr)
            logs['learning_rate'] = lr_val  # Обновляем для логов в конце эпохи
            # Стандартный вывод Keras verbose=1 уже включает loss, val_loss и время.
            # Мы можем добавить LR в print после стандартного вывода, если очень нужно,
            # но он уже будет в TensorBoard.
            # print(f"Epoch {epoch+1} Summary: Effective LR for next epoch (approx): {lr_val:.2e}")


# --- Основная функция обучения ---
def train_detector_main():
    # --- Извлечение Параметров (без изменений) ---
    images_subdir_name_param = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
    annotations_subdir_name_param = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')
    detector_dataset_ready_rel_param = "data/Detector_Dataset_Ready"
    detector_dataset_ready_abs_param = _project_root / detector_dataset_ready_rel_param
    model_base_name_param = DETECTOR_CONFIG.get('model_base_name', 'RoadDefectDetector_FPN')
    backbone_layer_name_in_model_param = DETECTOR_CONFIG.get('backbone_layer_name_in_model', 'Backbone_MobileNetV2_FPN')
    continue_from_checkpoint_param = DETECTOR_CONFIG.get('continue_from_checkpoint', False)
    path_to_checkpoint_rel_param = DETECTOR_CONFIG.get('path_to_checkpoint', None)
    unfreeze_backbone_param = DETECTOR_CONFIG.get('unfreeze_backbone', False)
    unfreeze_layers_from_end_param = DETECTOR_CONFIG.get('unfreeze_backbone_layers_from_end', 0)
    finetune_keep_bn_frozen_param = DETECTOR_CONFIG.get('finetune_keep_bn_frozen', True)
    batch_size_param = DETECTOR_CONFIG.get('batch_size', 4)
    epochs_param = DETECTOR_CONFIG.get('epochs', 150)
    initial_lr_param = DETECTOR_CONFIG.get('initial_learning_rate', 0.0005)
    finetune_lr_param = DETECTOR_CONFIG.get('finetune_learning_rate', 1e-5)
    early_stop_patience_param = DETECTOR_CONFIG.get('early_stopping_patience', 25)
    reduce_lr_patience_param = DETECTOR_CONFIG.get('reduce_lr_patience', 10)
    reduce_lr_factor_param = DETECTOR_CONFIG.get('reduce_lr_factor', 0.2)
    min_lr_plateau_param = DETECTOR_CONFIG.get('min_lr_on_plateau', 1e-7)
    use_augmentation_train_param = DETECTOR_CONFIG.get('use_augmentation', True)
    logs_base_dir_abs_param = _project_root / BASE_CONFIG.get('logs_base_dir', 'logs')
    weights_base_dir_abs_param = _project_root / BASE_CONFIG.get('weights_base_dir', 'weights')
    graphs_dir_abs_param = _project_root / BASE_CONFIG.get('graphs_dir', 'graphs')
    for p_dir_param in [graphs_dir_abs_param, weights_base_dir_abs_param, logs_base_dir_abs_param]:
        p_dir_param.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Запуск Обучения Детектора Объектов (FPN Архитектура) ---")
    print(f"--- Конфигурация Обучения ---")
    # ... (вывод основных параметров конфига) ...
    print(f"  Путь к датасету: {detector_dataset_ready_abs_param}")
    print(f"  Имя базовой модели: {model_base_name_param}")
    print(f"  Batch Size: {batch_size_param}")
    print(f"  Всего эпох для этого запуска: {epochs_param}")
    print(f"  Аугментация для обучающей выборки: {use_augmentation_train_param}")

    model = None
    initial_epoch_to_start_fit = 0
    training_mode_description_var = "initial_frozen_bb"
    current_actual_lr_var = initial_lr_param

    # ... (логика загрузки/создания модели и определения режима как была) ...
    model_checkpoint_to_load_abs_var = None
    if path_to_checkpoint_rel_param:
        model_checkpoint_to_load_abs_var = weights_base_dir_abs_param / path_to_checkpoint_rel_param
    load_model_attempted = False
    if continue_from_checkpoint_param and model_checkpoint_to_load_abs_var and model_checkpoint_to_load_abs_var.exists():
        load_model_attempted = True;
        print(f"  Режим: Загрузка модели из чекпоинта: {model_checkpoint_to_load_abs_var}")
        try:
            model = tf.keras.models.load_model(str(model_checkpoint_to_load_abs_var),
                                               custom_objects={
                                                   'compute_detector_loss_v2_fpn': compute_detector_loss_v2_fpn},
                                               compile=False)
            print("  Модель успешно загружена.")
            if unfreeze_backbone_param:
                training_mode_description_var = "finetune_bb"; current_actual_lr_var = finetune_lr_param
            else:
                training_mode_description_var = "continue_frozen_bb"
        except Exception as e_load:
            print(f"  Ошибка при загрузке модели: {e_load}. Будет создана новая модель."); model = None
    if model is None:
        if load_model_attempted: print(f"  ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить чекпоинт. Создается новая модель.")
        print(f"  Создание новой FPN модели (build_object_detector_v2_fpn)...")
        model = build_object_detector_v2_fpn()
        if unfreeze_backbone_param:
            training_mode_description_var = "initial_unfrozen_bb";
            current_actual_lr_var = finetune_lr_param
            print(
                f"  ПРЕДУПРЕЖДЕНИЕ: Начальное обучение новой модели с РАЗМОРОЖЕННЫМ backbone. LR: {current_actual_lr_var}")
        else:
            training_mode_description_var = "initial_frozen_bb"
            print(f"  Режим: Начальное обучение новой модели с ЗАМОРОЖЕННЫМ backbone. LR: {current_actual_lr_var}")
    if unfreeze_backbone_param:
        print(f"  Применение параметров Fine-tuning'а к Backbone...")
        try:
            backbone_layer = model.get_layer(backbone_layer_name_in_model_param)
            backbone_layer.trainable = True
            num_layers_to_finetune = unfreeze_layers_from_end_param
            if num_layers_to_finetune > 0 and hasattr(backbone_layer, 'layers'):
                print(f"    Размораживаются ПОСЛЕДНИЕ {num_layers_to_finetune} слоев в '{backbone_layer.name}'.")
                num_bb_layers = len(backbone_layer.layers)
                for i, layer_obj in enumerate(backbone_layer.layers):
                    if i < num_bb_layers - num_layers_to_finetune:
                        layer_obj.trainable = False
                    else:
                        if finetune_keep_bn_frozen_param and isinstance(layer_obj, tf.keras.layers.BatchNormalization):
                            layer_obj.trainable = False
                        else:
                            layer_obj.trainable = True
            elif num_layers_to_finetune == 0:
                print(f"    Backbone '{backbone_layer.name}' ПОЛНОСТЬЮ РАЗМОРОЖЕН.")
                if finetune_keep_bn_frozen_param:
                    for layer_in_bb_obj in backbone_layer.layers:
                        if isinstance(layer_in_bb_obj,
                                      tf.keras.layers.BatchNormalization): layer_in_bb_obj.trainable = False
                    print("      (BatchNormalization слои в Backbone оставлены замороженными)")
        except Exception as e_unfreeze:
            print(f"  ОШИБКА при разморозке backbone: {e_unfreeze}")
    elif model:
        try:
            backbone_layer = model.get_layer(backbone_layer_name_in_model_param)
            if backbone_layer.trainable:
                backbone_layer.trainable = False
                print(f"  INFO: Backbone '{backbone_layer.name}' принудительно ЗАМОРОЖЕН (unfreeze_backbone=false).")
        except ValueError:
            pass
    print(f"  Итоговый режим обучения для этого запуска: {training_mode_description_var}")
    print(f"  Используемый Learning Rate для компиляции: {current_actual_lr_var}")
    print(f"  Аугментация для обучающей выборки: {use_augmentation_train_param}")

    # 2. Подготовка данных (без изменений)
    # ...
    train_split_dir = detector_dataset_ready_abs_param / "train"
    val_split_dir = detector_dataset_ready_abs_param / "validation"
    train_image_paths, train_xml_paths = collect_split_data_paths(str(train_split_dir), images_subdir_name_param,
                                                                  annotations_subdir_name_param)
    val_image_paths, val_xml_paths = collect_split_data_paths(str(val_split_dir), images_subdir_name_param,
                                                              annotations_subdir_name_param)
    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(
        f"\n  Найдено для обучения: {len(train_image_paths)} файлов. Для валидации: {len(val_image_paths) if val_image_paths else 'НЕТ'}.")
    train_dataset = create_detector_tf_dataset(train_image_paths, train_xml_paths, batch_size_param, shuffle=True,
                                               augment=use_augmentation_train_param)
    validation_dataset = None
    if val_image_paths: validation_dataset = create_detector_tf_dataset(val_image_paths, val_xml_paths,
                                                                        batch_size_param, shuffle=False, augment=False)
    if train_dataset is None: print("Не удалось создать обучающий датасет. Выход."); return

    print("\nСтруктура модели (финальная перед компиляцией):");
    model.summary(line_length=150)
    print(f"\nКомпиляция модели с learning_rate = {current_actual_lr_var}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_actual_lr_var),
                  loss=compute_detector_loss_v2_fpn)

    # 4. Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"detector_{model_base_name_param}_{training_mode_description_var}_{timestamp}"
    log_dir = logs_base_dir_abs_param / log_dir_name
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1,
                                                    write_graph=False)  # write_graph=False для экономии места

    callbacks_list = [tensorboard_cb, LearningRateLogger()]  # Используем новый логгер LR

    best_model_filename_cb_name = f'{model_base_name_param}_{training_mode_description_var}_best.keras'
    checkpoint_filepath_best_abs_cb_path = weights_base_dir_abs_param / best_model_filename_cb_name

    if validation_dataset:
        mcp_cb = tf.keras.callbacks.ModelCheckpoint(str(checkpoint_filepath_best_abs_cb_path), save_weights_only=False,
                                                    monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=early_stop_patience_param, verbose=1,
                                                 restore_best_weights=True)
        # ReduceLROnPlateau будет выводить информацию о смене LR сам
        rlrop_cb = tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=reduce_lr_factor_param,
                                                        patience=reduce_lr_patience_param, verbose=1,
                                                        min_lr=min_lr_plateau_param)
        callbacks_list.extend([mcp_cb, es_cb, rlrop_cb])
        print(f"  Лучшая модель будет сохраняться в: {checkpoint_filepath_best_abs_cb_path} (по val_loss)")

    # 5. Запуск Обучения
    print(f"\n--- Запуск model.fit() ---")
    print(f"  Эпох: {epochs_param}, Начальная эпоха: {initial_epoch_to_start_fit}")
    print(f"  Логи TensorBoard: {log_dir}")

    overall_training_start_time = time.time()  # Замеряем общее время обучения

    try:
        history = model.fit(
            train_dataset,
            epochs=epochs_param,
            validation_data=validation_dataset,
            callbacks=callbacks_list,
            verbose=1,  # verbose=1 для стандартного прогресс-бара Keras, который включает время/шаг
            initial_epoch=initial_epoch_to_start_fit
        )

        overall_training_end_time = time.time()
        total_training_duration_sec = overall_training_end_time - overall_training_start_time
        total_training_duration_min = total_training_duration_sec / 60
        total_training_duration_hr = total_training_duration_min / 60

        print(f"\n--- Обучение детектора (режим: {training_mode_description_var}) завершено ---")
        print(
            f"Общее время обучения: {total_training_duration_sec:.2f} сек ({total_training_duration_min:.2f} мин / {total_training_duration_hr:.2f} час)")

        # ... (код сохранения финальной модели и графика такой же) ...
        final_epoch_num_actual = history.epoch[-1] + 1 if history.epoch else initial_epoch_to_start_fit + epochs_param
        final_model_filename_ts_name = f'{model_base_name_param}_{training_mode_description_var}_final_epoch{final_epoch_num_actual}_{timestamp}.keras'
        final_model_save_path_abs_path = weights_base_dir_abs_param / final_model_filename_ts_name
        model.save(final_model_save_path_abs_path)
        print(f"Финальная модель (на момент остановки) сохранена в: {final_model_save_path_abs_path}")
        if validation_dataset and checkpoint_filepath_best_abs_cb_path.exists():
            print(f"Лучшая модель по val_loss также сохранена/обновлена в: {checkpoint_filepath_best_abs_cb_path}")
        plot_save_path_abs_path = graphs_dir_abs_param / f'detector_history_{model_base_name_param}_{training_mode_description_var}_{timestamp}.png'
        plot_training_history(history, str(plot_save_path_abs_path), run_desc=training_mode_description_var)

    except Exception as e_fit:
        print(f"ОШИБКА model.fit(): {e_fit}");
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()