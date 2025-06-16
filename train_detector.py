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

# --- Загрузка Конфигураций (Глобально, чтобы они были доступны для функций) ---
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
    # Преобразуем в Path объект для удобства
    split_dir_abs_path = Path(split_dir_abs_path_str)
    current_images_dir = split_dir_abs_path / images_subdir_name
    current_annotations_dir = split_dir_abs_path / annotations_subdir_name

    if not current_images_dir.is_dir() or not current_annotations_dir.is_dir():
        # print(f"  ПРЕДУПРЕЖДЕНИЕ: Директория {current_images_dir} или {current_annotations_dir} не найдена для split'а. Пропускаем.")
        return image_paths, xml_paths  # Возвращаем пустые списки

    img_files = []
    # Ищем изображения с разными расширениями, без учета регистра через Path.glob
    for ext_pattern in ['*.jpg', '*.jpeg', '*.png']:
        img_files.extend(list(current_images_dir.glob(ext_pattern)))
        img_files.extend(list(current_images_dir.glob(ext_pattern.upper())))  # для .JPG и т.д.

    img_files = sorted(list(set(img_files)))  # Уникальные и отсортированные

    for img_p_obj in img_files:
        xml_p_obj = current_annotations_dir / (img_p_obj.stem + ".xml")  # Имя файла без расширения + .xml
        if xml_p_obj.exists():
            image_paths.append(str(img_p_obj))
            xml_paths.append(str(xml_p_obj))
        # else:
        # print(f"    ПРЕДУПРЕЖДЕНИЕ (collect_split): XML для {img_p_obj.name} не найден в {current_annotations_dir}")
    return image_paths, xml_paths


class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        lr_val = "N/A (оптимизатор не найден)"
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            current_lr_opt = self.model.optimizer.learning_rate
            if isinstance(current_lr_opt, tf.Variable):
                lr_val = current_lr_opt.numpy()
            elif isinstance(current_lr_opt, tf.keras.optimizers.schedules.LearningRateSchedule):
                if hasattr(self.model.optimizer, 'iterations') and self.model.optimizer.iterations is not None:
                    try:
                        lr_val = current_lr_opt(self.model.optimizer.iterations).numpy()
                    except:
                        pass  # Останется "N/A" если iterations не инициализирован
                else:
                    lr_val = "N/A (scheduled, no iterations)"
            elif isinstance(current_lr_opt, (float, np.float32, np.float64)):  # Добавил np.float64
                lr_val = float(current_lr_opt)

        logs['epoch_duration_sec'] = epoch_duration  # Для TensorBoard
        lr_str = f"{lr_val:.1e}" if isinstance(lr_val, float) else str(lr_val)
        # Keras verbose=1 выводит loss, val_loss, и время на шаг. Мы добавим эту строку в конце.
        # Чтобы она появилась после стандартного вывода Keras, мы печатаем ее здесь.
        # Стандартный вывод Keras для эпохи заканчивается переводом строки, так что это будет следующей строкой.
        print(f"Epoch {epoch + 1:03d} processed. Duration: {epoch_duration:7.2f}s, Current LR: {lr_str}")


def plot_training_history(history, save_path_plot_obj, run_desc=""):
    plt.figure(figsize=(12, 5))
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss']:  # Проверяем, что val_loss не пустой
        plt.plot(history.history['val_loss'], label='Validation Loss')

    title_str = f'История обучения детектора ({run_desc})' if run_desc else 'История обучения детектора'
    plt.title(title_str)
    plt.ylabel('Loss');
    plt.xlabel('Epoch')
    if 'loss' in history.history or ('val_loss' in history.history and history.history['val_loss']):
        plt.legend(loc='upper right')  # Показываем легенду, только если есть что показывать
    plt.grid(True);
    plt.tight_layout()
    try:
        plt.savefig(str(save_path_plot_obj))
        print(f"График истории обучения сохранен в: {save_path_plot_obj}")
    except Exception as e:
        print(f"ОШИБКА сохранения графика истории обучения: {e}")
    plt.close()


# --- Основная функция обучения ---
def train_detector_main():
    # --- Извлечение Параметров из Глобальных Конфигов ВНУТРИ Функции ---
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

    early_stop_patience_param = DETECTOR_CONFIG.get('early_stopping_patience', 20)
    reduce_lr_patience_param = DETECTOR_CONFIG.get('reduce_lr_patience', 7)
    reduce_lr_factor_param = DETECTOR_CONFIG.get('reduce_lr_factor', 0.2)
    min_lr_plateau_param = DETECTOR_CONFIG.get('min_lr_on_plateau', 1e-7)
    use_augmentation_train_param = DETECTOR_CONFIG.get('use_augmentation', True)

    logs_base_dir_abs_param = _project_root / BASE_CONFIG.get('logs_base_dir', 'logs')
    weights_base_dir_abs_param = _project_root / BASE_CONFIG.get('weights_base_dir', 'weights')
    graphs_dir_abs_param = _project_root / BASE_CONFIG.get('graphs_dir', 'graphs')
    for p_dir_param in [graphs_dir_abs_param, weights_base_dir_abs_param, logs_base_dir_abs_param]:
        p_dir_param.mkdir(parents=True, exist_ok=True)
    # --- Конец Извлечения Параметров ---

    print(f"\n--- Запуск Обучения Детектора Объектов (FPN Архитектура) ---")

    model = None
    initial_epoch_to_start_fit = 0
    training_mode_description_var = "initial_frozen_bb"  # Дефолтное значение
    current_actual_lr_var = initial_lr_param
    current_actual_aug_var = use_augmentation_train_param

    model_checkpoint_to_load_abs_var = None
    if path_to_checkpoint_rel_param:
        model_checkpoint_to_load_abs_var = weights_base_dir_abs_param / path_to_checkpoint_rel_param

    load_model_attempted = False
    if continue_from_checkpoint_param and model_checkpoint_to_load_abs_var and model_checkpoint_to_load_abs_var.exists():
        load_model_attempted = True
        print(f"Режим: Загрузка модели из чекпоинта: {model_checkpoint_to_load_abs_var}")
        try:
            model = tf.keras.models.load_model(
                str(model_checkpoint_to_load_abs_var),
                custom_objects={'compute_detector_loss_v2_fpn': compute_detector_loss_v2_fpn}, compile=False)
            print("Модель успешно загружена.")
            if unfreeze_backbone_param:
                training_mode_description_var = "finetune_bb"
                current_actual_lr_var = finetune_lr_param
            else:
                training_mode_description_var = "continue_frozen_bb"
        except Exception as e_load:
            print(f"Ошибка при загрузке модели из чекпоинта: {e_load}. Будет создана новая модель.")
            model = None

    if model is None:
        if load_model_attempted: print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить чекпоинт. Создается новая модель.")
        print("\nСоздание новой FPN модели (build_object_detector_v2_fpn)...")
        model = build_object_detector_v2_fpn()
        if unfreeze_backbone_param:
            training_mode_description_var = "initial_unfrozen_bb"
            current_actual_lr_var = finetune_lr_param
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Начальное обучение новой модели с РАЗМОРОЖЕННЫМ backbone. LR: {current_actual_lr_var}")
            try:
                backbone_layer = model.get_layer(backbone_layer_name_in_model_param)
                backbone_layer.trainable = True
                if unfreeze_layers_from_end_param == 0 and finetune_keep_bn_frozen_param:  # Полная разморозка, но BN заморожены
                    for layer_in_bb in backbone_layer.layers:
                        if isinstance(layer_in_bb, tf.keras.layers.BatchNormalization): layer_in_bb.trainable = False
                    print("    (BatchNormalization слои в Backbone оставлены замороженными)")
                # Логика частичной разморозки также должна учитывать finetune_keep_bn_frozen_param
            except Exception as e_unf_new:
                print(f"Ошибка при начальной разморозке нового backbone: {e_unf_new}")
        else:
            training_mode_description_var = "initial_frozen_bb"
            current_actual_lr_var = initial_lr_param
            print(f"Режим: Начальное обучение новой модели с ЗАМОРОЖЕННЫМ backbone. LR: {current_actual_lr_var}")

    # Применяем разморозку к ЗАГРУЖЕННОЙ модели, если это режим fine-tuning
    if model and training_mode_description_var == "finetune_bb":
        print(f"\nПрименение параметров Fine-tuning'а к Backbone (загруженному)...")
        try:
            backbone_layer = model.get_layer(backbone_layer_name_in_model_param)
            backbone_layer.trainable = True
            if unfreeze_layers_from_end_param > 0 and hasattr(backbone_layer, 'layers'):
                print(f"  Размораживаются ПОСЛЕДНИЕ {unfreeze_layers_from_end_param} слоев в '{backbone_layer.name}'.")
                num_bb_layers = len(backbone_layer.layers)
                for i, layer in enumerate(backbone_layer.layers):
                    if i < num_bb_layers - unfreeze_layers_from_end_param:
                        layer.trainable = False
                    else:
                        if finetune_keep_bn_frozen_param and isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.trainable = False
                        else:
                            layer.trainable = True
            elif unfreeze_layers_from_end_param == 0:
                print(f"  Backbone '{backbone_layer.name}' ПОЛНОСТЬЮ РАЗМОРОЖЕН.")
                if finetune_keep_bn_frozen_param:
                    for layer_in_bb in backbone_layer.layers:
                        if isinstance(layer_in_bb, tf.keras.layers.BatchNormalization): layer_in_bb.trainable = False
                    print("    (BatchNormalization слои в Backbone оставлены замороженными)")
        except Exception as e_unfreeze:
            print(f"ОШИБКА при разморозке backbone для fine-tuning'а: {e_unfreeze}")

    print(f"Итоговый режим обучения для этого запуска: {training_mode_description_var}")

    # 2. Подготовка данных
    train_split_dir = detector_dataset_ready_abs_param / "train"
    val_split_dir = detector_dataset_ready_abs_param / "validation"
    train_image_paths, train_xml_paths = collect_split_data_paths(str(train_split_dir), images_subdir_name_param,
                                                                  annotations_subdir_name_param)
    val_image_paths, val_xml_paths = collect_split_data_paths(str(val_split_dir), images_subdir_name_param,
                                                              annotations_subdir_name_param)
    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(
        f"\nНайдено для обучения: {len(train_image_paths)} файлов. Для валидации: {len(val_image_paths) if val_image_paths else 'НЕТ'}.")

    train_dataset = create_detector_tf_dataset(train_image_paths, train_xml_paths, batch_size_param, shuffle=True,
                                               augment=current_actual_aug_var)
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
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    callbacks_list = [tensorboard_cb, EpochTimeLogger()]

    best_model_filename_cb_name = f'{model_base_name_param}_{training_mode_description_var}_best.keras'
    checkpoint_filepath_best_abs_cb_path = weights_base_dir_abs_param / best_model_filename_cb_name

    if validation_dataset:
        mcp_cb = tf.keras.callbacks.ModelCheckpoint(str(checkpoint_filepath_best_abs_cb_path), save_weights_only=False,
                                                    monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=early_stop_patience_param, verbose=1,
                                                 restore_best_weights=True)
        rlrop_cb = tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=reduce_lr_factor_param,
                                                        patience=reduce_lr_patience_param, verbose=1,
                                                        min_lr=min_lr_plateau_param)
        callbacks_list.extend([mcp_cb, es_cb, rlrop_cb])
        print(f"Лучшая модель будет сохраняться в: {checkpoint_filepath_best_abs_cb_path} (по val_loss)")
    else:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен. Сохранение лучшей модели и EarlyStopping/ReduceLROnPlateau по val_loss отключены.")

    print(f"\nЗапуск обучения на {epochs_param} эпох (initial_epoch = {initial_epoch_to_start_fit})...")
    print(f"  Логи TensorBoard: {log_dir}")

    try:
        history = model.fit(
            train_dataset, epochs=epochs_param, validation_data=validation_dataset,
            callbacks=callbacks_list, verbose=1, initial_epoch=initial_epoch_to_start_fit
        )
        print(f"\n--- Обучение детектора (режим: {training_mode_description_var}) завершено ---")

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
        print(f"ОШИБКА model.fit(): {e_fit}"); import traceback; traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()