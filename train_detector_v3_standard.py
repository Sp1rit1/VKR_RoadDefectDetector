# RoadDefectDetector/train_detector_v3_standard.py

import os
import sys
import yaml
import logging
import math
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random # Для установки сида random



# --- Настройка путей для импорта ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('albumentations').setLevel(logging.WARNING)



# Дополнительная проверка, которую использует Keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


try:
    from src.datasets.data_loader_v3_standard import create_dataset, generate_all_anchors, DataGenerator # Импортируем DataGenerator
    from src.models.detector_v3_standard import build_detector_v3_standard
    from src.losses.detection_losses_v3_standard import DetectorLoss
except ImportError as e:
    logger.error(f"Ошибка импорта модулей проекта: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)

def set_global_seed(seed):
    if seed is not None:
        logger.info(f"Установка глобальных сидов на: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    else:
        logger.info("Глобальный сид НЕ установлен.")

# --- Кастомный планировщик Learning Rate с Warmup и Cosine Decay ---
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, steps_per_epoch, warmup_epochs, name='WarmupCosineDecay'):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs_in_phase = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.name = name
        self.total_steps_in_phase = self.total_epochs_in_phase * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch

        if self.warmup_steps >= self.total_steps_in_phase and self.total_steps_in_phase > 0:
             logger.warning(f"Warmup steps ({self.warmup_steps}) >= Total steps in phase ({self.total_steps_in_phase}). Warmup будет длиться всю фазу.")

    def on_train_begin(self, logs=None):
        if self.model.optimizer is None:
             logger.error("Оптимизатор не установлен в модели! LR Scheduler не будет работать.")
             return
        # ИСПРАВЛЕНО: Используем прямое присваивание, если learning_rate - это Variable
        # tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_lr) # Старый способ
        self.model.optimizer.learning_rate.assign(self.initial_lr) # Новый, более надежный способ для tf.Variable
        logger.info(f"LR Scheduler '{self.name}': Начинаем фазу с LR = {self.model.optimizer.learning_rate.numpy():.7f}")

    def on_train_batch_begin(self, batch, logs=None):
        current_step_tf = self.model.optimizer.iterations
        current_step_in_phase = tf.cast(current_step_tf, tf.float32)

        if current_step_in_phase < self.warmup_steps:
            current_lr = self.initial_lr * (current_step_in_phase / self.warmup_steps) if self.warmup_steps > 0 else self.initial_lr
        else:
            steps_after_warmup = current_step_in_phase - self.warmup_steps
            total_decay_steps = self.total_steps_in_phase - self.warmup_steps
            if total_decay_steps > 0:
                 cosine_decay_factor = 0.5 * (1 + tf.cos(math.pi * steps_after_warmup / total_decay_steps))
                 current_lr = self.initial_lr * cosine_decay_factor
            else:
                 current_lr = self.initial_lr
        current_lr = tf.maximum(current_lr, 0.0)
        # ИСПРАВЛЕНО: Используем прямое присваивание
        # tf.keras.backend.set_value(self.model.optimizer.learning_rate, current_lr) # Старый способ
        self.model.optimizer.learning_rate.assign(current_lr) # Новый способ

    def on_epoch_end(self, epoch, logs=None):
        # ИСПРАВЛЕНО: Получаем значение через .numpy()
        current_lr = self.model.optimizer.learning_rate.numpy()
        if logs is not None:
             logs['learning_rate'] = current_lr
        logger.info(f"LR Scheduler '{self.name}': LR на конце эпохи {epoch + (self.params.get('initial_epoch', 0)) + 1} = {current_lr:.7f}")

def calculate_steps(dataset_size, batch_size):
    if dataset_size == 0 or batch_size == 0: return 0
    return math.ceil(dataset_size / batch_size)

# --- Основная функция тренировки ---
def train_detector(main_config_path, predict_config_path, run_seed=None):
    """Обучает детектор в две фазы."""
    logger.info("--- Запуск обучения детектора ---")

    # 1. Загрузка конфигов
    try:
        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        predict_config = None
        if predict_config_path and Path(predict_config_path).exists():
             with open(predict_config_path, 'r', encoding='utf-8') as f:
                 predict_config = yaml.safe_load(f)
             logger.info("Конфигурационные файлы успешно загружены.")
        else:
             logger.warning(f"predict_config.yaml не найден по пути: {predict_config_path} или путь не передан.")
             if main_config: logger.info("Основной конфигурационный файл успешно загружен.")
             else: logger.error("Ошибка: Не удалось загрузить основной конфигурационный файл."); sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Ошибка: Конфигурационный файл не найден - {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка загрузки или парсинга конфигов: {e}")
        sys.exit(1)

    # 2. Установка глобальных сидов для всей тренировки
    set_global_seed(run_seed)

    # 3. Подготовка путей для логов и весов
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_base_dir = PROJECT_ROOT / main_config.get('logs_base_dir', 'logs')
    weights_base_dir = PROJECT_ROOT / main_config.get('weights_base_dir', 'weights')
    log_dir_run = log_base_dir / main_config['log_dir'] / timestamp
    saved_model_dir = weights_base_dir / main_config['saved_model_dir']
    log_dir_run.mkdir(parents=True, exist_ok=True)
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Логи будут сохраняться в: {log_dir_run}")
    logger.info(f"Веса модели будут сохраняться в: {saved_model_dir}")

    # 4. Создание датасетов
    batch_size = main_config.get('batch_size', 8)
    use_augmentation_train = main_config.get('use_augmentation', True)

    logger.info(f"Создание тренировочного датасета (batch_size={batch_size}, augmentation={use_augmentation_train})...")

    # Генерируем all_anchors один раз
    all_anchors_for_generator = generate_all_anchors(
        main_config['input_shape'], [8, 16, 32],
        main_config['anchor_scales'], main_config['anchor_ratios']
    )

    # Создаем экземпляр DataGenerator для получения размера
    train_generator_instance = DataGenerator( # Используем импортированный DataGenerator
        config=main_config,
        all_anchors=all_anchors_for_generator, # Передаем якоря
        is_training=True,
        debug_mode=False
    )
    train_dataset_size = len(train_generator_instance)
    del train_generator_instance

    # Теперь создаем tf.data.Dataset
    train_dataset = create_dataset(
        main_config,
        is_training=True,
        batch_size=batch_size,
        debug_mode=False
        # shuffle_seed и aug_seed не передаются, если create_dataset их не принимает
    )

    logger.info(f"Тренировочный датасет: {train_dataset_size} изображений, batch_size={batch_size}.")
    steps_per_epoch = calculate_steps(train_dataset_size, batch_size)
    logger.info(f"Шагов в эпоху тренировки: {steps_per_epoch}")

    logger.info(f"Создание валидационного датасета (batch_size={batch_size}, augmentation=False)...")
    val_generator_instance = DataGenerator(
        config=main_config,
        all_anchors=all_anchors_for_generator,
        is_training=False,
        debug_mode=False
    )
    val_dataset_size = len(val_generator_instance)
    del val_generator_instance

    val_dataset = create_dataset(
        main_config,
        is_training=False,
        batch_size=batch_size,
        debug_mode=False
    )

    logger.info(f"Валидационный датасет: {val_dataset_size} изображений, batch_size={batch_size}.")
    validation_steps = calculate_steps(val_dataset_size, batch_size)
    logger.info(f"Шагов в эпоху валидации: {validation_steps}")

    if steps_per_epoch == 0: logger.error("Тренировочный датасет пустой!"); sys.exit(1)
    if validation_steps == 0: logger.warning("Валидационный датасет пустой!")

    # 5. Создание модели
    logger.info("Создание модели...")
    model = build_detector_v3_standard(main_config)
    logger.info("Модель создана.")
    model.summary(print_fn=lambda x: logger.info(x))

    # 6. Настройка Callbacks
    logger.info("Настройка коллбэков...")
    # ИСПРАВЛЕНО: Имя файла для ModelCheckpoint
    base_filename_for_checkpoint = Path(main_config['best_model_filename']).stem
    checkpoint_filename_h5 = f"{base_filename_for_checkpoint}.weights.h5" # Используем .weights.h5
    checkpoint_filepath = os.path.join(str(saved_model_dir), checkpoint_filename_h5)
    logger.info(f"Путь для сохранения чекпоинта: {checkpoint_filepath}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir_run), histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        # [ИЗМЕНЕНИЕ] Добавляем EarlyStopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            # Используем .get() для безопасности, если параметра вдруг нет
            patience=main_config.get('early_stopping_patience', 20),
            min_delta=main_config.get('early_stopping_min_delta', 0.0001),
            mode='min',
            verbose=1,
            restore_best_weights=True  # Это самый важный параметр!
        )
    ]

    # 7. Фаза 1: Обучение с замороженным Backbone
    logger.info("--- Фаза 1: Обучение с замороженным Backbone ---")
    optimizer_phase1 = tfa.optimizers.AdamW(
        learning_rate=main_config['initial_learning_rate'],
        weight_decay=main_config.get('weight_decay', 1e-4)
    )

    all_anchors = generate_all_anchors(
        main_config['input_shape'], [8, 16, 32],
        main_config['anchor_scales'], main_config['anchor_ratios']
    )

    loss_fn = DetectorLoss(main_config, all_anchors=all_anchors)

    model.compile(optimizer=optimizer_phase1, loss_fn=loss_fn)

    logger.info("Модель скомпилирована для Фазы 1.")

    logger.info("Замораживаем ВСЕ слои BatchNormalization в модели...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        # Дополнительно проходим по вложенным моделям (backbone, heads)
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    sub_layer.trainable = False

    lr_scheduler_phase1 = WarmupCosineDecay(
        initial_lr=main_config['initial_learning_rate'], total_epochs=main_config['epochs_phase1'],
        steps_per_epoch=steps_per_epoch, warmup_epochs=main_config.get('warmup_epochs', 0), name='LR_Phase1'
    )
    callbacks_phase1 = callbacks + [lr_scheduler_phase1]
    logger.info(f"Начинаем Фазу 1 обучения на {main_config['epochs_phase1']} эпохах...")
    model.fit(
        train_dataset, epochs=main_config['epochs_phase1'], steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset if validation_steps > 0 else None,
        validation_steps=validation_steps if validation_steps > 0 else None,
        callbacks=callbacks_phase1, verbose=1
    )
    logger.info("Фаза 1 обучения завершена.")

    # 8. Фаза 2: Fine-tuning
    logger.info("--- Фаза 2: Fine-tuning с размороженным Backbone ---")
    if Path(checkpoint_filepath).exists():
        logger.info(f"Загрузка лучших весов с Фазы 1 из: {checkpoint_filepath}")
        model.load_weights(checkpoint_filepath)
    else:
        logger.warning(f"Чекпоинт Фазы 1 не найден: {checkpoint_filepath}. Продолжаем с текущими весами.")

    model.trainable = True
    backbone_layer_name = main_config.get('backbone_layer_name', 'efficientnetb0')
    try:
        model.get_layer(backbone_layer_name).trainable = True
        logger.info(f"Backbone '{backbone_layer_name}' разморожен для Фазы 2.")
    except ValueError:
        logger.warning(f"Backbone '{backbone_layer_name}' не найден. Вся модель уже trainable.")

    optimizer_phase2 = tfa.optimizers.AdamW(
        learning_rate=main_config['fine_tune_learning_rate'],
        weight_decay=main_config.get('weight_decay', 1e-4)
    )

    loss_fn = DetectorLoss(main_config, all_anchors=all_anchors)

    model.compile(optimizer=optimizer_phase2, loss_fn=loss_fn)

    logger.info("Модель скомпилирована для Фазы 2 с новым LR.")
    lr_scheduler_phase2 = WarmupCosineDecay(
        initial_lr=main_config['fine_tune_learning_rate'], total_epochs=main_config['epochs_phase2'],
        steps_per_epoch=steps_per_epoch, warmup_epochs=main_config.get('warmup_epochs_phase2', 0), name='LR_Phase2'
    )
    callbacks_phase2 = callbacks + [lr_scheduler_phase2]
    logger.info(f"Начинаем Фазу 2 обучения на {main_config['epochs_phase2']} эпохах...")

    logger.info("Сброс итераций оптимизатора для корректной работы LR-шедулера...")
    optimizer_phase2.iterations.assign(0)

    model.fit(
        train_dataset, epochs=main_config['epochs_phase1'] + main_config['epochs_phase2'],
        initial_epoch=main_config['epochs_phase1'],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset if validation_steps > 0 else None,
        validation_steps=validation_steps if validation_steps > 0 else None,
        callbacks=callbacks_phase2, verbose=1
    )
    logger.info("Фаза 2 обучения завершена.")
    logger.info("--- Обучение детектора завершено ---")

    final_weights_path = os.path.join(str(saved_model_dir), 'final_model_weights.keras') # Можно оставить .keras для финальных весов
    try: model.save_weights(final_weights_path); logger.info(f"Финальные веса сохранены: {final_weights_path}")
    except Exception as e: logger.error(f"Ошибка сохранения финальных весов: {e}")


# --- Точка входа скрипта ---
if __name__ == '__main__':
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    logger.info("Запуск train_detector_v3_standard.py как основного скрипта.")
    MAIN_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
    PREDICT_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "predict_config.yaml"
    RUN_SEED = 42
    train_detector(main_config_path=MAIN_CONFIG_PATH, predict_config_path=PREDICT_CONFIG_PATH, run_seed=RUN_SEED)