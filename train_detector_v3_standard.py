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
# Определяем корень проекта, поднимаясь вверх по дереву директорий, пока не найдем маркер (например, папку 'src')
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path
# Поднимаемся на один уровень выше, пока не найдем папку 'src' или не достигнем корня файловой системы
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Добавляем корень проекта в sys.path, если его там нет
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT)) # Вставляем в начало для приоритета
    _added_to_sys_path = True
else:
    _added_to_sys_path = False


# --- Настройка логирования ---
logger = logging.getLogger(__name__)
# Установим уровень только если он не был установлен ранее
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Отключим логирование из Albumentations если оно мешает
logging.getLogger('albumentations').setLevel(logging.WARNING)


# --- Импорт наших модулей ---
try:
    # Импортируем функцию создания датасета из нашего data_loader
    # Теперь импорты должны работать, т.к. корень проекта в sys.path
    from src.datasets.data_loader_v3_standard import create_dataset, generate_all_anchors, DataGenerator # Импортируем DataGenerator для получения размера
    # Импортируем функцию построения модели
    from src.models.detector_v3_standard import build_detector_v3_standard
    # Импортируем нашу функцию потерь
    from src.losses.detection_losses_v3_standard import DetectorLoss
    # Импортируем plot_utils для потенциальной визуализации (опционально в коллбэке)
    from src.utils import plot_utils # Хотя напрямую здесь не используется, может быть нужен коллбэкам
    # Импортируем postprocessing для оценки или кастомных коллбэков
    # from src.utils.postprocessing import decode_predictions, perform_nms # Эти нужны будут в evaluate/predict скриптах

except ImportError as e:
    logger.error(f"Ошибка импорта модулей проекта: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


# --- Установка глобальных сидов для воспроизводимости ---
def set_global_seed(seed):
    """Устанавливает сид для random, numpy и tensorflow."""
    if seed is not None:
        logger.info(f"Установка глобальных сидов на: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # tf.keras.utils.set_random_seed(seed) # Альтернативный способ для Keras >= 2.6
        # Albumentations использует сид numpy.random
    else:
        logger.info("Глобальный сид НЕ установлен (используются случайные сиды).")


# --- Кастомный планировщик Learning Rate с Warmup и Cosine Decay ---
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, warmup_epochs, name='WarmupCosineDecay'):
        """
        Инициализирует планировщик LR с Warmup и Cosine Decay для ОДНОЙ фазы обучения.

        Args:
            initial_lr (float): Максимальная скорость обучения, которая будет достигнута после warmup.
            total_epochs (int): Общее количество эпох в ТЕКУЩЕЙ фазе обучения.
            warmup_epochs (int): Количество эпох для линейного "разогрева" LR.
            name (str): Имя коллбэка.
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs_in_phase = total_epochs
        self.warmup_epochs = warmup_epochs
        self.name = name

        # Эти значения будут установлены в on_train_begin, когда будет доступен model.optimizer и self.params
        self.steps_per_epoch = 0
        self.total_steps_in_phase = 0
        self.warmup_steps = 0
        # [ИЗМЕНЕНИЕ] Локальный счетчик шагов
        self._step = 0


    def on_train_begin(self, logs=None):
        """
        Фиксирует "ноль" итераций и точное количество шагов в эпоху в момент старта фазы обучения.
        Вызывается Keras в начале model.fit().
        """
        if self.model.optimizer is None:
            logger.error("Оптимизатор не установлен в модели! LR Scheduler не будет работать.")
            return

            # [ИСПРАВЛЕНО] Получаем точное количество шагов в эпоху из Keras
            # с безопасным fallback'ом
        if self.params.get("steps"):
            self.steps_per_epoch = self.params['steps']
            logger.info(f"LR Scheduler '{self.name}': Количество шагов в эпоху уточнено Keras: {self.steps_per_epoch}")
        else:
            logger.warning(
                f"LR Scheduler '{self.name}': Не удалось получить точное количество шагов в эпоху от Keras. Попытка расчета из Dataset.")
            # Безопасная оценка размера датасета, избегая приватных полей
            try:
                # Попробуем получить dataset через model.train_dataset (TF 2.10+)
                card = tf.data.experimental.cardinality(self.model.train_dataset)
                if card != tf.data.INFINITE_CARDINALITY and card.numpy() > 0:
                    self.steps_per_epoch = int(card.numpy())
                    logger.info(
                        f"LR Scheduler '{self.name}': Количество шагов в эпоху рассчитано из Dataset: {self.steps_per_epoch}")
                else:
                    raise ValueError("Dataset cardinality is 0 or unknown.")

            except (AttributeError, ValueError):
                logger.warning("Не удалось определить размер датасета через train_dataset, используем fallback.")
                self.steps_per_epoch = 1  # Безопасный fallback

            # Пересчитываем общее количество шагов и шагов для warmup на основе точного steps_per_epoch
        self.total_steps_in_phase = self.total_epochs_in_phase * self.steps_per_epoch
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch

        # Инициализируем локальный счетчик шагов
        self._step = 0

        logger.info(
            f"LR Scheduler '{self.name}': Начинаем фазу. LR = {self.model.optimizer.learning_rate.numpy():.7f}, Total Steps = {self.total_steps_in_phase}")



    def on_train_batch_begin(self, batch, logs=None):

        current_step_in_phase = self._step + 1
        # <<<----------- КОНЕЦ БЛОКА НА ВСТАВКУ ------------->


        # Логика расчета LR (остается без изменений)
        if current_step_in_phase < self.warmup_steps:
            # Warmup phase: линейное увеличение LR
            target_lr = self.initial_lr
            warmup_steps = self.warmup_steps
            current_lr = target_lr * (current_step_in_phase / warmup_steps) if warmup_steps > 0 else target_lr
        else:
            # Cosine Decay phase
            steps_after_warmup = current_step_in_phase - self.warmup_steps
            total_decay_steps = self.total_steps_in_phase - self.warmup_steps

            if total_decay_steps > 0:
                 cosine_decay_factor = 0.5 * (1 + math.cos(math.pi * steps_after_warmup / total_decay_steps))
                 current_lr = self.initial_lr * cosine_decay_factor
            else:
                 current_lr = self.initial_lr


        current_lr = max(current_lr, 0.0)
        # [ИСПРАВЛЕНО] Проверяем тип, прежде чем вызывать .assign()
        if isinstance(self.model.optimizer.learning_rate, tf.Variable):
             self.model.optimizer.learning_rate.assign(tf.cast(current_lr, dtype=tf.float32))
        else:
             # Для LR Schedulers
             self.model.optimizer.learning_rate = tf.cast(current_lr, dtype=tf.float32)

        # Увеличиваем локальный счетчик
        self._step += 1


    def on_epoch_end(self, epoch, logs=None):
        """
        Логирует LR в конце каждой эпохи.
        """
        # Логируем LR в конце каждой эпохи
        current_lr = self.model.optimizer.learning_rate
        if isinstance(current_lr, tf.Variable):
             current_lr = current_lr.numpy()
        # Логируем с учетом номера эпохи в ФАЗЕ + начальная эпоха
        initial_epoch_in_fit = self.params.get('initial_epoch', 0)
        epoch_in_phase = epoch - initial_epoch_in_fit

        if logs is not None:
             logs['learning_rate'] = current_lr # Логируем в TensorBoard
        logger.info(f"LR Scheduler '{self.name}': Эпоха {epoch + 1} (в фазе: {epoch_in_phase+1}/{self.total_epochs_in_phase}), LR = {current_lr:.7f}")


# --- Вспомогательная функция для вычисления шагов в эпоху ---
def calculate_steps(dataset_size, batch_size):
    """Вычисляет количество шагов в эпоху, округляя вниз."""
    if dataset_size == 0 or batch_size == 0:
        return 0
    # Используем len() экземпляра DataGenerator для получения размера датасета
    # [ИЗМЕНЕНИЕ] Округляем вниз (floor) для консервативной оценки
    return math.floor(dataset_size / batch_size)


# === [НОВАЯ ФУНКЦИЯ] Вспомогательная функция для надежной заморозки Backbone ===
def _freeze_backbone(model, prefix=('block', 'stem')):
    """
    Замораживает слои в модели, чьи имена начинаются с одного из заданных префиксов,
    и также замораживает все слои BatchNormalization в модели.

    Args:
        model (tf.keras.Model): Модель для заморозки.
        prefix (tuple or str): Префикс или кортеж префиксов для замораживаемых слоев.
    """
    if isinstance(prefix, str):
        prefix = (prefix,) # Преобразуем в кортеж, если передана строка

    frozen, bn_frozen = 0, 0
    # [ИСПРАВЛЕНО] Убран рекурсивный проход. Keras model.layers уже дает все слои.
    for layer in model.layers:
        # Заморозка слоев Backbone по префиксу (напр., 'block' для EfficientNet)
        if layer.name.startswith(prefix):
            layer.trainable = False; frozen += 1
        # [ВАЖНО] Замораживаем ВСЕ BN-слои, включая те, что в головах, для Фазы 1
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False; bn_frozen += 1

    logger.info(f"Backbone freeze: {frozen} слоев с префиксами {prefix} заморожены. {bn_frozen} BN слоев заморожены.")



# --- Основная функция тренировки ---
# [ИЗМЕНЕНИЕ] train_detector - убраны steps_per_epoch, добавлена надежная заморозка BN, переработаны коллбэки
def train_detector(main_config_path, predict_config_path=None, run_seed=None):
    """Обучает детектор в две фазы."""
    logger.info("--- Запуск обучения детектора ---")

    # 1. Загрузка конфигов
    # ... (остается без изменений) ...
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
    # ... (остается без изменений) ...
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_base_dir = PROJECT_ROOT / main_config.get('logs_base_dir', 'logs')
    weights_base_dir = PROJECT_ROOT / main_config.get('weights_base_dir', 'weights')
    log_dir_run = log_base_dir / main_config['log_dir'] / timestamp # Папка для логов конкретного запуска
    saved_model_dir_base = weights_base_dir / main_config['saved_model_dir'] # Базовая папка для сохранения модели

    log_dir_run.mkdir(parents=True, exist_ok=True)
    saved_model_dir_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Логи будут сохраняться в: {log_dir_run}")
    logger.info(f"Веса модели будут сохраняться в: {saved_model_dir_base}")

    # 4. Создание датасетов
    batch_size = main_config.get('batch_size', 8)
    use_augmentation_train = main_config.get('use_augmentation', True) # По умолчанию используем аугментацию для тренировки

    logger.info(f"Создание тренировочного датасета (batch_size={batch_size}, augmentation={use_augmentation_train})...")

    # [ИСПРАВЛЕНО] Берем fpn_strides из конфига
    fpn_strides_from_config = main_config.get('fpn_strides', [8, 16, 32])

    # Генерируем all_anchors один раз для датасетов и функции потерь
    all_anchors_for_dataset_and_loss = generate_all_anchors(
        main_config['input_shape'],
        fpn_strides_from_config, # Используем strides из конфига
        main_config['anchor_scales'],
        main_config['anchor_ratios']
    )

    # Создаем tf.data.Dataset для тренировки
    train_dataset = create_dataset(
        main_config,
        is_training=True,
        batch_size=batch_size,
        debug_mode=False
    )

    logger.info(f"Создание валидационного датасета (batch_size={batch_size}, augmentation=False)...")
    val_dataset = create_dataset(
        main_config,
        is_training=False,
        batch_size=batch_size,
        debug_mode=False
    )

    # 5. Создание модели
    # ... (остается без изменений) ...
    logger.info("Создание модели...")
    model = build_detector_v3_standard(main_config)
    logger.info("Модель создана.")
    model.summary(print_fn=lambda x: logger.info(x))

    # 6. Создание функции потерь
    # ... (остается без изменений) ...
    loss_fn = DetectorLoss(main_config, all_anchors=all_anchors_for_dataset_and_loss)
    logger.info("Функция потерь DetectorLoss создана.")


    # 7. Фаза 1: Обучение с замороженным Backbone
    logger.info("--- Фаза 1: Обучение с замороженным Backbone ---")

    # [ИСПРАВЛЕНО] Используем вспомогательную функцию для надежной заморозки
    logger.info("--- Фаза 1: Обучение с замороженным Backbone ---")

    if main_config.get('freeze_backbone', True):
        _freeze_backbone(model, prefix=('block', 'stem'))
    else:
        logger.info("Backbone не замораживается для Фазы 1 (freeze_backbone=False в конфиге).")

    # Компиляция для Фазы 1
    optimizer_phase1 = tfa.optimizers.AdamW(
        learning_rate=main_config['initial_learning_rate'],
        weight_decay=main_config.get('weight_decay', 1e-4),
        clipnorm=main_config.get('clipnorm', 1.0)
    )

    # [ИСПРАВЛЕНО] Переименовываем loss_fn в loss
    model.compile(optimizer=optimizer_phase1, loss=loss_fn)  # Используем loss вместо loss_fn
    logger.info("Модель скомпилирована для Фазы 1.")

    # 8. Настройка Callbacks для Фазы 1
    logger.info("Настройка коллбэков для Фазы 1...")

    # [ИСПРАВЛЕНО] Создаем НОВЫЙ объект ModelCheckpoint для Фазы 1
    checkpoint_filename_h5_phase1 = main_config.get('best_model_filename', 'best_model.weights.h5')
    if not str(checkpoint_filename_h5_phase1).lower().endswith(('.h5', '.weights.h5')):
         checkpoint_filename_h5_phase1 += '.weights.h5'
    checkpoint_filepath_phase1 = os.path.join(str(saved_model_dir_base), checkpoint_filename_h5_phase1)
    logger.info(f"Путь для сохранения чекпоинта Фазы 1: {checkpoint_filepath_phase1}")

    model_checkpoint_callback_phase1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_phase1,
        save_weights_only=True,
        monitor='val_loss', # val_loss для валидации
        mode='min',
        save_best_only=True, # Сохраняем только лучшую
        verbose=1
    )

    # [ИСПРАВЛЕНО] Создаем НОВЫЙ объект EarlyStopping для Фазы 1
    early_stopping_callback_phase1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=main_config.get('early_stopping_patience', 20), # Количество эпох без улучшения
        min_delta=main_config.get('early_stopping_min_delta', 0.0001),
        mode='min',
        verbose=1,
        restore_best_weights=True # Важно!
    )

    # LR Scheduler для Фазы 1
    # [ИЗМЕНЕНИЕ] LR Scheduler теперь не требует steps_per_epoch в конструкторе
    lr_scheduler_phase1 = WarmupCosineDecay(
        initial_lr=main_config['initial_learning_rate'],
        total_epochs=main_config['epochs_phase1'], # Длительность Фазы 1
        warmup_epochs=main_config.get('warmup_epochs', 0), # Warmup из конфига
        name='LR_Phase1'
    )

    # [ИСПРАВЛЕНО] Создаем новый TensorBoard callback для Фазы 1
    tensorboard_callback_phase1 = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir_run / 'phase1'),
        histogram_freq=main_config.get('tensorboard_histogram_freq', 1),
        update_freq='epoch'
    )


    callbacks_phase1 = [
        tensorboard_callback_phase1,
        model_checkpoint_callback_phase1,
        early_stopping_callback_phase1,
        lr_scheduler_phase1,
    ]

    # 9. Обучение Фазы 1
    logger.info(f"Начинаем Фазу 1 обучения на {main_config['epochs_phase1']} эпохах...")
    # [ИЗМЕНЕНО] НЕ передаем steps_per_epoch и validation_steps в model.fit
    history1 = model.fit(
        train_dataset,
        epochs=main_config['epochs_phase1'],
        # steps_per_epoch=None, # Убрано
        validation_data=val_dataset, # validation_steps не нужен, если val_dataset не зациклен
        callbacks=callbacks_phase1,
        verbose=1 # Логгировать прогресс
    )
    logger.info("Фаза 1 обучения завершена.")

    # 10. Фаза 2: Fine-tuning
    logger.info("--- Фаза 2: Fine-tuning с размороженным Backbone ---")

    # Загружаем лучшие веса с Фазы 1
    if Path(checkpoint_filepath_phase1).exists():
        logger.info(f"Загрузка лучших весов с Фазы 1 из: {checkpoint_filepath_phase1}")
        model.load_weights(checkpoint_filepath_phase1)
    else:
        logger.warning(
            f"Чекпоинт Фазы 1 не найден: {checkpoint_filepath_phase1}. Продолжаем с весами последней эпохи Фазы 1.")

    for layer in model.layers:
        layer.trainable = True

        # Затем, настраиваем BN слои, как рекомендовано
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.97
            layer.epsilon = 1e-3
            # layer.renorm = True # renorm может быть нестабилен, начнем без него
        if hasattr(layer, 'layers'):  # Рекурсивно проходим по вложенным моделям (например, Backbone)
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    sub_layer.momentum = 0.97
                    sub_layer.epsilon = 1e-3

    logger.info("Все слои модели и BN разморожены для Фазы 2. BN momentum=0.97, epsilon=1e-3.")


    # Компиляция для Фазы 2
    optimizer_phase2 = tfa.optimizers.AdamW(
        learning_rate=main_config['fine_tune_learning_rate'],
        weight_decay=main_config.get('weight_decay', 1e-4),
        clipnorm=main_config.get('clipnorm', 1.0)
    )
    # [ИСПРАВЛЕНО] Сбрасываем счетчик итераций оптимизатора
    logger.info("Сброс счетчика итераций оптимизатора для Фазы 2...")
    optimizer_phase2.iterations.assign(0)

    # [ИСПРАВЛЕНО] Переименовываем loss_fn в loss
    model.compile(optimizer=optimizer_phase2, loss=loss_fn) # Используем ту же функцию потерь
    logger.info("Модель скомпилирована для Фазы 2 с более низким LR.")

    # 11. Настройка Callbacks для Фазы 2
    logger.info("Настройка коллбэков для Фазы 2...")

    # [ИСПРАВЛЕНО] Создаем НОВЫЙ объект ModelCheckpoint для Фазы 2
    # Используем то же имя файла, чтобы лучшие веса Фазы 2 перезаписали веса Фазы 1, если они лучше
    model_checkpoint_callback_phase2 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_phase1, # Перезаписываем тот же файл
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # [ИСПРАВЛЕНО] Создаем НОВЫЙ объект EarlyStopping для Фазы 2
    early_stopping_callback_phase2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=main_config.get('early_stopping_patience', 20),
        min_delta=main_config.get('early_stopping_min_delta', 0.0001),
        mode='min',
        verbose=1,
        restore_best_weights=True # Восстановить веса лучшей эпохи при остановке
    )

    # LR Scheduler для Фазы 2
    lr_scheduler_phase2 = WarmupCosineDecay(
        initial_lr=main_config['fine_tune_learning_rate'],
        total_epochs=main_config['epochs_phase2'],
        warmup_epochs=main_config.get('warmup_epochs_phase2', 0),
        name='LR_Phase2'
    )

    # [ИСПРАВЛЕНО] Создаем новый TensorBoard callback для Фазы 2 с отдельной папкой
    tensorboard_callback_phase2 = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir_run / 'phase2'),
        histogram_freq=main_config.get('tensorboard_histogram_freq', 1),
        update_freq='epoch'
    )

    callbacks_phase2 = [
        tensorboard_callback_phase2,
        model_checkpoint_callback_phase2,
        early_stopping_callback_phase2,
        lr_scheduler_phase2,
    ]


    # 12. Обучение Фазы 2
    logger.info(f"Начинаем Фазу 2 обучения на {main_config['epochs_phase2']} эпохах...")
    # [ИЗМЕНЕНО] НЕ передаем steps_per_epoch и validation_steps в model.fit
    history2 = model.fit(
        train_dataset,
        epochs=main_config['epochs_phase1'] + main_config['epochs_phase2'],
        initial_epoch=main_config['epochs_phase1'],
        validation_data=val_dataset,
        callbacks=callbacks_phase2,
        verbose=1
    )
    logger.info("Фаза 2 обучения завершена.")
    logger.info("--- Обучение детектора завершено ---")



# --- Точка входа скрипта ---
if __name__ == '__main__':
    # Настройка логирования в консоль для запуска как main скрипт
    # Проверяем, не настроено ли уже (например, при запуске из другого скрипта)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__) # Получим логгер еще раз после настройки

    logger.info("Запуск train_detector_v3_standard.py как основного скрипта.")

    # --- Пути к конфигам ---
    # Предполагаем, что конфиги находятся в src/configs относительно PROJECT_ROOT
    MAIN_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
    PREDICT_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "predict_config.yaml" # predict_config нужен для evaluation/postprocessing params

    # --- Параметры запуска тренировки ---
    # Установить сид для воспроизводимости запуска (влияет на инициализацию весов, перемешивание датасета и аугментацию)
    # None для случайного запуска
    RUN_SEED = 42

    # Запускаем тренировку
    train_detector(
        main_config_path=MAIN_CONFIG_PATH,
        predict_config_path=PREDICT_CONFIG_PATH,
        run_seed=RUN_SEED
    )