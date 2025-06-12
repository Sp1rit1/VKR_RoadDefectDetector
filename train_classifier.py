# RoadDefectDetector/train_classifier.py
import tensorflow as tf
import yaml
import os
import datetime

# --- Импорты из папки src ---
# Предполагается, что этот скрипт (train_classifier.py) находится в КОРНЕВОЙ папке проекта RoadDefectDetector/
from src.datasets.classifier_data_loader import create_classifier_datasets
from src.models.image_classifier import build_classifier_model

# --- Загрузка Конфигураций ---
_current_project_root = os.path.dirname(os.path.abspath(__file__))  # Корень проекта

_base_config_path = os.path.join(_current_project_root, 'src', 'configs', 'base_config.yaml')
_classifier_config_path = os.path.join(_current_project_root, 'src', 'configs', 'classifier_config.yaml')

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    with open(_classifier_config_path, 'r', encoding='utf-8') as f:
        CLASSIFIER_CONFIG = yaml.safe_load(f)
except FileNotFoundError as e:
    print(f"ОШИБКА: Не найден один из файлов конфигурации.")
    print(f"Проверьте пути: \n{_base_config_path}\n{_classifier_config_path}")
    print(f"Детали ошибки: {e}")
    exit()
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл конфигурации.")
    print(f"Детали ошибки: {e}")
    exit()


# --- Основная функция обучения ---
def train():
    print("--- Обучение классификатора Дорога/Не дорога (Режим без валидации / Тестовый режим) ---")

    # 1. Получение датасетов
    print("Загрузка и подготовка датасетов для классификатора...")
    # create_classifier_datasets вернет (train_ds, val_ds, class_names)
    # val_ds может быть None или пустым, если prepare_classifier_dataset.py не создал валидационные данные
    train_ds, val_ds, class_names = create_classifier_datasets()

    if train_ds is None:
        print("Не удалось загрузить обучающий датасет для классификатора. Обучение прервано.")
        print("Убедитесь, что скрипт 'prepare_classifier_dataset.py' был успешно запущен и создал данные.")
        return

    if val_ds is None:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет не найден или пуст. Обучение будет проходить без валидации.")
        print("Это нормально для первоначального теста на очень малом количестве данных,")
        print("но для реального обучения необходима валидационная выборка.")

    print(f"Классы, найденные загрузчиком данных: {class_names}")
    if CLASSIFIER_CONFIG['num_classes'] != len(class_names):
        print(f"ОШИБКА: 'num_classes' в classifier_config.yaml ({CLASSIFIER_CONFIG['num_classes']}) "
              f"не соответствует количеству найденных классов ({len(class_names)}: {class_names}).")
        print("Пожалуйста, исправьте 'num_classes' в 'src/configs/classifier_config.yaml'.")
        return

    # 2. Создание модели
    print("\nСоздание модели классификатора...")
    model = build_classifier_model()
    print("\nСтруктура модели классификатора:")
    model.summary()

    # 3. Компиляция модели
    print("\nКомпиляция модели...")
    num_model_outputs = model.output_shape[-1]
    if num_model_outputs == 1:
        loss_function = 'binary_crossentropy'
        print("Используется loss: binary_crossentropy (ожидается один выходной нейрон с sigmoid)")
    elif num_model_outputs == CLASSIFIER_CONFIG['num_classes'] and CLASSIFIER_CONFIG['num_classes'] > 1:
        loss_function = 'sparse_categorical_crossentropy'
        print(
            f"Используется loss: sparse_categorical_crossentropy (ожидается {num_model_outputs} выходных нейронов с softmax)")
    else:
        print(
            f"ОШИБКА: Несоответствие 'num_classes' ({CLASSIFIER_CONFIG['num_classes']}) и формы выхода модели ({model.output_shape}).")
        return

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CLASSIFIER_CONFIG['train_params']['learning_rate']),
                  loss=loss_function,
                  metrics=['accuracy'])

    # 4. Настройка Callbacks
    logs_dir_abs = os.path.join(_current_project_root, BASE_CONFIG.get('logs_base_dir', 'logs'),
                                "classifier_fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir_abs, histogram_freq=1)

    weights_dir_abs = os.path.join(_current_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))
    os.makedirs(weights_dir_abs, exist_ok=True)
    # Используем формат .keras для сохранения всей модели
    # Изменим имя файла, чтобы не перезаписать "лучшую" модель, если валидации нет
    checkpoint_filename = 'classifier_trained_model.keras'
    checkpoint_filepath = os.path.join(weights_dir_abs, checkpoint_filename)

    callbacks_list = [tensorboard_callback]

    if val_ds:
        print("Валидационный датасет доступен. Используем ModelCheckpoint и EarlyStopping с val_accuracy/val_loss.")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath.replace(".keras", "_best.keras"),  # Отдельное имя для лучшей по валидации
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True)  # Меньшее терпение для теста
        callbacks_list.extend([model_checkpoint_callback, early_stopping_callback])
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет НЕ доступен.")
        print(
            f"Модель будет сохраняться в {checkpoint_filepath} по окончании обучения или каждой эпохи (если save_freq='epoch').")
        # Если нет валидации, можно сохранять модель в конце каждой эпохи или просто в конце обучения.
        # ModelCheckpoint без monitor будет сохранять в конце каждой эпохи.
        # Или можно просто model.save() после model.fit().
        # Для простоты теста, мы просто сохраним модель в конце.
        pass  # ModelCheckpoint не будем добавлять, сохраним вручную после fit

    # Уменьшаем количество эпох для быстрого теста, если нет валидации или мало данных
    epochs_to_run = CLASSIFIER_CONFIG['train_params']['epochs']
    if val_ds is None:  # Если валидации нет, то это точно тест на очень малых данных
        epochs_to_run = min(5, epochs_to_run)  # Не более 5 эпох для теста
        print(f"Запуск на {epochs_to_run} эпох (тестовый режим без валидации).")

    print(f"\nНачало обучения...")
    print(f"Размер батча: {CLASSIFIER_CONFIG['train_params']['batch_size']}")
    print(f"Начальная скорость обучения: {CLASSIFIER_CONFIG['train_params']['learning_rate']}")
    print(f"Логи TensorBoard будут сохраняться в: {logs_dir_abs}")
    if val_ds:
        print(f"Лучшая модель будет сохранена в: {checkpoint_filepath.replace('.keras', '_best.keras')}")
    else:
        print(f"Модель будет сохранена в: {checkpoint_filepath} после обучения.")

    # 5. Запуск Обучения
    history = model.fit(
        train_ds,
        epochs=epochs_to_run,
        validation_data=val_ds,  # Будет None, если val_ds is None
        callbacks=callbacks_list  # Будет содержать коллбэки в зависимости от наличия val_ds
    )

    print("\n--- Обучение классификатора завершено ---")

    # Сохраняем модель, если не было валидации (ModelCheckpoint не сработал на save_best_only)
    # или если хотим сохранить финальное состояние независимо от лучшего.
    if not val_ds or not any(
            isinstance(cb, tf.keras.callbacks.ModelCheckpoint) and cb.save_best_only for cb in callbacks_list):
        model.save(checkpoint_filepath)
        print(f"Финальная модель сохранена в: {checkpoint_filepath}")
    elif val_ds and not os.path.exists(checkpoint_filepath.replace('.keras', '_best.keras')):
        # Если была валидация, но по какой-то причине лучший файл не сохранился (например, обучение прервано до улучшения)
        # сохраним текущее состояние.
        model.save(checkpoint_filepath.replace(".keras", "_current_final.keras"))
        print(
            f"Финальная модель (текущее состояние) сохранена в: {checkpoint_filepath.replace('.keras', '_current_final.keras')}")

    # Оценка, если была валидация
    if val_ds:
        print(
            "\nОценка модели (потенциально лучшей, если сработал EarlyStopping с restore_best_weights) на валидационном наборе:")
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"  Потери на валидации: {val_loss:.4f}")
        print(f"  Точность на валидации: {val_acc * 100:.2f}%")


if __name__ == '__main__':
    train()