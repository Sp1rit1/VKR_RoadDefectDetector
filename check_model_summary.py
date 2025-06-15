# check_model_summary.py
import tensorflow as tf
import yaml
import os
import sys

# --- Добавляем src в sys.path, чтобы импортировать кастомные объекты, если нужно ---
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Попытка импортировать кастомную функцию потерь (если она нужна для загрузки)
try:
    from losses.detection_losses import compute_detector_loss_v1
    CUSTOM_OBJECTS = {'compute_detector_loss_v1': compute_detector_loss_v1}
    print("INFO: Кастомная функция потерь загружена для custom_objects.")
except ImportError:
    CUSTOM_OBJECTS = {}
    print("ПРЕДУПРЕЖДЕНИЕ: Кастомная функция потерь не найдена. Модель будет загружаться без нее (может вызвать ошибку, если она не стандартная).")
except Exception as e:
    CUSTOM_OBJECTS = {}
    print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при импорте кастомных объектов: {e}")


# --- Путь к модели ---
# Замени на актуальный путь к твоей лучшей сохраненной модели
# Сначала попробуй загрузить модель, которую ты собираешься использовать для fine-tuning'а
# Например, ту, что указана в detector_config.yaml -> path_to_checkpoint

# Загружаем базовый конфиг, чтобы взять путь к весам
_base_config_path = os.path.join(_project_root, 'src', 'configs', 'base_config.yaml')
_detector_config_path = os.path.join(_project_root, 'src', 'configs', 'detector_config.yaml') # Нам нужен detector_config для path_to_checkpoint

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"Ошибка загрузки конфигов: {e}")
    print("Укажите путь к модели вручную в скрипте.")
    exit()

WEIGHTS_BASE_DIR_ABS = os.path.join(_project_root, BASE_CONFIG.get('weights_base_dir', 'weights'))
PATH_TO_CHECKPOINT_REL_CFG = DETECTOR_CONFIG.get('path_to_checkpoint', None)

if PATH_TO_CHECKPOINT_REL_CFG:
    MODEL_FILE_PATH = os.path.join(WEIGHTS_BASE_DIR_ABS, PATH_TO_CHECKPOINT_REL_CFG)
else:
    print("ОШИБКА: 'path_to_checkpoint' не указан в detector_config.yaml")
    # Задай путь вручную, если нужно протестировать конкретную модель
    # MODEL_FILE_PATH = os.path.join(WEIGHTS_BASE_DIR_ABS, "ИМЯ_ТВОЕЙ_МОДЕЛИ.keras")
    exit()


if not os.path.exists(MODEL_FILE_PATH):
    print(f"ОШИБКА: Файл модели не найден: {MODEL_FILE_PATH}")
    print("Пожалуйста, проверьте путь 'path_to_checkpoint' в 'src/configs/detector_config.yaml' "
          "и наличие файла модели в папке 'weights/'.")
else:
    print(f"Загрузка модели из: {MODEL_FILE_PATH}")
    try:
        # Загружаем модель. compile=False обычно безопаснее, если нам нужна только структура и веса.
        # Если Keras требует функцию потерь при загрузке, даже с compile=False, передаем ее.
        loaded_model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
        print("Модель успешно загружена.")
        print("\n--- Структура Загруженной Модели (model.summary()) ---")
        loaded_model.summary(line_length=150)

        print("\n--- Имена Слоев Верхнего Уровня Модели ---")
        for layer in loaded_model.layers:
            print(f"Имя: {layer.name}, Тип: {type(layer).__name__}, Обучаемый: {layer.trainable}")
            # Если ты знаешь, что твой backbone - это вложенная модель, можно пойти глубже
            if layer.name == DETECTOR_CONFIG.get('backbone_layer_name_in_model_expected', 'Backbone_MobileNetV2'): # Имя из конфига, которое ты ожидаешь
                print(f"  >>> Найден предполагаемый слой Backbone: '{layer.name}'")
                if isinstance(layer, tf.keras.Model) and hasattr(layer, 'layers'):
                    print(f"    Это вложенная модель (Functional API). Слои внутри '{layer.name}':")
                    # for sub_layer_idx, sub_layer in enumerate(layer.layers):
                    #     print(f"      {sub_layer_idx}: {sub_layer.name} ({type(sub_layer).__name__})")
                else:
                    print(f"    Это не вложенная модель tf.keras.Model, а слой типа: {type(layer).__name__}")


        # Пример, как найти конкретный слой по имени и проверить его обучаемость
        # Замени 'Backbone_MobileNetV2' на имя, которое ты ожидаешь или видишь в summary
        expected_backbone_name = DETECTOR_CONFIG.get('backbone_layer_name_in_model', 'Backbone_MobileNetV2')
        try:
            backbone_layer_found = loaded_model.get_layer(expected_backbone_name)
            print(f"\n--- Проверка конкретного слоя Backbone ('{expected_backbone_name}') ---")
            print(f"Слой '{backbone_layer_found.name}' найден.")
            print(f"  Тип слоя: {type(backbone_layer_found).__name__}")
            print(f"  Обучаемый (trainable): {backbone_layer_found.trainable}")
            if isinstance(backbone_layer_found, tf.keras.Model):
                print(f"  Этот слой Backbone сам является моделью (Functional API) и содержит {len(backbone_layer_found.layers)} внутренних слоев.")
        except ValueError:
            print(f"\nОШИБКА: Слой с именем '{expected_backbone_name}' НЕ НАЙДЕН в загруженной модели!")
            print("Пожалуйста, проверьте вывод model.summary() выше и исправьте 'backbone_layer_name_in_model' в detector_config.yaml.")


    except Exception as e:
        print(f"Произошла ошибка при загрузке или анализе модели: {e}")
        import traceback
        traceback.print_exc()