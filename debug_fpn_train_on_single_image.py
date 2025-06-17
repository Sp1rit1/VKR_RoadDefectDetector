# debug_fpn_train_on_single_image.py
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import time  # Для замера времени
import random  # Хотя для одного изображения не используется, оставим для консистентности

# --- Настройка sys.path для импорта из src ---
# Предполагается, что этот скрипт лежит в корне проекта RoadDefectDetector/
_project_root_debug = os.path.dirname(os.path.abspath(__file__))
_src_path_debug = os.path.join(_project_root_debug, 'src')
if str(_src_path_debug) not in sys.path:
    sys.path.insert(0, str(_src_path_debug))

# --- Импорты из твоих модулей ---
# detector_data_loader теперь должен быть доступен, так как src в sys.path
try:
    from datasets.detector_data_loader import (
        load_and_prepare_detector_fpn_py_func,  # Нам нужна именно py_func для одного примера
        FPN_LEVELS_CONFIG_GLOBAL,  # Глобальный конфиг уровней FPN из data_loader
        CLASSES_LIST_GLOBAL_FOR_DETECTOR,  # Список классов из data_loader
        TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH,  # Размеры изображений из data_loader
        NUM_CLASSES_DETECTOR,  # Количество классов из data_loader
        FPN_LEVEL_NAMES_ORDERED  # Порядок уровней FPN из data_loader
    )

    print("INFO (debug_script): Компоненты из detector_data_loader.py успешно импортированы.")
except ImportError as e_imp_ddl:
    print(
        f"КРИТИЧЕСКАЯ ОШИБКА (debug_script): Не удалось импортировать из src.datasets.detector_data_loader: {e_imp_ddl}")
    print("Убедитесь, что __init__.py есть в src/ и src/datasets/, и что sys.path настроен правильно.")
    exit()

try:
    from models.object_detector import build_object_detector_v2_fpn  # Твоя FPN модель

    print("INFO (debug_script): Модель build_object_detector_v2_fpn успешно импортирована.")
except ImportError as e_imp_model:
    print(
        f"КРИТИЧЕСКАЯ ОШИБКА (debug_script): Не удалось импортировать build_object_detector_v2_fpn из src.models.object_detector: {e_imp_model}")
    exit()

try:
    from losses.detection_losses import compute_detector_loss_v2_fpn  # Твоя FPN функция потерь

    print("INFO (debug_script): Функция потерь compute_detector_loss_v2_fpn успешно импортирована.")
except ImportError as e_imp_loss:
    print(
        f"КРИТИЧЕСКАЯ ОШИБКА (debug_script): Не удалось импортировать compute_detector_loss_v2_fpn из src.losses.detection_losses: {e_imp_loss}")
    exit()

# --- Загрузка основного detector_config для параметров модели и потерь ---
_detector_config_path_debug = os.path.join(_src_path_debug, 'configs', 'detector_config.yaml')
DETECTOR_CONFIG_DEBUG = {}
try:
    with open(_detector_config_path_debug, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_DEBUG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_DEBUG, dict):
        DETECTOR_CONFIG_DEBUG = {}
        print(f"ПРЕДУПРЕЖДЕНИЕ (debug_script): detector_config.yaml пуст или не словарь. Используются дефолты.")
except FileNotFoundError:
    print(f"ОШИБКА: Файл detector_config.yaml не найден: {_detector_config_path_debug}")
    DETECTOR_CONFIG_DEBUG.setdefault('train_params',
                                     {'learning_rate': 1e-4, 'initial_learning_rate': 1e-4})  # Дефолтный LR
    DETECTOR_CONFIG_DEBUG.setdefault('fpn_detector_params', {'classes': ['pit', 'crack']})
    print("ПРЕДУПРЕЖДЕНИЕ (debug_script): detector_config.yaml не найден. Используются минимальные дефолты.")
except yaml.YAMLError as e_cfg_debug:
    print(f"ОШИБКА YAML (debug_script): Не удалось прочитать detector_config.yaml: {e_cfg_debug}")
    DETECTOR_CONFIG_DEBUG.setdefault('train_params', {'learning_rate': 1e-4, 'initial_learning_rate': 1e-4})
    DETECTOR_CONFIG_DEBUG.setdefault('fpn_detector_params', {'classes': ['pit', 'crack']})
    print("ПРЕДУПРЕЖДЕНИЕ (debug_script): Ошибка чтения detector_config.yaml. Используются минимальные дефолты.")

# --- Константы для отладки ---
# !!! ЗАМЕНИ ЭТИ ПУТИ НА ПУТИ К ТВОЕМУ ОДНОМУ ТЕСТОВОМУ ИЗОБРАЖЕНИЮ И XML !!!
# Пути теперь относительно корня проекта
DEBUG_IMAGE_PATH = os.path.join(_project_root_debug, "debug_data", "China_Drone_000180.jpg")  # ИЗМЕНИ НА СВОЙ ФАЙЛ
DEBUG_XML_PATH = os.path.join(_project_root_debug, "debug_data", "China_Drone_000180.xml")  # ИЗМЕНИ НА СВОЙ ФАЙЛ

NUM_ITERATIONS_DEBUG = DETECTOR_CONFIG_DEBUG.get('train_params', {}).get('epochs_test_overfit', 300)
LEARNING_RATE_DEBUG = DETECTOR_CONFIG_DEBUG.get('train_params', {}).get('initial_learning_rate', 0.0001)

VISUALIZE_Y_TRUE_DEBUG = True  # Установи в True, чтобы увидеть y_true для этого изображения

# --- Попытка импортировать визуализацию ---
_plot_utils_imported_successfully_debug_script = False
visualize_fpn_gt_assignments_debug_func = lambda *args, **kwargs: print(
    "Визуализация GT отключена (plot_utils не импортирован в debug_script).")
if VISUALIZE_Y_TRUE_DEBUG:
    try:
        from utils.plot_utils import visualize_fpn_gt_assignments as viz_func_imported_debug

        _plot_utils_imported_successfully_debug_script = True
        visualize_fpn_gt_assignments_debug_func = viz_func_imported_debug
        print("INFO (debug_script): plot_utils для визуализации GT загружен.")
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ (debug_script): plot_utils не найден, визуализация GT не будет выполнена.")
    except Exception as e_plot_debug:
        print(f"ПРЕДУПРЕЖДЕНИЕ (debug_script): Ошибка импорта plot_utils: {e_plot_debug}")


def main_debug_train():
    print("\n--- Отладочный Запуск Обучения FPN на Одном Изображении ---")

    if not os.path.exists(DEBUG_IMAGE_PATH) or not os.path.exists(DEBUG_XML_PATH):
        print(f"ОШИБКА: Тестовое изображение ({DEBUG_IMAGE_PATH}) или XML ({DEBUG_XML_PATH}) не найдены.")
        print("Пожалуйста, создай папку 'debug_data' в корне проекта и положи туда одно изображение и его XML.")
        print(f"Текущая рабочая директория: {os.getcwd()}")
        return

    print(f"Загрузка и обработка: {os.path.basename(DEBUG_IMAGE_PATH)}")

    # Вызываем py_func напрямую
    # Она возвращает: image_np, y_true_p3_np, y_true_p4_np, y_true_p5_np, boxes_viz_np, ids_viz_np
    returned_from_py_func = load_and_prepare_detector_fpn_py_func(
        tf.constant(DEBUG_IMAGE_PATH, dtype=tf.string),
        tf.constant(DEBUG_XML_PATH, dtype=tf.string),
        tf.constant(False, dtype=tf.bool)  # Без аугментации для отладки
    )

    image_processed_np = returned_from_py_func[0]
    y_true_p3_np = returned_from_py_func[1]
    y_true_p4_np = returned_from_py_func[2]
    y_true_p5_np = returned_from_py_func[3]
    y_true_fpn_tuple_np = (y_true_p3_np, y_true_p4_np, y_true_p5_np)

    scaled_gt_boxes_for_viz_np = returned_from_py_func[4]
    class_ids_for_viz_np = returned_from_py_func[5]

    if image_processed_np is None or y_true_fpn_tuple_np[0] is None:  # Проверка хотя бы первого y_true
        print("ОШИБКА: Не удалось обработать данные для отладочного примера.")
        return

    image_batch_tf = tf.expand_dims(tf.convert_to_tensor(image_processed_np, dtype=tf.float32), axis=0)
    y_true_fpn_batch_tf_list = []
    for y_true_level_np in y_true_fpn_tuple_np:
        y_true_fpn_batch_tf_list.append(
            tf.expand_dims(tf.convert_to_tensor(y_true_level_np, dtype=tf.float32), axis=0)
        )
    y_true_fpn_batch_tf = tuple(y_true_fpn_batch_tf_list)

    print("\nФорма входного изображения для модели:", image_batch_tf.shape)
    for i, y_true_level in enumerate(y_true_fpn_batch_tf):
        level_name_debug = FPN_LEVEL_NAMES_ORDERED[i] if i < len(FPN_LEVEL_NAMES_ORDERED) else f"Unknown_Level_{i}"
        print(f"Форма y_true для уровня FPN {level_name_debug}: {y_true_level.shape}")

    if VISUALIZE_Y_TRUE_DEBUG and _plot_utils_imported_successfully_debug_script:
        print("\nВизуализация Ground Truth назначения для отладочного изображения...")
        try:
            visualize_fpn_gt_assignments_debug_func(
                image_processed_np,
                y_true_fpn_tuple_np,
                original_gt_boxes_for_reference=(scaled_gt_boxes_for_viz_np, class_ids_for_viz_np),
                title_prefix=f"Debug GT: {os.path.basename(DEBUG_IMAGE_PATH)}"
            )
            print("Проверь появившееся окно с визуализацией. Закрой его, чтобы продолжить обучение.")
        except Exception as e_viz_debug:
            print(f"ОШИБКА при вызове visualize_fpn_gt_assignments: {e_viz_debug}")
            import traceback
            traceback.print_exc()

    print("\nСоздание модели build_object_detector_v2_fpn...")
    # build_object_detector_v2_fpn использует глобальные конфиги, загруженные в object_detector.py
    # Убедись, что они там загружаются правильно
    model = build_object_detector_v2_fpn()
    # model.summary(line_length=120) # Можно раскомментировать для проверки структуры

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEBUG)
    print(f"\nИспользуется Learning Rate: {LEARNING_RATE_DEBUG}")
    print(f"\nЗапуск {NUM_ITERATIONS_DEBUG} итераций обучения на одном примере...")

    for iteration in range(NUM_ITERATIONS_DEBUG):
        with tf.GradientTape() as tape:
            y_pred_fpn_list_tf = model(image_batch_tf, training=True)

            if not isinstance(y_pred_fpn_list_tf, (list, tuple)) or len(y_pred_fpn_list_tf) != len(y_true_fpn_batch_tf):
                print(f"ОШИБКА: Выход модели имеет неожиданную структуру. "
                      f"Ожидался кортеж/список из {len(y_true_fpn_batch_tf)} элементов, "
                      f"получено {len(y_pred_fpn_list_tf) if isinstance(y_pred_fpn_list_tf, (list, tuple)) else type(y_pred_fpn_list_tf)}")
                return

            total_loss = compute_detector_loss_v2_fpn(y_true_fpn_batch_tf, y_pred_fpn_list_tf)

        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Проверка градиентов на NaN/Inf (опционально, но полезно при отладке)
        # for i_grad, grad in enumerate(gradients):
        #     if grad is None:
        #         print(f"WARNING: Градиент для переменной {model.trainable_variables[i_grad].name} равен None")
        #         continue
        #     if tf.reduce_any(tf.math.is_nan(grad)).numpy() or tf.reduce_any(tf.math.is_inf(grad)).numpy():
        #         print(f"ОШИБКА: Градиент для переменной {model.trainable_variables[i_grad].name} содержит NaN или Inf!")
        #         # Можно добавить вывод самих градиентов для анализа
        #         # print(grad.numpy())
        #         # return # Прервать обучение

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Итерация: {iteration + 1}/{NUM_ITERATIONS_DEBUG}, Потеря (Loss): {total_loss.numpy():.6f}")
            if np.isnan(total_loss.numpy()) or np.isinf(total_loss.numpy()):
                print("ОШИБКА: Потеря стала NaN или Inf! Остановка.")
                break

    print("\n--- Отладочное обучение на одном примере завершено. ---")

    # Сохранение модели для последующего анализа, если нужно
    # debug_model_save_path = os.path.join(_project_root_debug, "weights", "debug_fpn_single_image_trained.keras")
    # model.save(debug_model_save_path)
    # print(f"Модель после отладочного обучения сохранена в: {debug_model_save_path}")

    if _plot_utils_imported_successfully_debug_script:
        print("\nПредсказание и визуализация на том же изображении после цикла обучения...")
        # Здесь нужен будет код для predict_detector.py: decode_predictions, apply_nms, draw_detections
        # Для простоты пока оставим так.
        print("Визуализация предсказаний после обучения пока не реализована в этом отладочном скрипте.")


if __name__ == '__main__':
    # Убедись, что:
    # 1. DEBUG_IMAGE_PATH и DEBUG_XML_PATH указывают на существующие файлы.
    # 2. Папка debug_data создана в корне проекта.
    # 3. src/configs/detector_config.yaml содержит правильные параметры FPN,
    #    особенно fpn_anchor_configs с якорями для P3, P4, P5 и их num_anchors_this_level.
    # 4. Модули в src/datasets, src/models, src/losses, src/utils не содержат ошибок импорта/синтаксиса.
    main_debug_train()