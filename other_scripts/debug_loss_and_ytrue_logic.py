# debug_loss_and_ytrue_logic.py
import tensorflow as tf
import yaml
import os
import sys
from pathlib import Path
import random  # Для выбора случайного файла

# --- Настройка sys.path для импорта из src ---
_project_root_debug_loss = Path(__file__).resolve().parent
_src_path_debug_loss = _project_root_debug_loss / 'src'
if str(_src_path_debug_loss) not in sys.path:
    sys.path.insert(0, str(_src_path_debug_loss))

# --- Импорты из твоих модулей ---
try:
    from datasets.other_loaders.detector_data_loader import (
        parse_xml_annotation,  # Нужен для чтения GT из XML
        assign_gt_to_fpn_levels_and_encode_by_scale,  # КЛЮЧЕВАЯ ФУНКЦИЯ для y_true
        FPN_LEVELS_CONFIG_GLOBAL,  # Глобальный конфиг уровней FPN
        CLASSES_LIST_GLOBAL_FOR_DETECTOR,
        NUM_CLASSES_DETECTOR,
        FPN_LEVEL_NAMES_ORDERED,
        BASE_CONFIG as DDL_BASE_CONFIG, # Для путей, если нужно
        # _images_subdir_name_cfg as DDL_IMAGES_SUBDIR,
        # _annotations_subdir_name_cfg as DDL_ANNOTATIONS_SUBDIR
    )
    from losses.other_losses.detection_losses import compute_detector_loss_v2_fpn

    print("INFO (debug_loss_script): Компоненты из data_loader и losses успешно импортированы.")
except ImportError as e_imp_main_debug_loss:
    print(f"КРИТИЧЕСКАЯ ОШИБКА (debug_loss_script): Не удалось импортировать компоненты: {e_imp_main_debug_loss}")
    exit()

# --- Загрузка КОПИИ detector_config для параметров ---
# Мы не будем менять основной конфиг, а будем использовать его значения
_detector_config_path_debug_loss = _src_path_debug_loss / 'configs' / 'detector_config.yaml'
DETECTOR_CONFIG_FOR_DEBUG_LOSS = {}
try:
    with open(_detector_config_path_debug_loss, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_FOR_DEBUG_LOSS = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_FOR_DEBUG_LOSS, dict): DETECTOR_CONFIG_FOR_DEBUG_LOSS = {}
except Exception:
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки detector_config.yaml. Некоторые параметры могут быть неверны.")
    # Задаем только критичные дефолты для fpn_detector_params, если их нет
    DETECTOR_CONFIG_FOR_DEBUG_LOSS.setdefault('fpn_detector_params', {}).setdefault('focal_loss_objectness_params',
                                                                                    {'use_focal_loss': True,
                                                                                     'alpha': 0.25, 'gamma': 2.0})
    DETECTOR_CONFIG_FOR_DEBUG_LOSS.setdefault('fpn_detector_params', {}).setdefault('loss_weights', {'coordinates': 1.0,
                                                                                                     'objectness': 1.0,
                                                                                                     'no_object': 0.5,
                                                                                                     'classification': 1.0})

# Параметры Focal Loss и веса потерь из конфига (используются в compute_detector_loss_v2_fpn)
# Убедимся, что compute_detector_loss_v2_fpn ИСПОЛЬЗУЕТ ИХ ИЗ КОНФИГА,
# а не жестко закодированные (через импорт DETECTOR_CONFIG в losses.py)
# или передадим их как аргументы, если функция потерь это поддерживает.
# Для простоты, сейчас будем полагаться на то, что detection_losses.py сам их загрузит.
# Но лучше передавать их явно.

# --- Выбор тестового изображения и аннотации ---
# Возьмем из твоего "мастер-датасета" Defective_Road_Images
# Пути должны быть настроены в base_config.yaml, который читается detector_data_loader.py
# Мы можем использовать DDL_BASE_CONFIG, если он импортирован
try:
    master_path_from_base_cfg = DDL_BASE_CONFIG.get('master_dataset_path', 'data/Master_Dataset')
    img_subdir_from_base_cfg = DDL_BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
    ann_subdir_from_base_cfg = DDL_BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')
    defective_parent_subdir_from_base_cfg = DDL_BASE_CONFIG.get('source_defective_road_img_parent_subdir',
                                                                'Defective_Road_Images')

    if not os.path.isabs(master_path_from_base_cfg):
        master_path_abs_debug = (_project_root_debug_loss / master_path_from_base_cfg).resolve()
    else:
        master_path_abs_debug = Path(master_path_from_base_cfg).resolve()

    debug_annotations_dir = master_path_abs_debug / defective_parent_subdir_from_base_cfg / ann_subdir_from_base_cfg
    debug_images_dir = master_path_abs_debug / defective_parent_subdir_from_base_cfg / img_subdir_from_base_cfg

    all_xml_files_debug = sorted(list(debug_annotations_dir.glob("*.xml")))
    if not all_xml_files_debug:
        print(f"ОШИБКА: Не найдены XML файлы в {debug_annotations_dir}")
        exit()

    # Выбираем ОДИН СЛУЧАЙНЫЙ XML для теста
    selected_xml_path_debug_str = str(random.choice(all_xml_files_debug))
    selected_base_name = Path(selected_xml_path_debug_str).stem

    selected_image_path_debug_str = None
    for ext_debug in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        candidate_debug = debug_images_dir / (selected_base_name + ext_debug)
        if candidate_debug.exists():
            selected_image_path_debug_str = str(candidate_debug)
            break
    if selected_image_path_debug_str is None:
        print(f"ОШИБКА: Не найдено изображение для {selected_xml_path_debug_str}")
        exit()

    print(f"--- Отладка y_true и Потерь для Изображения: {os.path.basename(selected_image_path_debug_str)} ---")

except KeyError as e_key:
    print(f"ОШИБКА: Не найдены необходимые ключи в base_config.yaml для путей к данным: {e_key}")
    print("         Убедитесь, что 'master_dataset_path' и подструктура 'dataset' определены.")
    exit()


def generate_y_true_for_single_image(image_path_str, xml_path_str):
    """Генерирует y_true для одного изображения, используя логику из data_loader."""
    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path_str).convert('RGB')
        # image_np_original_uint8 = np.array(pil_image, dtype=np.uint8) # Не нужно для assign_gt...
        original_w_px_val, original_h_px_val = pil_image.size
    except Exception as e_img:
        print(f"Ошибка загрузки изображения {image_path_str} для y_true: {e_img}")
        return None

    gt_objects, _, _, _ = parse_xml_annotation(xml_path_str)  # Используем импортированную parse_xml_annotation
    if gt_objects is None:
        print(f"Ошибка парсинга XML {xml_path_str} для y_true")
        return None

    boxes_list_pixels = []
    class_ids_list = []
    if gt_objects:
        for obj in gt_objects:
            boxes_list_pixels.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            class_ids_list.append(obj['class_id'])

    # Вызываем assign_gt_to_fpn_levels_and_encode_by_scale с пиксельными координатами
    # и оригинальными размерами изображения.
    # FPN_LEVELS_CONFIG_GLOBAL должен быть уже инициализирован при импорте detector_data_loader
    y_true_p3_np, y_true_p4_np, y_true_p5_np = assign_gt_to_fpn_levels_and_encode_by_scale(
        boxes_list_pixels,
        class_ids_list,
        original_w_px_val,
        original_h_px_val,
        FPN_LEVELS_CONFIG_GLOBAL  # Передаем глобальный конфиг якорей и сеток
    )
    return (tf.constant(y_true_p3_np, dtype=tf.float32),
            tf.constant(y_true_p4_np, dtype=tf.float32),
            tf.constant(y_true_p5_np, dtype=tf.float32))


def create_mock_y_pred(y_true_fpn_tuple_tf, scenario="perfect", noise_level=0.1):
    """Создает фиктивный y_pred на основе y_true."""
    y_pred_list = []
    for y_true_level_tf in y_true_fpn_tuple_tf:
        if scenario == "perfect":
            # Для "идеального" случая, логиты должны быть такими, чтобы sigmoid(logit) был близок к y_true
            # y_true_objectness_and_class = y_true_level_tf[..., 4:] # objectness + one-hot classes
            # pred_logits_obj_cls = tf.where(y_true_objectness_and_class > 0.5, 10.0, -10.0) # Большие логиты

            # Более простой способ для всех компонентов:
            y_pred_level_logits = tf.identity(y_true_level_tf)  # Копируем y_true
            # Для objectness и классов (которые были 0 или 1), преобразуем в логиты
            # objectness (индекс 4)
            y_pred_level_logits_obj = tf.where(y_true_level_tf[..., 4:5] > 0.5, 10.0, -10.0)
            # классы (индексы 5:)
            y_pred_level_logits_cls = tf.where(y_true_level_tf[..., 5:] > 0.5, 10.0, -10.0)

            # Собираем обратно, координаты оставляем как есть (они уже tx,ty,tw,th)
            y_pred_level_logits = tf.concat([
                y_true_level_tf[..., 0:4],  # true_boxes_encoded
                y_pred_level_logits_obj,  # pred_objectness_logits
                y_pred_level_logits_cls  # pred_classes_logits
            ], axis=-1)

        elif scenario == "random":
            y_pred_level_logits = tf.random.normal(shape=tf.shape(y_true_level_tf), stddev=1.0)
        elif scenario == "noisy_true":
            noise = tf.random.normal(shape=tf.shape(y_true_level_tf), stddev=noise_level)
            # Для objectness и классов, которые бинарные, шум может быть проблематичен.
            # Лучше добавить шум к логитам, соответствующим y_true.
            temp_pred_logits_obj = tf.where(y_true_level_tf[..., 4:5] > 0.5, 10.0, -10.0) + noise[..., 4:5]
            temp_pred_logits_cls = tf.where(y_true_level_tf[..., 5:] > 0.5, 10.0, -10.0) + noise[..., 5:]
            y_pred_level_logits = tf.concat([
                y_true_level_tf[..., 0:4] + noise[..., 0:4],  # Шум к координатам
                temp_pred_logits_obj,
                temp_pred_logits_cls
            ], axis=-1)
        else:  # По умолчанию "плохой" или "нулевой"
            y_pred_level_logits = tf.zeros_like(y_true_level_tf)

        y_pred_list.append(y_pred_level_logits)
    return tuple(y_pred_list)


# --- Основной отладочный код ---
if __name__ == "__main__":
    # 1. Генерируем y_true для нашего выбранного изображения
    y_true_for_selected_image_tuple_tf = generate_y_true_for_single_image(
        selected_image_path_debug_str,
        selected_xml_path_debug_str
    )

    if y_true_for_selected_image_tuple_tf is None:
        print("Не удалось сгенерировать y_true. Выход.")
        exit()

    # Добавляем batch измерение
    y_true_batch_tf_list_debug = [tf.expand_dims(yt, axis=0) for yt in y_true_for_selected_image_tuple_tf]
    y_true_batch_tf_debug = tuple(y_true_batch_tf_list_debug)

    print("\nСтатистика по 'позитивным' якорям в сгенерированном y_true:")
    for i_debug_yt, yt_level_debug_item in enumerate(y_true_batch_tf_debug):
        level_name_debug_stat = FPN_LEVEL_NAMES_ORDERED[i_debug_yt]
        # yt_level_debug_item имеет форму (1, Gh, Gw, A, 5+C)
        true_objectness_level = yt_level_debug_item[..., 4:5]  # (1, Gh, Gw, A, 1)
        num_pos_this_level = tf.reduce_sum(tf.cast(tf.equal(true_objectness_level, 1.0), tf.float32))
        print(f"  Уровень {level_name_debug_stat}: Найдено 'позитивных' якорей в y_true = {num_pos_this_level.numpy()}")

    print("\nФормы сгенерированного y_true для FPN (батч из 1):")
    for i_yt_debug, yt_level_debug in enumerate(y_true_batch_tf_debug):
        print(f"  Уровень {FPN_LEVEL_NAMES_ORDERED[i_yt_debug]}: {yt_level_debug.shape}")

    # 2. Тестируем функцию потерь с "идеальными" y_pred
    print("\n--- Тест 1: Идеальные предсказания ---")
    y_pred_perfect_batch_tf = create_mock_y_pred(y_true_batch_tf_debug, scenario="perfect")

    # Устанавливаем флаг для детального вывода из функции потерь
    os.environ['DEBUG_TRAINING_LOOP_ACTIVE'] = '1'
    loss_details_perfect = compute_detector_loss_v2_fpn(y_true_batch_tf_debug, y_pred_perfect_batch_tf)

    print("  Детальные Потери (Идеальные):")
    if isinstance(loss_details_perfect, dict):
        for k, v_tensor in loss_details_perfect.items():
            print(f"    {k}: {v_tensor.numpy():.8f}")
    else:  # Если функция потерь вернула только total_loss
        print(f"    total_loss: {loss_details_perfect.numpy():.8f}")

    # 3. Тестируем функцию потерь со "случайными" y_pred
    print("\n--- Тест 2: Случайные предсказания ---")
    y_pred_random_batch_tf = create_mock_y_pred(y_true_batch_tf_debug, scenario="random")
    loss_details_random = compute_detector_loss_v2_fpn(y_true_batch_tf_debug, y_pred_random_batch_tf)

    print("  Детальные Потери (Случайные):")
    if isinstance(loss_details_random, dict):
        for k, v_tensor in loss_details_random.items():
            print(f"    {k}: {v_tensor.numpy():.6f}")
    else:
        print(f"    total_loss: {loss_details_random.numpy():.6f}")

    # 4. Тестируем функцию потерь с "зашумленными правильными" y_pred
    print("\n--- Тест 3: Зашумленные правильные предсказания (малый шум) ---")
    y_pred_noisy_batch_tf = create_mock_y_pred(y_true_batch_tf_debug, scenario="noisy_true",
                                               noise_level=0.01)  # Очень малый шум
    loss_details_noisy = compute_detector_loss_v2_fpn(y_true_batch_tf_debug, y_pred_noisy_batch_tf)

    print("  Детальные Потери (Малый шум):")
    if isinstance(loss_details_noisy, dict):
        for k, v_tensor in loss_details_noisy.items():
            print(f"    {k}: {v_tensor.numpy():.8f}")  # Больше знаков после запятой
    else:
        print(f"    total_loss: {loss_details_noisy.numpy():.8f}")

    # Убираем флаг, если он не нужен другим скриптам
    if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ:
        del os.environ["DEBUG_TRAINING_LOOP_ACTIVE"]

    print("\n--- Отладка y_true и функции потерь завершена. ---")