# debug_fpn_train_on_single_image.py
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import time
import random
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# --- Настройка sys.path для импорта из src ---
_project_root_debug = Path(__file__).resolve().parent
_src_path_debug = _project_root_debug / 'src'
if str(_src_path_debug) not in sys.path:
    sys.path.insert(0, str(_src_path_debug))
if str(_project_root_debug) not in sys.path:
    sys.path.insert(0, str(_project_root_debug))

# --- Импорты из твоих модулей ---
try:
    from datasets.detector_data_loader import (
        load_and_prepare_detector_fpn_py_func,
        FPN_LEVELS_CONFIG_GLOBAL, CLASSES_LIST_GLOBAL_FOR_DETECTOR,
        TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, NUM_CLASSES_DETECTOR,
        FPN_LEVEL_NAMES_ORDERED,
        BASE_CONFIG as DDL_BASE_CONFIG,
        _images_subdir_name_cfg as DDL_IMAGES_SUBDIR_NAME,
        _annotations_subdir_name_cfg as DDL_ANNOTATIONS_SUBDIR_NAME
    )
    from models.object_detector import build_object_detector_v2_fpn
    from losses.detection_losses import compute_detector_loss_v2_fpn

    print("INFO (debug_script): Основные компоненты (data_loader, model, loss) успешно импортированы.")
except ImportError as e_imp_main:
    print(f"КРИТИЧЕСКАЯ ОШИБКА (debug_script): Не удалось импортировать основные компоненты: {e_imp_main}")
    exit()

_predict_utils_imported = False
_plot_utils_imported_for_preds = False
try:
    # ЗАМЕНИ 'run_prediction_pipeline' на имя твоего скрипта, где лежат эти функции
    from run_prediction_pipeline import decode_single_level_predictions_generic, apply_nms_and_filter_generic

    _predict_utils_imported = True
    print("INFO (debug_script): Функции decode и nms импортированы.")
except ImportError as e_imp_pred_utils:
    print(f"ПРЕДУПРЕЖДЕНИЕ (debug_script): Не удалось импортировать decode/nms: {e_imp_pred_utils}")
try:
    from utils.plot_utils import visualize_fpn_detections_vs_gt

    _plot_utils_imported_for_preds = True
    print("INFO (debug_script): Функция visualize_fpn_detections_vs_gt импортирована.")
except ImportError as e_imp_plot_utils:
    print(f"ПРЕДУПРЕЖДЕНИЕ (debug_script): Не удалось импортировать visualize_fpn_detections_vs_gt: {e_imp_plot_utils}")

# --- Загрузка detector_config ---
_detector_config_path_debug = os.path.join(_src_path_debug, 'configs', 'detector_config.yaml')
DETECTOR_CONFIG_DEBUG = {}
CONFIG_LOAD_SUCCESS_DEBUG = True # Флаг для проверки успешной загрузки
try:
    with open(_detector_config_path_debug, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_DEBUG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_DEBUG, dict) or not DETECTOR_CONFIG_DEBUG: # Проверка, что не пустой
        print(f"ПРЕДУПРЕЖДЕНИЕ (debug_script): detector_config.yaml ({_detector_config_path_debug}) пуст или имеет неверный формат.")
        CONFIG_LOAD_SUCCESS_DEBUG = False
except FileNotFoundError:
    print(f"ОШИБКА (debug_script): Файл detector_config.yaml не найден: {_detector_config_path_debug}")
    CONFIG_LOAD_SUCCESS_DEBUG = False
except yaml.YAMLError as e_cfg_debug:
    print(f"ОШИБКА YAML (debug_script): Не удалось прочитать detector_config.yaml: {e_cfg_debug}")
    CONFIG_LOAD_SUCCESS_DEBUG = False

# Если конфиг не загружен, используем минимальные жестко закодированные дефолты,
# чтобы скрипт хотя бы попытался запуститься, но с предупреждением.
if not CONFIG_LOAD_SUCCESS_DEBUG:
    print("ПРЕДУПРЕЖДЕНИЕ (debug_script): Используются АВАРИЙНЫЕ ДЕФОЛТЫ из-за ошибки загрузки detector_config.yaml.")
    # Аварийные дефолты, чтобы скрипт не упал на KeyError, но они могут быть нерелевантны
    DETECTOR_CONFIG_DEBUG = {
        'fpn_detector_params': {
            'train_params': {'initial_learning_rate': 1e-4, 'epochs_for_debug': 100}, # Добавим epochs_for_debug
            'predict_params': {'confidence_threshold': 0.1, 'iou_threshold': 0.45, 'max_detections': 100}
        },
        'initial_learning_rate': 1e-4, # Общий
        'epochs': 100 # Общие эпохи
    }

# --- Константы для отладки из загруженного или дефолтного конфига ---

# Параметры обучения
# Сначала пытаемся взять из fpn_detector_params.train_params (если там есть специфичные для FPN)
# потом из общей секции train_params (если она есть на верхнем уровне)
# потом из общей секции (если initial_learning_rate и epochs там)
_fpn_train_params = DETECTOR_CONFIG_DEBUG.get('fpn_detector_params', {}).get('train_params', {})
_general_train_params = DETECTOR_CONFIG_DEBUG.get('train_params', {}) # Если есть общая секция train_params

LEARNING_RATE_DEBUG = _fpn_train_params.get('initial_learning_rate',
                                        _general_train_params.get('initial_learning_rate',
                                                                DETECTOR_CONFIG_DEBUG.get('initial_learning_rate', 0.0001)))

NUM_ITERATIONS_DEBUG = _fpn_train_params.get('epochs_for_debug', # Сначала ищем специальный параметр
                                           _general_train_params.get('epochs', # Потом общие эпохи
                                                                   DETECTOR_CONFIG_DEBUG.get('epochs', 200)))

VISUALIZE_EVERY_N_ITERATIONS = max(1, NUM_ITERATIONS_DEBUG // 1) # Визуализируем ~10-20 раз + первая и последняя

# Параметры для NMS и визуализации
# Сначала ищем в fpn_detector_params.predict_params, потом в общей секции predict_params
_fpn_predict_params = DETECTOR_CONFIG_DEBUG.get('fpn_detector_params', {}).get('predict_params', {})
_general_predict_params = DETECTOR_CONFIG_DEBUG.get('predict_params', {})

CONF_THRESH_VIZ = _fpn_predict_params.get('confidence_threshold',
                                        _general_predict_params.get('confidence_threshold', 0.05))

IOU_THRESH_NMS_VIZ = _fpn_predict_params.get('iou_threshold',
                                           _general_predict_params.get('iou_threshold', 0.3))

MAX_DETS_VIZ = _fpn_predict_params.get('max_detections',
                                     _general_predict_params.get('max_detections', 50))

print(f"\n--- Параметры для отладочного запуска ---")
print(f"  Learning Rate: {LEARNING_RATE_DEBUG}")
print(f"  Количество итераций (эпох на 1 примере): {NUM_ITERATIONS_DEBUG}")
print(f"  Визуализация каждые ~{VISUALIZE_EVERY_N_ITERATIONS} итераций")
print(f"  Порог уверенности для визуализации (NMS): {CONF_THRESH_VIZ}")
print(f"  Порог IoU для визуализации (NMS): {IOU_THRESH_NMS_VIZ}")

os.environ['DEBUG_TRAINING_LOOP_ACTIVE'] = '1' # Для детального вывода функции


def check_weights_changed(model, initial_weights_list, trainable_vars_names_list):
    changed_count = 0;
    not_changed_count = 0
    for i, var in enumerate(model.trainable_variables):
        if i < len(initial_weights_list):
            if not np.array_equal(initial_weights_list[i], var.numpy()):
                changed_count += 1
            else:
                not_changed_count += 1
    print(
        f"  Проверка весов: Изменилось: {changed_count}, Не изменилось: {not_changed_count} (из {len(model.trainable_variables)}).")
    return changed_count > 0


def main_debug_train_single_random_defective_image():
    print("\n--- Отладка FPN: Обучение на ОДНОМ СЛУЧАЙНОМ Изображении из 'Defective_Road_Images' ---")

    master_dataset_root_abs = Path(DDL_BASE_CONFIG.get('master_dataset_path', 'data/Master_Dataset_Fallback'))
    if not os.path.isabs(master_dataset_root_abs):
        master_dataset_root_abs = _project_root_debug / master_dataset_root_abs
    defective_images_parent_dir_name = DDL_BASE_CONFIG.get('source_defective_road_img_parent_subdir',
                                                           "Defective_Road_Images")
    source_images_dir = Path("C:/Users/0001/Desktop/Diplom/RoadDefectDetector/debug_data")############################################master_dataset_root_abs / defective_images_parent_dir_name / DDL_IMAGES_SUBDIR_NAME
    source_annotations_dir = Path("C:/Users/0001/Desktop/Diplom/RoadDefectDetector/debug_data")########################################master_dataset_root_abs / defective_images_parent_dir_name / DDL_ANNOTATIONS_SUBDIR_NAME

    if not source_images_dir.is_dir() or not source_annotations_dir.is_dir():
        print(f"ОШИБКА: Не найдены директории для '{defective_images_parent_dir_name}'.");
        return
    all_xml_files = sorted(list(source_annotations_dir.glob("*.xml")))
    if not all_xml_files: print(f"ОШИБКА: XML файлы не найдены в {source_annotations_dir}"); return

    selected_xml_path_obj = random.choice(all_xml_files)
    base_name = selected_xml_path_obj.stem
    selected_image_path_obj = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        candidate = source_images_dir / (base_name + ext)
        if candidate.exists(): selected_image_path_obj = candidate; break
    if selected_image_path_obj is None: print(
        f"ОШИБКА: Не найдено изображение для XML: {selected_xml_path_obj.name}"); return

    DEBUG_IMAGE_PATH_SELECTED = str(selected_image_path_obj)
    DEBUG_XML_PATH_SELECTED = str(selected_xml_path_obj)
    print(f"Выбрано для отладки: {os.path.basename(DEBUG_IMAGE_PATH_SELECTED)}")

    py_func_outputs = load_and_prepare_detector_fpn_py_func(
        tf.constant(DEBUG_IMAGE_PATH_SELECTED, dtype=tf.string),
        tf.constant(DEBUG_XML_PATH_SELECTED, dtype=tf.string),
        tf.constant(False, dtype=tf.bool)
    )
    image_processed_np = py_func_outputs[0]
    y_true_fpn_tuple_np_for_loss = tuple(py_func_outputs[1:4])
    original_gt_boxes_viz_np = py_func_outputs[4]
    original_gt_class_ids_viz_np = py_func_outputs[5]
    original_gt_for_reference_viz = (original_gt_boxes_viz_np, original_gt_class_ids_viz_np)

    image_batch_tf = tf.expand_dims(tf.convert_to_tensor(image_processed_np, dtype=tf.float32), axis=0)
    y_true_fpn_batch_tf_for_loss = tuple(
        [tf.expand_dims(tf.convert_to_tensor(yt_l, dtype=tf.float32), axis=0) for yt_l in y_true_fpn_tuple_np_for_loss])

    model = build_object_detector_v2_fpn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEBUG)
    initial_trainable_weights = [var.numpy().copy() for var in model.trainable_variables]
    trainable_variable_names = [var.name for var in model.trainable_variables]
    print(f"\nЗапуск {NUM_ITERATIONS_DEBUG} итераций обучения на {os.path.basename(DEBUG_IMAGE_PATH_SELECTED)}...")

    for iteration in range(NUM_ITERATIONS_DEBUG):
        with tf.GradientTape() as tape:
            y_pred_fpn_list_tf_logits = model(image_batch_tf, training=True)
            loss_details_dict = compute_detector_loss_v2_fpn(y_true_fpn_batch_tf_for_loss, y_pred_fpn_list_tf_logits)
            total_loss = loss_details_dict['total_loss']

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (iteration + 1) % (NUM_ITERATIONS_DEBUG // 20 or 1) == 0 or iteration == 0:
            print(f"\n--- Итерация: {iteration + 1}/{NUM_ITERATIONS_DEBUG} ---")
            print("  Детальные Потери:")
            for k, v_tensor in loss_details_dict.items(): print(f"    {k}: {v_tensor.numpy():.6f}")
            if np.isnan(total_loss.numpy()) or np.isinf(total_loss.numpy()): print("ОШИБКА: Потеря NaN/Inf!"); break

        if _plot_utils_imported_for_preds and _predict_utils_imported and \
                ((
                         iteration + 1) % VISUALIZE_EVERY_N_ITERATIONS == 0 or iteration == NUM_ITERATIONS_DEBUG - 1 or iteration == 0):
            print(f"  --- Визуализация предсказаний на итерации {iteration + 1} ---")

            pred_for_viz_raw_logits_list = model(image_batch_tf,
                                                 training=False)  # Список из 3х тензоров (B,Gh,Gw,A,5+C)

            # Декодируем каждый уровень FPN отдельно
            decoded_boxes_per_level_viz = []
            obj_conf_per_level_viz = []
            class_probs_per_level_viz = []

            for level_idx_viz, level_raw_pred_item in enumerate(pred_for_viz_raw_logits_list):
                level_name_item = FPN_LEVEL_NAMES_ORDERED[level_idx_viz]
                level_cfg_item = FPN_LEVELS_CONFIG_GLOBAL[level_name_item]

                # Используем decode_single_level_predictions_generic
                dec_boxes_lvl, dec_obj_lvl, dec_cls_lvl = decode_single_level_predictions_generic(
                    level_raw_pred_item,
                    level_cfg_item['anchors_wh_normalized'],  # np.array
                    level_cfg_item['grid_h'],
                    level_cfg_item['grid_w'],
                    NUM_CLASSES_DETECTOR,  # int
                )
                decoded_boxes_per_level_viz.append(dec_boxes_lvl)
                obj_conf_per_level_viz.append(dec_obj_lvl)
                class_probs_per_level_viz.append(dec_cls_lvl)

            # Теперь передаем СПИСКИ тензоров в apply_nms_and_filter_generic
            final_pred_boxes_yxyx_viz, final_pred_scores_viz, \
                final_pred_classes_viz, num_valid_dets_viz = apply_nms_and_filter_generic(
                all_decoded_boxes_xywh_norm_list=decoded_boxes_per_level_viz,  # Список [ (B,Gh,Gw,A,4), ... ]
                all_obj_confidence_list=obj_conf_per_level_viz,  # Список [ (B,Gh,Gw,A,1), ... ]
                all_class_probs_list=class_probs_per_level_viz,  # Список [ (B,Gh,Gw,A,NumCls), ... ]
                num_output_classes=NUM_CLASSES_DETECTOR,  # Имя аргумента из твоей функции
                confidence_threshold_nms=CONF_THRESH_VIZ,  # Имя аргумента из твоей функции
                iou_threshold_nms=IOU_THRESH_NMS_VIZ,  # Имя аргумента из твоей функции
                max_total_detections_nms=MAX_DETS_VIZ  # Имя аргумента из твоей функции
            )
            num_valid_dets_np_viz = num_valid_dets_viz[0].numpy()

            print(
                f"    После NMS (conf={CONF_THRESH_VIZ:.2f}, iou={IOU_THRESH_NMS_VIZ:.2f}) найдено {num_valid_dets_np_viz} объектов для визуализации.")

            visualize_fpn_detections_vs_gt(
                image_processed_np,
                y_true_fpn_tuple_np_for_loss,
                final_pred_boxes_yxyx_viz[0][:num_valid_dets_np_viz].numpy(),
                final_pred_scores_viz[0][:num_valid_dets_np_viz].numpy(),
                final_pred_classes_viz[0][:num_valid_dets_np_viz].numpy().astype(int),
                FPN_LEVEL_NAMES_ORDERED,
                FPN_LEVELS_CONFIG_GLOBAL,
                CLASSES_LIST_GLOBAL_FOR_DETECTOR,
                original_gt_boxes_for_reference=original_gt_for_reference_viz,
                title_prefix=f"Iter {iteration + 1} Preds: {os.path.basename(DEBUG_IMAGE_PATH_SELECTED)}",
                show_grid_for_level="ALL"
            )
            plt.close('all')

    if iteration == NUM_ITERATIONS_DEBUG - 1:
        print("\n--- Проверка изменения весов в конце ---")
        check_weights_changed(model, initial_trainable_weights, trainable_variable_names)
    print("\n--- Отладочное обучение на одном примере завершено. ---")


if __name__ == '__main__':
    main_debug_train_single_random_defective_image()