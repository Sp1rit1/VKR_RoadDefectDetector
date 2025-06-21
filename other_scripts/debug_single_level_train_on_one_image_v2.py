# debug_single_level_train_on_one_image_v2.py
import tensorflow as tf
import yaml
import os
import sys
import numpy as np
from pathlib import Path
import time
import random
import cv2
from PIL import Image as PILImage  # Импортируем для get_debug_data_v2 и, возможно, здесь

# --- Настройка sys.path ---
_project_root_debug_v2 = Path(__file__).resolve().parent
_src_path_debug_v2 = _project_root_debug_v2 / 'src'
if str(_src_path_debug_v2) not in sys.path:
    sys.path.insert(0, str(_src_path_debug_v2))

# --- Импорты из твоих V2 модулей ---
from datasets.other_loaders.detector_data_loader_single_level_v2 import (
    create_detector_single_level_v2_tf_dataset,
    TARGET_IMG_HEIGHT_SDL_V2_G, TARGET_IMG_WIDTH_SDL_V2_G,
    SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G, CLASSES_LIST_SDL_V2_G,
    ANCHORS_WH_P4_DEBUG_SDL_V2_G,
    GRID_H_P4_DEBUG_SDL_V2_G, GRID_W_P4_DEBUG_SDL_V2_G,
    NUM_CLASSES_SDL_V2_G, _p4_debug_stride_cfg_v2 as SL_STRIDE_DEBUG_G,
    NUM_ANCHORS_P4_DEBUG_SDL_V2_G,
    parse_xml_annotation  # <<<--- ИМПОРТИРУЕМ parse_xml_annotation
)

random.seed(None)
from models.other_models.object_detector_single_level_v2 import build_object_detector_single_level_v2
from losses.other_losses.detection_losses_single_level_v2 import compute_detector_loss_single_level_v2
from datasets.other_loaders.detector_data_loader_single_level_v2 import preprocess_image_and_boxes

_matplotlib_and_plot_utils_available_for_this_script = False
_visualize_gt_debug_func = lambda *args, **kwargs: print("ПРЕДУПРЕЖДЕНИЕ: plot_utils_v2/matplotlib не для GT.")
_draw_predictions_imported_func = lambda *args, **kwargs: print(
    "ПРЕДУПРЕЖДЕНИЕ: plot_utils_v2.draw_detections_on_image_single_level не импортирована.")
try:
    import matplotlib.pyplot as plt
    from utils.other_utils.plot_utils_v2 import visualize_single_level_gt_assignments as viz_func_imported_debug
    from utils.other_utils.plot_utils_v2 import draw_detections_on_image_single_level as draw_preds_func_imported_debug

    _visualize_gt_debug_func = viz_func_imported_debug
    _draw_predictions_imported_func = draw_preds_func_imported_debug
    _matplotlib_and_plot_utils_available_for_this_script = True
    print("INFO (debug_train_v2): matplotlib и функции из plot_utils_v2 успешно импортированы.")
except ImportError:
    pass
except Exception:
    pass


# Встроенные decode_predictions_for_debug и apply_nms_for_debug (как в предыдущем ответе)
def decode_predictions_for_debug(raw_predictions_tensor, anchors_wh_norm_arg, grid_h_arg, grid_w_arg, num_classes_arg,
                                 stride_val_arg):  # ... (код) ...
    batch_size = tf.shape(raw_predictions_tensor)[0];
    pred_xy_raw = raw_predictions_tensor[..., 0:2];
    pred_wh_raw = raw_predictions_tensor[..., 2:4];
    pred_obj_logit = raw_predictions_tensor[..., 4:5];
    pred_class_logits = raw_predictions_tensor[..., 5:];
    gy_indices = tf.tile(tf.range(grid_h_arg, dtype=tf.float32)[:, tf.newaxis], [1, grid_w_arg]);
    gx_indices = tf.tile(tf.range(grid_w_arg, dtype=tf.float32)[tf.newaxis, :], [grid_h_arg, 1]);
    grid_coords_xy = tf.stack([gx_indices, gy_indices], axis=-1);
    grid_coords_xy = grid_coords_xy[tf.newaxis, :, :, tf.newaxis, :];
    grid_coords_xy = tf.tile(grid_coords_xy, [batch_size, 1, 1, tf.shape(anchors_wh_norm_arg)[0], 1]);
    pred_xy_on_grid = (tf.sigmoid(pred_xy_raw) + grid_coords_xy);
    pred_xy_normalized = pred_xy_on_grid / tf.constant([grid_w_arg, grid_h_arg], dtype=tf.float32);
    anchors_tensor_arg = tf.constant(anchors_wh_norm_arg, dtype=tf.float32);
    anchors_reshaped_arg = anchors_tensor_arg[tf.newaxis, tf.newaxis, tf.newaxis, :, :];
    pred_wh_normalized = (tf.exp(pred_wh_raw) * anchors_reshaped_arg);
    decoded_boxes_xywh_norm = tf.concat([pred_xy_normalized, pred_wh_normalized], axis=-1);
    pred_obj_confidence = tf.sigmoid(pred_obj_logit);
    pred_class_probs = tf.sigmoid(pred_class_logits);
    return decoded_boxes_xywh_norm, pred_obj_confidence, pred_class_probs


def apply_nms_for_debug(decoded_boxes_xywh_norm_batch, obj_confidence_batch, class_probs_batch,
                        confidence_threshold_arg, iou_threshold_arg, max_detections_arg):  # ... (код) ...
    batch_size = tf.shape(decoded_boxes_xywh_norm_batch)[0];
    gh_nms, gw_nms, na_nms = tf.shape(decoded_boxes_xywh_norm_batch)[1], tf.shape(decoded_boxes_xywh_norm_batch)[2], \
    tf.shape(decoded_boxes_xywh_norm_batch)[3];
    nc_nms = tf.shape(class_probs_batch)[-1];
    num_total_boxes_per_image_nms = gh_nms * gw_nms * na_nms;
    boxes_flat_xywh = tf.reshape(decoded_boxes_xywh_norm_batch, [batch_size, num_total_boxes_per_image_nms, 4]);
    obj_conf_flat = tf.reshape(obj_confidence_batch, [batch_size, num_total_boxes_per_image_nms, 1]);
    class_probs_flat = tf.reshape(class_probs_batch, [batch_size, num_total_boxes_per_image_nms, nc_nms]);
    final_scores_per_class = obj_conf_flat * class_probs_flat;
    boxes_ymin_xmin_ymax_xmax = tf.concat([boxes_flat_xywh[..., 1:2] - boxes_flat_xywh[..., 3:4] / 2.0,
                                           boxes_flat_xywh[..., 0:1] - boxes_flat_xywh[..., 2:3] / 2.0,
                                           boxes_flat_xywh[..., 1:2] + boxes_flat_xywh[..., 3:4] / 2.0,
                                           boxes_flat_xywh[..., 0:1] + boxes_flat_xywh[..., 2:3] / 2.0], axis=-1);
    boxes_for_nms = tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2);
    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms, scores=final_scores_per_class,
        max_output_size_per_class=max(1, max_detections_arg // nc_nms if nc_nms > 0 else max_detections_arg),
        max_total_size=max_detections_arg, iou_threshold=iou_threshold_arg, score_threshold=confidence_threshold_arg,
        clip_boxes=False);
    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


# --- Загрузка Конфигураций ---
# ... (код загрузки BASE_CONFIG_DEBUG_V2 и DETECTOR_CONFIG_DEBUG_V2 как был) ...
_base_config_path_debug_v2 = _src_path_debug_v2 / 'configs' / 'base_config.yaml';
_detector_config_path_debug_v2 = _src_path_debug_v2 / 'configs' / 'detector_config_single_level_v2.yaml'
BASE_CONFIG_DEBUG_V2 = {};
DETECTOR_CONFIG_DEBUG_V2 = {};
_config_load_error = False
try:
    with open(_base_config_path_debug_v2, 'r', encoding='utf-8') as f:
        BASE_CONFIG_DEBUG_V2 = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG_DEBUG_V2,
                      dict) or not BASE_CONFIG_DEBUG_V2: BASE_CONFIG_DEBUG_V2 = {}; _config_load_error = True
    with open(_detector_config_path_debug_v2, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_DEBUG_V2 = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_DEBUG_V2,
                      dict) or not DETECTOR_CONFIG_DEBUG_V2: DETECTOR_CONFIG_DEBUG_V2 = {}; _config_load_error = True
except Exception as e:
    _config_load_error = True; print(f"ОШИБКА загрузки конфигов: {e}")
if _config_load_error or not BASE_CONFIG_DEBUG_V2 or not DETECTOR_CONFIG_DEBUG_V2:
    print("ОШИБКА: Конфиги не загружены.");
    DETECTOR_CONFIG_DEBUG_V2.setdefault('fpn_detector_params', {}).setdefault('input_shape', [416, 416, 3]);
    DETECTOR_CONFIG_DEBUG_V2.get('fpn_detector_params', {}).setdefault('classes', ['pit', 'crack']);
    DETECTOR_CONFIG_DEBUG_V2.setdefault('train_params', {});
    BASE_CONFIG_DEBUG_V2.setdefault('dataset', {})

# --- Параметры для отладочного запуска ---
# ... (как было, DEBUG_NUM_EXAMPLES_TO_OVERFIT и т.д.) ...
_train_params_debug_cfg = DETECTOR_CONFIG_DEBUG_V2.get('train_params', {});
_sl_params_debug = DETECTOR_CONFIG_DEBUG_V2.get('single_level_detector_params',
                                                DETECTOR_CONFIG_DEBUG_V2.get('fpn_detector_params', {}))
DEBUG_NUM_EXAMPLES_TO_OVERFIT = _train_params_debug_cfg.get('debug_num_examples_overfit', 1);
DEBUG_BATCH_SIZE = DEBUG_NUM_EXAMPLES_TO_OVERFIT
DEBUG_EPOCHS = _train_params_debug_cfg.get('debug_epochs_overfit', 200);
DEBUG_LEARNING_RATE = _train_params_debug_cfg.get('debug_lr_overfit', 0.001)
DEBUG_USE_AUGMENTATION = False;
DEBUG_PLOT_EVERY_N_EPOCHS = _train_params_debug_cfg.get('debug_plot_every_n_epochs', 25)
DEBUG_FREEZE_BACKBONE_IN_DEBUG_LOOP = DETECTOR_CONFIG_DEBUG_V2.get('freeze_backbone', True)
ENABLE_VISUALIZATION_DEBUG_SCRIPT = DETECTOR_CONFIG_DEBUG_V2.get('debug_callback_enable_visualization', True)
os.environ['DEBUG_DETECTOR_LOSS_V2'] = '1'
_predict_config_path_debug = _src_path_debug_v2 / 'configs' / 'predict_config.yaml';
PREDICT_CONFIG_DEBUG = {}
try:
    with open(_predict_config_path_debug, 'r', encoding='utf-8') as f:
        PREDICT_CONFIG_DEBUG = yaml.safe_load(f)
    if not isinstance(PREDICT_CONFIG_DEBUG, dict): PREDICT_CONFIG_DEBUG = {}
except Exception:
    PREDICT_CONFIG_DEBUG = {}
NMS_CONF_THRESH_DEBUG = PREDICT_CONFIG_DEBUG.get('default_conf_thresh', 0.05);
NMS_IOU_THRESH_DEBUG = PREDICT_CONFIG_DEBUG.get('default_iou_thresh', 0.45);
NMS_MAX_DETS_DEBUG = PREDICT_CONFIG_DEBUG.get('default_max_dets', 20)


# get_debug_data_v2 (без изменений)
def get_debug_data_v2(num_samples, project_root_path, base_cfg, detector_cfg_for_paths):
    # ... (твой код get_debug_data_v2, с исправленным print'ом p_ret_log_main_func_val) ...
    prepared_path_key = 'prepared_dataset_path_detector_ready';
    detector_ready_path_rel = detector_cfg_for_paths.get(prepared_path_key,
                                                         detector_cfg_for_paths.get('fpn_detector_params', {}).get(
                                                             prepared_path_key, 'data/Detector_Dataset_Ready'));
    detector_ready_path_abs = project_root_path / detector_ready_path_rel
    images_subdir_name = base_cfg.get('dataset', {}).get('images_dir', 'JPEGImages');
    ann_subdir_name = base_cfg.get('dataset', {}).get('annotations_dir', 'Annotations')
    train_images_dir = detector_ready_path_abs / "train" / images_subdir_name;
    train_ann_dir = detector_ready_path_abs / "train" / ann_subdir_name
    if not train_images_dir.is_dir() or not train_ann_dir.is_dir(): print(
        f"ОШИБКА (get_debug_data_v2): Директории не найдены."); return None, None
    all_img_paths_temp = [];
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'];
    [all_img_paths_temp.extend(list(train_images_dir.glob(ext_pat))) for ext_pat in valid_extensions]
    seen_stems_for_debug = set();
    unique_image_paths_for_debug = [];
    [unique_image_paths_for_debug.append(
        r_p) if r_p.stem.lower() not in seen_stems_for_debug and not seen_stems_for_debug.add(
        r_p.stem.lower()) else None for r_p in [f_p.resolve() for f_p in all_img_paths_temp]]
    final_image_paths_debug = [];
    final_xml_paths_debug = []
    for img_p_debug in unique_image_paths_for_debug:
        xml_p_debug = train_ann_dir / (img_p_debug.stem + ".xml")
        if xml_p_debug.exists(): final_image_paths_debug.append(str(img_p_debug)); final_xml_paths_debug.append(
            str(xml_p_debug))
    if not final_image_paths_debug: print("ОШИБКА (get_debug_data_v2): Нет валидных пар."); return None, None
    paired_files_debug = list(zip(final_image_paths_debug, final_xml_paths_debug));
    random.shuffle(paired_files_debug)
    num_to_take_debug = min(num_samples, len(paired_files_debug));
    if num_to_take_debug == 0: print("ОШИБКА (get_debug_data_v2): Недостаточно данных."); return None, None
    selected_pairs_debug = paired_files_debug[:num_to_take_debug];
    debug_img_paths_ret = [p[0] for p in selected_pairs_debug];
    debug_xml_paths_ret = [p[1] for p in selected_pairs_debug]
    print(f"Для отладки выбрано {len(debug_img_paths_ret)} изображений:");
    for p_ret_log_main_func_val_v3 in debug_img_paths_ret: print(f"  - {os.path.basename(p_ret_log_main_func_val_v3)}")
    return debug_img_paths_ret, debug_xml_paths_ret


# Конец get_debug_data_v2

def main_debug_train_v2():
    print("--- Отладочный Запуск Обучения Одноуровневой Модели V2 ---")
    print(
        f"  Параметры: Примеров={DEBUG_NUM_EXAMPLES_TO_OVERFIT}, Эпох={DEBUG_EPOCHS}, LR={DEBUG_LEARNING_RATE}, Аугм={DEBUG_USE_AUGMENTATION}")
    should_visualize_this_run = ENABLE_VISUALIZATION_DEBUG_SCRIPT and _matplotlib_and_plot_utils_available_for_this_script
    print(f"  Визуализация GT перед обучением (если включена): {should_visualize_this_run}")
    print(
        f"  Визуализация Предсказаний каждые {DEBUG_PLOT_EVERY_N_EPOCHS} эпох (если включена): {should_visualize_this_run}")
    print(f"  Backbone будет заморожен: {DEBUG_FREEZE_BACKBONE_IN_DEBUG_LOOP}")

    # --- 1. Получаем ПУТИ к отладочным данным ---
    debug_image_paths_list, debug_xml_paths_list = get_debug_data_v2(
        DEBUG_NUM_EXAMPLES_TO_OVERFIT, _project_root_debug_v2,
        BASE_CONFIG_DEBUG_V2, DETECTOR_CONFIG_DEBUG_V2
    )
    if not debug_image_paths_list: return

    # --- Предварительная визуализация GT (Вариант Б) ---
    if should_visualize_this_run and debug_image_paths_list:
        print(f"\nПредварительная визуализация GT для: {os.path.basename(debug_image_paths_list[0])}")
        try:
            # Загружаем оригинальное изображение
            pil_img_orig = PILImage.open(debug_image_paths_list[0]).convert('RGB')
            img_np_orig_uint8_for_viz = np.array(pil_img_orig, dtype=np.uint8)

            # Парсим XML
            gt_objects_for_viz, xml_w_viz, xml_h_viz, _ = parse_xml_annotation(debug_xml_paths_list[0],
                                                                               CLASSES_LIST_SDL_V2_G)

            original_gt_boxes_pixels = []
            original_gt_class_ids = []
            if gt_objects_for_viz:
                for obj in gt_objects_for_viz:
                    original_gt_boxes_pixels.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
                    original_gt_class_ids.append(obj['class_id'])

            # Нужны нормализованные к TARGET_SIZE для отрисовки красных рамок
            # Для этого нужно применить preprocess_image_and_boxes к оригинальному изображению и пиксельным GT
            # (немного дублируем логику из data_loader, но это для автономности этой визуализации)
            img_tensor_for_viz_gt = tf.convert_to_tensor(img_np_orig_uint8_for_viz.astype(np.float32), dtype=tf.float32)
            boxes_tensor_pixels_for_viz_gt = tf.constant(original_gt_boxes_pixels,
                                                         dtype=tf.float32) if original_gt_boxes_pixels else tf.zeros(
                (0, 4), dtype=tf.float32)

            # Получаем обработанное изображение (416x416, [0,1]) и масштабированные GT рамки
            img_processed_for_gt_viz, scaled_gt_boxes_norm_ref = preprocess_image_and_boxes(
                img_tensor_for_viz_gt, boxes_tensor_pixels_for_viz_gt,
                tf.constant(TARGET_IMG_HEIGHT_SDL_V2_G, dtype=tf.int32),
                tf.constant(TARGET_IMG_WIDTH_SDL_V2_G, dtype=tf.int32)
            )

            # Создаем "пустой" y_true, так как _visualize_gt_debug_func его ожидает для рисования зеленых
            # (но зеленых не будет, так как нет назначенных якорей без полного пайплайна data_loader'а)
            # Вместо этого, мы могли бы передать scaled_gt_boxes_norm_ref дважды,
            # но лучше адаптировать _visualize_gt_debug_func или создать новую функцию только для GT.
            # Пока что передадим пустой y_true.
            temp_y_true_shape = (
            GRID_H_P4_DEBUG_SDL_V2_G, GRID_W_P4_DEBUG_SDL_V2_G, NUM_ANCHORS_P4_DEBUG_SDL_V2_G, 5 + NUM_CLASSES_SDL_V2_G)
            empty_y_true_for_gt_viz = np.zeros(temp_y_true_shape, dtype=np.float32)
            empty_y_true_for_gt_viz[..., 4] = 0.0  # Все фон

            _visualize_gt_debug_func(
                img_processed_for_gt_viz.numpy(),  # Обработанное изображение
                empty_y_true_for_gt_viz,  # Пустой y_true (зеленых не будет)
                level_config_for_drawing=SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_V2_G,
                classes_list_for_drawing=CLASSES_LIST_SDL_V2_G,
                original_gt_boxes_for_ref=(
                scaled_gt_boxes_norm_ref.numpy(), np.array(original_gt_class_ids, dtype=np.int32)),
                title_prefix=f"Initial Original GT for {os.path.basename(debug_image_paths_list[0])}"
            )
            if plt and _matplotlib_and_plot_utils_available_for_this_script: plt.close('all')
        except Exception as e_pre_viz_main:
            print(f"ОШИБКА при предварительной визуализации GT: {e_pre_viz_main}")
            import traceback;
            traceback.print_exc()
    # --- Конец Предварительной визуализации GT ---

    print("\nСоздание отладочного TensorFlow датасета для обучения...")
    # create_detector_single_level_v2_tf_dataset теперь возвращает (img, y_true)
    debug_dataset = create_detector_single_level_v2_tf_dataset(
        debug_image_paths_list,  # Переименовал для ясности
        debug_xml_paths_list,  # Переименовал для ясности
        batch_size_arg=DEBUG_BATCH_SIZE,
        shuffle_arg=False,
        augment_arg=DEBUG_USE_AUGMENTATION
    )
    if debug_dataset is None: print("Не удалось создать отладочный датасет. Выход."); return
    debug_dataset = debug_dataset.repeat()

    print("\nСоздание модели детектора...")
    model = build_object_detector_single_level_v2(
        force_initial_freeze_backbone_arg=DEBUG_FREEZE_BACKBONE_IN_DEBUG_LOOP
    )
    model.summary(line_length=120, show_trainable=True)
    print(f"\nКомпиляция модели с LR = {DEBUG_LEARNING_RATE}...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=DEBUG_LEARNING_RATE),
                  loss=compute_detector_loss_single_level_v2)

    # --- Отслеживание весов ---
    # ... (код для tracked_initial_vars как был, с initial_weights_to_track_by_index) ...
    initial_weights_to_track_by_index = {};
    variables_to_log_info_at_start = []
    num_trainable_vars_total_main_v3 = len(model.trainable_variables);
    print(f"  Всего обучаемых переменных: {num_trainable_vars_total_main_v3}")
    indices_to_track_main_v3 = [];
    if num_trainable_vars_total_main_v3 > 0:
        if num_trainable_vars_total_main_v3 <= 9:
            indices_to_track_main_v3 = list(range(num_trainable_vars_total_main_v3))
        else:
            indices_to_track_main_v3.extend(list(range(min(3, num_trainable_vars_total_main_v3))));
            if num_trainable_vars_total_main_v3 > 6: mid_start_main_v3 = num_trainable_vars_total_main_v3 // 2 - 1; mid_end_main_v3 = mid_start_main_v3 + 3; indices_to_track_main_v3.extend(
                list(range(mid_start_main_v3, min(mid_end_main_v3, num_trainable_vars_total_main_v3))))
            indices_to_track_main_v3.extend(
                list(range(max(0, num_trainable_vars_total_main_v3 - 3), num_trainable_vars_total_main_v3)))
        indices_to_track_main_v3 = sorted(list(set(indices_to_track_main_v3)))
        for i_var_track_init_main_v3 in indices_to_track_main_v3:
            if i_var_track_init_main_v3 < num_trainable_vars_total_main_v3: var_item_init_main_v3 = \
            model.trainable_variables[i_var_track_init_main_v3]; initial_weights_to_track_by_index[
                i_var_track_init_main_v3] = var_item_init_main_v3.numpy().copy(); variables_to_log_info_at_start.append(
                {'name': var_item_init_main_v3.name, 'index': i_var_track_init_main_v3,
                 'shape': var_item_init_main_v3.shape})
    if variables_to_log_info_at_start:
        print(f"  Будут отслеживаться {len(variables_to_log_info_at_start)} переменных:"); [
            print(f"    - [{v_log_item['index']}] {v_log_item['name']} ({v_log_item['shape']})") for v_log_item in
            variables_to_log_info_at_start]
    else:
        print("  ПРЕДУПРЕЖДЕНИЕ: Не выбрано переменных для отслеживания.")

    print(f"\nНачало отладочного обучения на {DEBUG_EPOCHS} эпох...")
    for epoch in range(DEBUG_EPOCHS):
        print(f"--- Эпоха {epoch + 1}/{DEBUG_EPOCHS} ---", end="")
        epoch_losses_list_main_v2 = []

        # Теперь debug_dataset возвращает (images_batch, y_true_batch)
        for step, (images_batch_train, y_true_batch_train) in enumerate(debug_dataset.take(1)):
            start_time_step_loop = time.time()
            with tf.GradientTape() as tape:
                y_pred_batch_train = model(images_batch_train, training=True)
                loss_value_or_dict_loop = model.loss(y_true_batch_train, y_pred_batch_train)
                total_loss_val_loop = loss_value_or_dict_loop.get('total_loss_debug') if isinstance(
                    loss_value_or_dict_loop, dict) else loss_value_or_dict_loop
            # ... (обновление весов, лог шага - как было) ...
            grads_loop = tape.gradient(total_loss_val_loop, model.trainable_variables);
            if model.trainable_variables and not any(g is None for g in grads_loop):
                model.optimizer.apply_gradients(zip(grads_loop, model.trainable_variables))
            elif model.trainable_variables:
                print(f"\nОШИБКА: None градиенты на эпохе {epoch + 1}.")
            step_duration_loop = time.time() - start_time_step_loop;
            epoch_losses_list_main_v2.append(total_loss_val_loop.numpy())
            log_str_item_main_loop_v3 = f" Шаг {step + 1}: "
            if isinstance(loss_value_or_dict_loop, dict):
                # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                for lname_v3, lval_v3 in loss_value_or_dict_loop.items():  # Обычный цикл for
                    component_name_v3 = lname_v3.replace('_debug', '').replace('total_', '').capitalize()
                    log_str_item_main_loop_v3 += f"{component_name_v3}={lval_v3.numpy():.4f} "
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            else:  # Если вернулся только скаляр total_loss
                log_str_item_main_loop_v3 += f"TotalLoss={total_loss_val_loop.numpy():.4f} "

            print(f"\r{log_str_item_main_loop_v3}Время: {step_duration_loop:.2f}с", end="")

            # --- ВИЗУАЛИЗАЦИЯ ---
            if should_visualize_this_run and \
                    (epoch == 0 or (epoch + 1) % DEBUG_PLOT_EVERY_N_EPOCHS == 0 or epoch == DEBUG_EPOCHS - 1):
                if epoch > 0 or step > 0: print()
                print(f"    Визуализация для эпохи {epoch + 1}...")

                # Для визуализации GT Assignments, мы уже показали один раз до цикла.
                # Если нужно показывать для текущего батча обучения:
                # _visualize_gt_debug_func(images_batch_train[0].numpy(), y_true_batch_train[0].numpy(), ...)

                # Визуализация Предсказаний Модели
                try:
                    decoded_boxes_xywh_p, obj_conf_p, class_probs_p = \
                        decode_predictions_for_debug(y_pred_batch_train, ANCHORS_WH_P4_DEBUG_SDL_V2_G,
                                                     GRID_H_P4_DEBUG_SDL_V2_G, GRID_W_P4_DEBUG_SDL_V2_G,
                                                     NUM_CLASSES_SDL_V2_G, SL_STRIDE_DEBUG_G)
                    final_boxes_n_p, final_scores_p, final_classes_ids_p, num_valid_dets_p = \
                        apply_nms_for_debug(decoded_boxes_xywh_p[0:1], obj_conf_p[0:1], class_probs_p[0:1],
                                            NMS_CONF_THRESH_DEBUG, NMS_IOU_THRESH_DEBUG, NMS_MAX_DETS_DEBUG)
                    num_preds_val_viz = int(num_valid_dets_p[0].numpy());
                    print(f"      Найдено {num_preds_val_viz} предсказаний после NMS.")
                    image_to_draw_on_preds_viz = (images_batch_train[0].numpy() * 255).astype(
                        np.uint8)  # Используем изображение из текущего батча
                    if num_preds_val_viz > 0:
                        img_with_preds_viz = _draw_predictions_imported_func(image_to_draw_on_preds_viz,
                                                                             final_boxes_n_p[0][
                                                                             :num_preds_val_viz].numpy(),
                                                                             final_scores_p[0][
                                                                             :num_preds_val_viz].numpy(),
                                                                             final_classes_ids_p[0][
                                                                             :num_preds_val_viz].numpy(),
                                                                             CLASSES_LIST_SDL_V2_G,
                                                                             TARGET_IMG_WIDTH_SDL_V2_G,
                                                                             TARGET_IMG_HEIGHT_SDL_V2_G)
                    else:
                        img_with_preds_viz = cv2.cvtColor(image_to_draw_on_preds_viz, cv2.COLOR_RGB2BGR); cv2.putText(
                            img_with_preds_viz, "No Detections", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                    if _matplotlib_and_plot_utils_available_for_this_script:
                        plt.figure(figsize=(8, 8));
                        plt.imshow(cv2.cvtColor(img_with_preds_viz, cv2.COLOR_BGR2RGB));
                        plt.title(f"DebugV2 E{epoch + 1} - Preds (Conf:{NMS_CONF_THRESH_DEBUG:.2f})");
                        plt.axis('off');
                        plt.show()
                    else:
                        cv2.imwrite(f"dbg_pred_e{epoch + 1}.png", img_with_preds_viz); print(
                            f"  Предсказания сохранены: dbg_pred_e{epoch + 1}.png")
                except Exception as e_viz_pred_main:
                    print(f"      ОШИБКА виз. предсказаний: {e_viz_pred_main}"); import traceback; traceback.print_exc()
                if plt and _matplotlib_and_plot_utils_available_for_this_script: plt.close('all')

        avg_epoch_loss_val = np.mean(epoch_losses_list_main_v2)
        print(f"\r{' ' * 150}\r--- Эпоха {epoch + 1}/{DEBUG_EPOCHS} --- Средний Loss: {avg_epoch_loss_val:.6f}")
        if avg_epoch_loss_val < 0.005 and epoch > 10: print(f"Loss < 0.005. Прерываем."); break

    print("\n--- Отладочное обучение завершено ---")
    # ... (исправленный блок проверки изменения весов как в предыдущем ответе) ...
    print("\nПроверка изменения весов:");
    changed_count_final_main_v3 = 0;
    not_changed_count_final_main_v3 = 0
    if not initial_weights_to_track_by_index:
        print("  Переменные не отслеживались.")
    else:
        for var_idx_tracked_main_v3, initial_val_tracked_main_v3 in initial_weights_to_track_by_index.items():
            if var_idx_tracked_main_v3 < len(model.trainable_variables):
                current_var_tensor_tracked_main_v3 = model.trainable_variables[var_idx_tracked_main_v3];
                var_name_for_log_main_v3 = current_var_tensor_tracked_main_v3.name;
                current_val_tracked_main_v3 = current_var_tensor_tracked_main_v3.numpy()
                if not np.array_equal(initial_val_tracked_main_v3, current_val_tracked_main_v3):
                    changed_count_final_main_v3 += 1; diff_sum_main_val_v3 = np.sum(
                        np.abs(initial_val_tracked_main_v3 - current_val_tracked_main_v3)); print(
                        f"  '{var_name_for_log_main_v3}' ({current_val_tracked_main_v3.shape}) ИЗМЕНИЛСЯ. СуммаРазн={diff_sum_main_val_v3:.3e}, МаксРазн={np.max(np.abs(initial_val_tracked_main_v3 - current_val_tracked_main_v3)):.3e}")
                else:
                    not_changed_count_final_main_v3 += 1; print(
                        f"  '{var_name_for_log_main_v3}' ({current_val_tracked_main_v3.shape}) НЕ изменился.")
            else:
                print(
                    f"  КРИТ. ПРЕДУПРЕЖДЕНИЕ: Индекс {var_idx_tracked_main_v3} вне диапазона."); not_changed_count_final_main_v3 += 1
    print(f"\nИтог: {changed_count_final_main_v3} изм., {not_changed_count_final_main_v3} не изм. (из отсл.).");
    if changed_count_final_main_v3 > 0:
        print("ОТЛИЧНО! Веса обновляются.")
    elif not model.trainable_variables and num_trainable_vars_total_main_v3 == 0:
        print("ИНФОРМАЦИЯ: В модели нет обучаемых переменных.")
    elif not initial_weights_to_track_by_index and num_trainable_vars_total_main_v3 > 0:
        print("ПРЕДУПРЕЖДЕНИЕ: Не удалось выбрать переменные для отслеживания.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Отслеживаемые веса не обновились.")


if __name__ == '__main__':
    main_debug_train_v2()