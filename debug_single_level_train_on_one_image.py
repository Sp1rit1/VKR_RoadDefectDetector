# debug_single_level_train_on_FIXED_image.py
import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

# --- Настройка sys.path для импорта из src ---
_project_root_debug_sl_fixed = Path(__file__).resolve().parent
_src_path_debug_sl_fixed = _project_root_debug_sl_fixed / 'src'
if str(_src_path_debug_sl_fixed) not in sys.path:
    sys.path.insert(0, str(_src_path_debug_sl_fixed))
if str(_project_root_debug_sl_fixed) not in sys.path:
    sys.path.insert(0, str(_project_root_debug_sl_fixed))

# --- Импорты из твоих ОТЛАДОЧНЫХ (_single_level_debug) модулей ---
try:
    from datasets.detector_data_loader_single_level_debug import (
        create_detector_single_level_tf_dataset,  # Функция для создания датасета для одного уровня
        # Глобальные переменные, определенные в detector_data_loader_single_level_debug.py
        # на основе detector_config_single_level_debug.yaml
        TARGET_IMG_HEIGHT_SDL_G, TARGET_IMG_WIDTH_SDL_G,
        CLASSES_LIST_SDL_G, NUM_CLASSES_SDL_G,
        GRID_H_P4_DEBUG_SDL_G, GRID_W_P4_DEBUG_SDL_G,  # Параметры для P4_debug
        NUM_ANCHORS_P4_DEBUG_SDL_G, ANCHORS_WH_P4_DEBUG_SDL_G,  # Якоря для P4_debug
        FPN_LEVEL_NAME_DEBUG_SDL_G, SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G  # Имя нашего единственного отладочного уровня (например, 'P4_debug')
    )
    from models.object_detector_single_level_debug import build_detector_single_level_p4_debug
    from losses.detection_losses_single_level_debug import compute_detector_loss_single_level_debug

    print("INFO (debug_sl_fixed_script): Отладочные компоненты (data_loader, model, loss) успешно импортированы.")
except ImportError as e_imp_main_sdl_fixed:
    print(
        f"КРИТИЧЕСКАЯ ОШИБКА (debug_sl_fixed_script): Не удалось импортировать отладочные компоненты: {e_imp_main_sdl_fixed}")
    import traceback

    traceback.print_exc()
    exit()

_plot_utils_imported_sdl_fixed = False
try:
    # Для визуализации нам все еще нужна функция, которая может рисовать один уровень.
    # visualize_fpn_detections_vs_gt может быть адаптирована или нам нужна более простая.
    # Давай предположим, что visualize_fpn_detections_vs_gt может принять y_true_tuple из одного элемента.
    from utils.plot_utils import visualize_fpn_detections_vs_gt  # Используем ее, адаптируя вызов

    _plot_utils_imported_sdl_fixed = True
    print("INFO (debug_sl_fixed_script): Функция visualize_fpn_detections_vs_gt импортирована.")
except ImportError as e_imp_plot_utils_sdl_fixed:
    print(
        f"ПРЕДУПРЕЖДЕНИЕ (debug_sl_fixed_script): Не удалось импортировать visualize_fpn_detections_vs_gt: {e_imp_plot_utils_sdl_fixed}")

# --- Загрузка ОТЛАДОЧНОГО detector_config_single_level_debug.yaml ---
# (Эта загрузка уже происходит внутри импортированных модулей, но для параметров цикла обучения возьмем их оттуда)
# Мы будем использовать переменные, которые УЖЕ загружены и обработаны в detector_data_loader_single_level_debug.py
# Например, DEBUG_SDL_CONFIG_GLOBAL из этого модуля.
# Для этого нам нужно импортировать его:
try:
    from datasets.detector_data_loader_single_level_debug import DEBUG_SDL_CONFIG_GLOBAL as DEBUG_CONFIG_FROM_LOADER
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать DEBUG_SDL_CONFIG_GLOBAL из отладочного загрузчика.")
    DEBUG_CONFIG_FROM_LOADER = {}  # Заглушка
    # exit() # Можно и выйти, если он критичен

# --- Константы для отладки из загруженного отладочного конфига ---
_fpn_params_from_debug_cfg = DEBUG_CONFIG_FROM_LOADER.get('fpn_detector_params', {})
LEARNING_RATE_DEBUG_SL_FIXED = DEBUG_CONFIG_FROM_LOADER.get('initial_learning_rate', 0.0001)
NUM_ITERATIONS_DEBUG_SL_FIXED = DEBUG_CONFIG_FROM_LOADER.get('epochs_for_debug', 200)

_predict_params_from_debug_cfg = DEBUG_CONFIG_FROM_LOADER.get('predict_params', {})
CONF_THRESH_VIZ_SL_FIXED = _predict_params_from_debug_cfg.get('confidence_threshold', 0.05)
IOU_THRESH_NMS_VIZ_SL_FIXED = _predict_params_from_debug_cfg.get('iou_threshold', 0.3)
MAX_DETS_VIZ_SL_FIXED = _predict_params_from_debug_cfg.get('max_detections', 50)

# Управление частотой визуализации
APPROX_NUM_VISUALIZATIONS_TOTAL_SL = 5
VISUALIZE_EVERY_N_ITERATIONS_SL_FIXED = max(1, NUM_ITERATIONS_DEBUG_SL_FIXED // (
            APPROX_NUM_VISUALIZATIONS_TOTAL_SL + 1e-6))
LOG_LOSS_EVERY_N_ITERATIONS_SL = max(1, NUM_ITERATIONS_DEBUG_SL_FIXED // 10 or 1)

print(f"\n--- Параметры для отладочного запуска ОДНОУРОВНЕВОЙ модели на ФИКСИРОВАННОМ изображении ---")
print(f"  (Параметры LR, Итерации, NMS взяты из detector_config_single_level_debug.yaml)")
print(f"  Learning Rate: {LEARNING_RATE_DEBUG_SL_FIXED}")
print(f"  Количество итераций: {NUM_ITERATIONS_DEBUG_SL_FIXED}")
print(f"  Логирование потерь каждые ~{int(LOG_LOSS_EVERY_N_ITERATIONS_SL)} итераций")
print(f"  Визуализация каждые ~{int(VISUALIZE_EVERY_N_ITERATIONS_SL_FIXED)} итераций")



def check_weights_changed_sl_fixed(model, initial_weights_list, iteration):
    # ... (код функции check_weights_changed_fpn_fixed можно переиспользовать)
    changed_count = 0;
    not_changed_count = 0
    # ... (остальная часть функции)
    print(f"  Проверка изменения весов на итерации {iteration}:")
    num_trainable_vars = len(model.trainable_variables)
    indices_to_check = list(range(min(5, num_trainable_vars)))
    if num_trainable_vars > 10:
        indices_to_check.extend(list(range(num_trainable_vars - 5, num_trainable_vars)))
    indices_to_check = sorted(list(set(indices_to_check)))
    for var_idx in indices_to_check:
        var = model.trainable_variables[var_idx];
        initial_w = initial_weights_list[var_idx];
        current_w = var.numpy()
        if not np.array_equal(initial_w, current_w):
            changed_count += 1;
            abs_diff_sum = np.sum(np.abs(current_w - initial_w))
            # print(f"    Вес [{var_idx}] '{var.name}' ИЗМЕНИЛСЯ. Сумма абс. разницы: {abs_diff_sum:.2e}")
        else:
            not_changed_count += 1
    total_checked = len(indices_to_check)
    if changed_count > 0:
        print(f"  ИТОГ ПРОВЕРКИ ВЕСОВ: {changed_count} из {total_checked} проверенных весов изменились.")
    else:
        print(f"  ИТОГ ПРОВЕРКИ ВЕСОВ: Ни один из {total_checked} проверенных весов НЕ изменился!")
    return changed_count > 0


def main_debug_single_level_train_on_fixed_image_func():
    print("\n--- Отладка ОДНОУРОВНЕВОЙ МОДЕЛИ: Обучение на ОДНОМ ФИКСИРОВАННОМ Изображении ---")
    DEBUG_IMAGE_FILENAME_FIXED = "China_Drone_000180.jpg"  # <<<--- ЗАМЕНИ НА ИМЯ ТВОЕГО ФАЙЛА ИЗОБРАЖЕНИЯ
    DEBUG_XML_FILENAME_FIXED = "China_Drone_000180.xml"  # <<<--- ЗАМЕНИ НА ИМЯ ТВОЕГО XML ФАЙЛА
    debug_data_folder_fixed = Path("C:/Users/0001/Desktop/Diplom/RoadDefectDetector/debug_data")

    fixed_image_path = debug_data_folder_fixed / DEBUG_IMAGE_FILENAME_FIXED
    fixed_xml_path = debug_data_folder_fixed / DEBUG_XML_FILENAME_FIXED

    if not fixed_image_path.exists(): print(f"ОШИБКА: Отладочное изображение не найдено: {fixed_image_path}"); return
    if not fixed_xml_path.exists(): print(f"ОШИБКА: Отладочный XML файл не найден: {fixed_xml_path}"); return
    print(f"Используется для отладки: {fixed_image_path.name}")

    # Создаем датасет из одного этого изображения с помощью ОТЛАДОЧНОГО загрузчика
    debug_dataset_fixed = create_detector_single_level_tf_dataset(
        [str(fixed_image_path)],  # Список из одного элемента
        [str(fixed_xml_path)],  # Список из одного элемента
        batch_size_arg=1,
        shuffle_arg=False,
        augment_arg=False  # Аугментация ОТКЛЮЧЕНА
    )

    if debug_dataset_fixed is None:
        print("ОШИБКА: Не удалось создать отладочный датасет. Выход.")
        return

    debug_dataset_fixed = debug_dataset_fixed.repeat()

    # Получаем один эталонный пример (image, y_true)
    try:
        image_batch_tf_train, y_true_batch_tf_train = next(iter(debug_dataset_fixed))
        image_processed_np_viz = image_batch_tf_train[0].numpy()  # Для визуализации
        y_true_single_level_np_viz = y_true_batch_tf_train[0].numpy()  # Для визуализации GT
    except Exception as e_ds_iter_fixed:
        print(f"ОШИБКА при получении данных из отладочного датасета: {e_ds_iter_fixed}");
        traceback.print_exc();
        return

    # Визуализация исходного GT перед обучением
    if _plot_utils_imported_sdl_fixed:
        print("\nВизуализация Ground Truth для выбранного изображения (до начала обучения):")
        # Адаптируем вызов visualize_fpn_detections_vs_gt для одного уровня
        # Передаем y_true для одного уровня как кортеж из одного элемента
        # и fpn_level_names/fpn_configs только для этого одного уровня
        current_level_name_viz_sl = FPN_LEVEL_NAME_DEBUG_SDL_G  # 'P4_debug'
        current_fpn_configs_viz_sl = {current_level_name_viz_sl: SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G}

        visualize_fpn_detections_vs_gt(  # Используем основную функцию, но с данными для одного уровня
            image_processed_np_viz,
            (y_true_single_level_np_viz,),  # y_true как кортеж из одного элемента
            np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32),
            # Пустые предсказания
            [current_level_name_viz_sl],  # Список из одного имени уровня
            current_fpn_configs_viz_sl,  # Конфиг для одного уровня
            CLASSES_LIST_SDL_G,  # Классы из отладочного загрузчика
            title_prefix=f"Initial GT for {fixed_image_path.name} (Single Level: {current_level_name_viz_sl})",
            show_grid_for_level=current_level_name_viz_sl
        )
        plt.close('all')

    model_sl_fixed = build_detector_single_level_p4_debug()  # Используем отладочную модель
    optimizer_sl_fixed = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DEBUG_SL_FIXED)
    initial_trainable_weights_sl_fixed_list = [var.numpy().copy() for var in model_sl_fixed.trainable_variables]

    print(
        f"\nЗапуск {NUM_ITERATIONS_DEBUG_SL_FIXED} итераций обучения на ОДНОМ ФИКСИРОВАННОМ изображении: {fixed_image_path.name}...")

    for iteration_fixed_sl in range(NUM_ITERATIONS_DEBUG_SL_FIXED):
        # Используем один и тот же батч на каждой итерации
        images_batch_for_train = image_batch_tf_train
        y_true_batch_for_train = y_true_batch_tf_train

        with tf.GradientTape() as tape_sl_fixed:
            y_pred_single_level_logits = model_sl_fixed(images_batch_for_train, training=True)
            loss_details_dict_sl = compute_detector_loss_single_level_debug(
                y_true_batch_for_train,
                y_pred_single_level_logits,
                True
            )
            total_loss_sl = loss_details_dict_sl['total_loss']

        gradients_sl = tape_sl_fixed.gradient(total_loss_sl, model_sl_fixed.trainable_variables)
        optimizer_sl_fixed.apply_gradients(zip(gradients_sl, model_sl_fixed.trainable_variables))

        log_this_iter_sl = (iteration_fixed_sl + 1) % int(LOG_LOSS_EVERY_N_ITERATIONS_SL) == 0 or \
                           iteration_fixed_sl == 0 or iteration_fixed_sl == NUM_ITERATIONS_DEBUG_SL_FIXED - 1
        visualize_this_iter_sl = (iteration_fixed_sl + 1) % int(VISUALIZE_EVERY_N_ITERATIONS_SL_FIXED) == 0 or \
                                 iteration_fixed_sl == 0 or iteration_fixed_sl == NUM_ITERATIONS_DEBUG_SL_FIXED - 1

        if log_this_iter_sl:
            print(f"\n--- Итерация SL (Фикс. Изобр.): {iteration_fixed_sl + 1}/{NUM_ITERATIONS_DEBUG_SL_FIXED} ---")
            print("  Детальные Потери (Один Уровень):")
            for k_sl, v_tensor_sl in loss_details_dict_sl.items(): print(f"    {k_sl}: {v_tensor_sl.numpy():.6f}")
            if np.isnan(total_loss_sl.numpy()) or np.isinf(total_loss_sl.numpy()): print(
                "ОШИБКА: Потеря NaN/Inf!"); break

        if visualize_this_iter_sl and _plot_utils_imported_sdl_fixed:  # Убрал _predict_utils, т.к. NMS внутри
            if not log_this_iter_sl:
                print(
                    f"\n--- Итерация SL (Фикс. Изобр.): {iteration_fixed_sl + 1}/{NUM_ITERATIONS_DEBUG_SL_FIXED} (Только Визуализация) ---")
                print(f"    total_loss: {total_loss_sl.numpy():.6f}")

            print(f"  --- Визуализация предсказаний SL на итерации {iteration_fixed_sl + 1} ---")
            # Получаем предсказания модели (y_pred_single_level_logits уже есть)
            # Для визуализации нам нужно их декодировать и применить NMS
            # Блок декодирования и NMS для одного уровня:
            raw_preds_single_level_viz_sl = y_pred_single_level_logits  # Уже (1, Gh_p4, Gw_p4, A_p4, 5+C)
            gh_viz = GRID_H_P4_DEBUG_SDL_G;
            gw_viz = GRID_W_P4_DEBUG_SDL_G
            anchors_viz = ANCHORS_WH_P4_DEBUG_SDL_G;
            num_anchors_viz = NUM_ANCHORS_P4_DEBUG_SDL_G

            pred_xy_raw_viz = raw_preds_single_level_viz_sl[..., 0:2];
            pred_wh_raw_viz = raw_preds_single_level_viz_sl[..., 2:4]
            pred_obj_logit_viz = raw_preds_single_level_viz_sl[..., 4:5];
            pred_class_logits_viz = raw_preds_single_level_viz_sl[..., 5:]
            gy_indices_viz = tf.tile(tf.range(gh_viz, dtype=tf.float32)[:, tf.newaxis], [1, gw_viz])
            gx_indices_viz = tf.tile(tf.range(gw_viz, dtype=tf.float32)[tf.newaxis, :], [gh_viz, 1])
            grid_coords_xy_viz = tf.stack([gx_indices_viz, gy_indices_viz], axis=-1)
            grid_coords_xy_viz = grid_coords_xy_viz[tf.newaxis, :, :, tf.newaxis, :];
            grid_coords_xy_viz = tf.tile(grid_coords_xy_viz, [1, 1, 1, num_anchors_viz, 1])
            pred_xy_on_grid_viz = (tf.sigmoid(pred_xy_raw_viz) + grid_coords_xy_viz)
            pred_xy_normalized_viz = pred_xy_on_grid_viz / tf.constant([gw_viz, gh_viz], dtype=tf.float32)
            anchors_tensor_viz = tf.constant(anchors_viz, dtype=tf.float32);
            anchors_reshaped_viz = anchors_tensor_viz[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
            pred_wh_normalized_viz = (tf.exp(pred_wh_raw_viz) * anchors_reshaped_viz)
            decoded_boxes_xywh_viz = tf.concat([pred_xy_normalized_viz, pred_wh_normalized_viz], axis=-1)
            pred_obj_confidence_viz = tf.sigmoid(pred_obj_logit_viz);
            pred_class_probs_viz = tf.sigmoid(pred_class_logits_viz)
            num_total_boxes_viz = gh_viz * gw_viz * num_anchors_viz
            boxes_flat_viz = tf.reshape(decoded_boxes_xywh_viz, [1, num_total_boxes_viz, 4])
            obj_conf_flat_viz = tf.reshape(pred_obj_confidence_viz, [1, num_total_boxes_viz, 1])
            class_probs_flat_viz = tf.reshape(pred_class_probs_viz, [1, num_total_boxes_viz, NUM_CLASSES_SDL_G])
            final_scores_per_class_viz = obj_conf_flat_viz * class_probs_flat_viz
            boxes_ymin_xmin_ymax_xmax_viz = tf.concat([
                boxes_flat_viz[..., 1:2] - boxes_flat_viz[..., 3:4] / 2.0,
                boxes_flat_viz[..., 0:1] - boxes_flat_viz[..., 2:3] / 2.0,
                boxes_flat_viz[..., 1:2] + boxes_flat_viz[..., 3:4] / 2.0,
                boxes_flat_viz[..., 0:1] + boxes_flat_viz[..., 2:3] / 2.0
            ], axis=-1)
            boxes_ymin_xmin_ymax_xmax_viz = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax_viz, 0.0, 1.0)
            nms_boxes_viz, nms_scores_viz, nms_classes_viz, nms_valid_dets_viz = tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(boxes_ymin_xmin_ymax_xmax_viz, axis=2), scores=final_scores_per_class_viz,
                max_output_size_per_class=MAX_DETS_VIZ_SL_FIXED // NUM_CLASSES_SDL_G if NUM_CLASSES_SDL_G > 0 else MAX_DETS_VIZ_SL_FIXED,
                max_total_size=MAX_DETS_VIZ_SL_FIXED, iou_threshold=IOU_THRESH_NMS_VIZ_SL_FIXED,
                score_threshold=CONF_THRESH_VIZ_SL_FIXED
            )
            num_valid_dets_np_viz_iter_sl = nms_valid_dets_viz[0].numpy()
            print(f"    После NMS найдено {num_valid_dets_np_viz_iter_sl} объектов.")

            visualize_fpn_detections_vs_gt(  # Передаем y_true для одного уровня как кортеж
                image_processed_np_viz, (y_true_single_level_np_viz,),
                nms_boxes_viz[0][:num_valid_dets_np_viz_iter_sl].numpy(),
                nms_scores_viz[0][:num_valid_dets_np_viz_iter_sl].numpy(),
                nms_classes_viz[0][:num_valid_dets_np_viz_iter_sl].numpy().astype(int),
                [FPN_LEVEL_NAME_DEBUG_SDL_G],  # Имя нашего одного уровня
                {FPN_LEVEL_NAME_DEBUG_SDL_G: SINGLE_LEVEL_CONFIG_FOR_ASSIGN_SDL_G},  # Конфиг для одного уровня
                CLASSES_LIST_SDL_G,
                title_prefix=f"Iter {iteration_fixed_sl + 1} SL Fixed: {fixed_image_path.name}",
                show_grid_for_level=FPN_LEVEL_NAME_DEBUG_SDL_G
            )
            plt.close('all')

    if iteration_fixed_sl == NUM_ITERATIONS_DEBUG_SL_FIXED - 1:
        print("\n--- Проверка изменения весов SL (Фикс. Изобр.) в конце ---")
        check_weights_changed_sl_fixed(model_sl_fixed, initial_trainable_weights_sl_fixed_list,
                                       NUM_ITERATIONS_DEBUG_SL_FIXED)

    print("\n--- Отладочное обучение SL на одном ФИКСИРОВАННОМ примере завершено. ---")


if __name__ == '__main__':
    main_debug_single_level_train_on_fixed_image_func()