# debug_training_loop.py
import sys
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import yaml
import time
import glob
import random
import argparse

# --- Устанавливаем флаг окружения для функции потерь ---
os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] = "1"
# -------------------------------------------------------------

# --- Добавляем src в sys.path ---
_project_root_debug_train = Path(__file__).resolve().parent
_src_path_debug_train = _project_root_debug_train / 'src'
if str(_src_path_debug_train) not in sys.path:
    sys.path.insert(0, str(_src_path_debug_train))
# ------------------------------------

# --- Импорты из твоих модулей ---
try:
    from datasets.detector_data_loader import create_detector_tf_dataset
    from models.object_detector import build_object_detector_v2_fpn
    from losses.detection_losses import compute_detector_loss_v2_fpn
    # Импортируем также глобальные переменные из data_loader для отладочного вывода
    from datasets.detector_data_loader import (
        TARGET_IMG_HEIGHT as DDL_TARGET_IMG_HEIGHT,
        TARGET_IMG_WIDTH as DDL_TARGET_IMG_WIDTH,
        CLASSES_LIST_GLOBAL_FOR_DETECTOR as DDL_CLASSES_LIST,
        FPN_LEVELS_CONFIG_GLOBAL as DDL_FPN_LEVELS_CONFIG,
        NUM_CLASSES_DETECTOR as DDL_NUM_CLASSES
    )

    print("INFO (debug_training_loop): Компоненты успешно импортированы.")
except ImportError as e_imp_dbg_train:
    print(f"ОШИБКА КРИТИЧЕСКАЯ (debug_training_loop): Не удалось импортировать компоненты: {e_imp_dbg_train}")
    exit()
except Exception as e_gen_dbg_train:
    print(f"ОШИБКА КРИТИЧЕСКАЯ (debug_training_loop) при импорте: {e_gen_dbg_train}")
    exit()

# --- Загрузка Конфигураций ---
_base_config_path_dbg_train = _src_path_debug_train / 'configs' / 'base_config.yaml'
_detector_config_path_dbg_train = _src_path_debug_train / 'configs' / 'detector_config.yaml'
BASE_CONFIG_DBG_TRAIN = {}
DETECTOR_CONFIG_DBG_TRAIN = {}
try:
    with open(_base_config_path_dbg_train, 'r', encoding='utf-8') as f:
        BASE_CONFIG_DBG_TRAIN = yaml.safe_load(f)
    with open(_detector_config_path_dbg_train, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_DBG_TRAIN = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG_DBG_TRAIN, dict) or not isinstance(DETECTOR_CONFIG_DBG_TRAIN, dict): raise ValueError(
        "Конфиги не словари")
except Exception as e_cfg_dbg_train:
    print(f"ОШИБКА (debug_training_loop): Загрузка конфигов не удалась: {e_cfg_dbg_train}. Выход.");
    exit()

# --- Параметры для отладочного запуска (получаем из argparse) ---
# Пути к данным
_detector_dataset_ready_path_rel_dbg_train = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_DBG_TRAIN = (
            _project_root_debug_train / _detector_dataset_ready_path_rel_dbg_train).resolve()
_images_subdir_name_from_local_cfg = BASE_CONFIG_DBG_TRAIN.get('dataset', {}).get('images_dir', 'JPEGImages')
_annotations_subdir_name_from_local_cfg = BASE_CONFIG_DBG_TRAIN.get('dataset', {}).get('annotations_dir', 'Annotations')
IMAGES_DIR_FOR_DEBUG_TRAIN = str(DETECTOR_DATASET_READY_ABS_DBG_TRAIN / "train" / _images_subdir_name_from_local_cfg)
ANNOTATIONS_DIR_FOR_DEBUG_TRAIN = str(
    DETECTOR_DATASET_READY_ABS_DBG_TRAIN / "train" / _annotations_subdir_name_from_local_cfg)


def get_debug_data(num_samples_needed, image_dir, annot_dir, seed=None):
    all_image_paths_raw = []
    for ext_dbg_data in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        all_image_paths_raw.extend(glob.glob(os.path.join(image_dir, ext_dbg_data)))
    seen_dbg_data = set();
    unique_image_paths_dbg_data = []
    for fp_dbg_data in all_image_paths_raw:
        res_fp_dbg_data = os.path.normpath(fp_dbg_data)
        if res_fp_dbg_data not in seen_dbg_data: unique_image_paths_dbg_data.append(res_fp_dbg_data); seen_dbg_data.add(
            res_fp_dbg_data)
    paired_files_for_debug = []
    for img_p_dbg_data in unique_image_paths_dbg_data:
        bn_dbg_data, _ = os.path.splitext(os.path.basename(img_p_dbg_data))
        xml_p_dbg_data = os.path.join(annot_dir, bn_dbg_data + ".xml")
        if os.path.exists(xml_p_dbg_data):
            paired_files_for_debug.append((img_p_dbg_data, xml_p_dbg_data, os.path.basename(img_p_dbg_data)))
    if not paired_files_for_debug: print(f"ОШИБКА (get_debug_data): Нет пар в {image_dir}."); return None, None, None
    print(
        f"INFO (get_debug_data): Найдено {len(paired_files_for_debug)} валидных пар в '{os.path.basename(image_dir)}'.")
    num_to_select = min(num_samples_needed, len(paired_files_for_debug))
    if num_to_select == 0: print("ОШИБКА (get_debug_data): Недостаточно данных."); return None, None, None
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(int(time.time()))
    selected_pairs_dbg_data = random.sample(paired_files_for_debug, num_to_select)
    debug_image_paths_list = [p[0] for p in selected_pairs_dbg_data]
    debug_xml_paths_list = [p[1] for p in selected_pairs_dbg_data]
    debug_image_basenames_list = [p[2] for p in selected_pairs_dbg_data]
    print(f"INFO (get_debug_data): Для отладки обучения будет использовано {len(debug_image_paths_list)} изображений.")
    return debug_image_paths_list, debug_xml_paths_list, debug_image_basenames_list


def debug_training_main(args):
    print("\n--- Отладка Цикла Обучения Детектора (FPN Модель) ---")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Количество шагов: {args.num_steps}")
    print(f"  Аугментация: {args.use_augmentation}")
    print(f"  Backbone заморожен (при создании модели): {args.freeze_backbone_build_flag}")
    if args.clip_grad_norm is not None: print(f"  Ограничение градиентов по норме: {args.clip_grad_norm}")

    num_unique_samples_for_debug = args.batch_size * args.num_steps
    debug_img_paths, debug_xml_paths, debug_img_basenames = get_debug_data(
        num_samples_needed=max(args.batch_size, num_unique_samples_for_debug),
        image_dir=IMAGES_DIR_FOR_DEBUG_TRAIN,
        annot_dir=ANNOTATIONS_DIR_FOR_DEBUG_TRAIN,
        seed=args.data_seed
    )
    if not debug_img_paths: return

    debug_dataset = create_detector_tf_dataset(
        debug_img_paths, debug_xml_paths,
        batch_size=args.batch_size,
        shuffle=False,
        augment=args.use_augmentation
    )
    debug_dataset = debug_dataset.repeat()
    if debug_dataset is None: print("Не удалось создать отладочный датасет."); return

    print("\nСоздание модели детектора (build_object_detector_v2_fpn)...")
    model = build_object_detector_v2_fpn(force_freeze_backbone_arg=args.freeze_backbone_build_flag)
    print("\nСтруктура модели для отладки:");
    model.summary(line_length=120)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    print("\n--- Начало Отладочного Цикла Обучения ---")
    dataset_iterator = iter(debug_dataset)

    weights_before_all_steps = {}
    if model.trainable_variables:
        for var in model.trainable_variables: weights_before_all_steps[var.name] = var.numpy().copy()
        print(f"  Сохранены начальные веса для {len(weights_before_all_steps)} обучаемых переменных.")
    else:
        print("  ПРЕДУПРЕЖДЕНИЕ: В модели нет обучаемых переменных.")

    # Ключевые слои для отслеживания градиентов (последние Conv2D в каждой голове)
    key_gradient_layers = [f"head_{level}_raw_preds/kernel:0" for level in DDL_FPN_LEVELS_CONFIG.keys()]
    key_gradient_layers += [f"head_{level}_raw_preds/bias:0" for level in DDL_FPN_LEVELS_CONFIG.keys()]

    for step in range(args.num_steps):
        start_idx = (step * args.batch_size) % len(debug_img_basenames)
        end_idx = start_idx + args.batch_size
        current_batch_img_names_list = [debug_img_basenames[i % len(debug_img_basenames)] for i in
                                        range(start_idx, end_idx)]
        print(
            f"\n--- Шаг Отладки {step + 1}/{args.num_steps} (Изображения: {', '.join(current_batch_img_names_list[:2])}{'...' if len(current_batch_img_names_list) > 2 else ''}) ---")

        try:
            images_batch, y_true_batch_tuple = next(dataset_iterator)
        except Exception as e_data_dbg_loop:
            print(f"  ОШИБКА при получении батча: {e_data_dbg_loop}"); break

        with tf.GradientTape() as tape:
            y_pred_batch_list = model(images_batch, training=True)
            loss_output = compute_detector_loss_v2_fpn(y_true_batch_tuple, y_pred_batch_list)
            actual_loss_for_grads = loss_output['total_loss'] if isinstance(loss_output, dict) else loss_output
            print(f"  Потери на шаге {step + 1}:")
            if isinstance(loss_output, dict):
                for k_loss, v_loss in loss_output.items(): print(f"    {k_loss}: {v_loss.numpy():.4f}")
            else:
                print(f"    total_loss: {actual_loss_for_grads.numpy():.4f}")

        if tf.math.is_nan(actual_loss_for_grads) or tf.math.is_inf(actual_loss_for_grads):
            print("  ОШИБКА КРИТИЧЕСКАЯ: Потеря стала NaN или Inf! Остановка.");
            break

        trainable_vars = model.trainable_variables
        if not trainable_vars: print("  ОШИБКА: Нет обучаемых переменных!"); break
        gradients = tape.gradient(actual_loss_for_grads, trainable_vars)

        if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
            gradients = [(tf.clip_by_norm(g, args.clip_grad_norm) if g is not None else None) for g in gradients]

        nan_inf_g, non_zero_g, null_g, valid_g = 0, 0, 0, 0
        print("  Анализ градиентов для ключевых слоев:")
        for i_grad_dbg, grad_dbg in enumerate(gradients):
            var_name_grad = trainable_vars[i_grad_dbg].name
            if grad_dbg is not None:
                valid_g += 1
                if tf.reduce_any(tf.math.is_nan(grad_dbg)) or tf.reduce_any(tf.math.is_inf(grad_dbg)): nan_inf_g += 1
                if tf.reduce_any(grad_dbg != 0): non_zero_g += 1
                if var_name_grad in key_gradient_layers:  # Выводим инфо для ключевых слоев
                    grad_norm_val = tf.norm(grad_dbg).numpy()
                    mean_abs_grad_val = tf.reduce_mean(tf.abs(grad_dbg)).numpy()
                    print(f"    '{var_name_grad}': Норма={grad_norm_val:.2e}, Средн.Abs={mean_abs_grad_val:.2e}")
            else:
                null_g += 1
        print(
            f"  Итог по градиентам: Всего={len(gradients)}, Валидных={valid_g}, Null={null_g}, NaN/Inf={nan_inf_g}, Ненулевых={non_zero_g}")
        if nan_inf_g > 0: print("  ОШИБКА КРИТИЧЕСКАЯ: Градиенты NaN/Inf!"); break
        if valid_g > 0 and non_zero_g == 0 and step > 0: print("  ПРЕДУПРЕЖДЕНИЕ: Все валидные градиенты нулевые!")

        optimizer.apply_gradients(zip(gradients, trainable_vars))

    if model.trainable_variables and weights_before_all_steps:
        print("\n--- Проверка изменения весов (после всех шагов) ---")
        changed_vars_count = 0;
        unchanged_vars_count = 0;
        problem_vars_count = 0
        print(f"{'Имя Переменной':<70} {'Форма':<20} {'Сумма Abs Разн.':<15} {'Макс Abs Разн.':<15} Статус")
        print("-" * 130)
        current_weights_dict = {var.name: var.numpy() for var in model.trainable_variables}
        for var_name_check, initial_weight_val_check in weights_before_all_steps.items():
            status = "НЕ НАЙДЕНА"
            sum_abs_diff_str = "N/A";
            max_abs_diff_str = "N/A"
            shape_str = str(initial_weight_val_check.shape)

            if var_name_check in current_weights_dict:
                current_weight_val_check = current_weights_dict[var_name_check]
                shape_str = str(current_weight_val_check.shape)  # Берем актуальную форму
                if initial_weight_val_check.shape == current_weight_val_check.shape:
                    abs_diff_tensor = np.abs(initial_weight_val_check - current_weight_val_check)
                    sum_abs_diff = np.sum(abs_diff_tensor)
                    max_abs_diff = np.max(abs_diff_tensor)
                    sum_abs_diff_str = f"{sum_abs_diff:.2e}"
                    max_abs_diff_str = f"{max_abs_diff:.2e}"
                    if not np.allclose(initial_weight_val_check, current_weight_val_check, atol=1e-7):
                        status = "ИЗМЕНЕНЫ"
                        changed_vars_count += 1
                    else:
                        status = "НЕ изменены"
                        unchanged_vars_count += 1
                else:
                    status = "ОШИБКА ФОРМЫ!"
                    sum_abs_diff_str = f"ДО {initial_weight_val_check.shape}"
                    max_abs_diff_str = f"ПОСЛЕ {current_weight_val_check.shape}"
                    problem_vars_count += 1
            else:
                problem_vars_count += 1

            # Выводим только для слоев "голов" и FPN для краткости, если их много
            if "detector_body" in var_name_check or "head_P" in var_name_check or "fpn_" in var_name_check or "detection_head" in var_name_check:
                print(f"{var_name_check:<70} {shape_str:<20} {sum_abs_diff_str:<15} {max_abs_diff_str:<15} {status}")

        print(
            f"\n  Итог: Измененных: {changed_vars_count}, НЕизмененных: {unchanged_vars_count}, Проблемных: {problem_vars_count}")
        if changed_vars_count == 0 and model.trainable_variables and problem_vars_count == 0 and unchanged_vars_count > 0:
            print("  ПРЕДУПРЕЖДЕНИЕ: Ни одна из отслеживаемых обучаемых переменных не изменила свои веса!")

    print("\n--- Отладочный Цикл Обучения Завершен ---")
    if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ: del os.environ["DEBUG_TRAINING_LOOP_ACTIVE"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отладочный цикл обучения для FPN детектора.")
    parser.add_argument("--batch_size", type=int,
                        default=DETECTOR_CONFIG_DBG_TRAIN.get('train_params', {}).get('batch_size', 2),
                        help="Размер батча.")
    parser.add_argument("--learning_rate", type=float,
                        default=DETECTOR_CONFIG_DBG_TRAIN.get('initial_learning_rate', 1e-4),
                        help="Скорость обучения.")
    parser.add_argument("--num_steps", type=int, default=50,  # Увеличил дефолтное количество шагов
                        help="Количество шагов (батчей) для отладки.")
    parser.add_argument("--use_augmentation", type=lambda x: (str(x).lower() == 'true'),
                        default=DETECTOR_CONFIG_DBG_TRAIN.get('use_augmentation', True),
                        help="Использовать ли аугментацию (true/false).")
    parser.add_argument("--freeze_backbone_build_flag", type=lambda x: (str(x).lower() == 'true'),
                        default=(not DETECTOR_CONFIG_DBG_TRAIN.get('unfreeze_backbone', False)),
                        help="Заморозить ли backbone ПРИ СОЗДАНИИ модели (true/false).")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed для выбора данных get_debug_data (None для случайного).")
    parser.add_argument("--clip_grad_norm", type=float, default=None,
                        help="Значение для ограничения нормы градиентов (например, 1.0). Отключено, если None или 0.")

    cli_args = parser.parse_args()
    debug_training_main(cli_args)