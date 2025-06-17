# src/losses/detection_losses.py
import tensorflow as tf
import yaml
import os
import numpy as np
import sys  # Для вывода в stderr в tf.print
from pathlib import Path  # Используем pathlib для путей

# --- Загрузка Конфигурации (на уровне модуля) ---
# Эти переменные будут доступны для функций, определенных в этом модуле.
_current_script_dir_loss_mod = Path(__file__).resolve().parent
_project_root_loss_mod = _current_script_dir_loss_mod.parent.parent
_detector_config_path_loss_mod = _project_root_loss_mod / 'src' / 'configs' / 'detector_config.yaml'

DETECTOR_CONFIG_MODULE_LEVEL = {}  # Используем имя, специфичное для этого модуля
CONFIG_LOAD_SUCCESS_MODULE_LEVEL = True
try:
    with open(_detector_config_path_loss_mod, 'r', encoding='utf-8') as f_mod_cfg:
        DETECTOR_CONFIG_MODULE_LEVEL = yaml.safe_load(f_mod_cfg)
    if not isinstance(DETECTOR_CONFIG_MODULE_LEVEL, dict):
        DETECTOR_CONFIG_MODULE_LEVEL = {}  # Сброс, если не словарь
        CONFIG_LOAD_SUCCESS_MODULE_LEVEL = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_MODULE_LEVEL = False
    # Предупреждение будет выведено, если этот модуль запускается как главный
    if __name__ == '__main__': print(
        f"ПРЕДУПРЕЖДЕНИЕ (detection_losses.py глобально): Файл detector_config.yaml не найден: {_detector_config_path_loss_mod}.")
except yaml.YAMLError as e_yaml_mod:
    CONFIG_LOAD_SUCCESS_MODULE_LEVEL = False
    if __name__ == '__main__': print(
        f"ОШИБКА (detection_losses.py глобально): YAML в detector_config.yaml: {e_yaml_mod}")

# --- Глобальные переменные модуля, инициализируемые из конфига ---
_fpn_params_loss_module_level = DETECTOR_CONFIG_MODULE_LEVEL.get('fpn_detector_params', {})
NUM_CLASSES_LOSS_FPN_MODULE_LEVEL = _fpn_params_loss_module_level.get('num_classes', 2)
FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL = _fpn_params_loss_module_level.get('detector_fpn_levels', ['P3', 'P4', 'P5'])

_loss_weights_cfg_module_level = DETECTOR_CONFIG_MODULE_LEVEL.get('loss_weights', {})
COORD_LOSS_WEIGHT_FPN_MODULE_LEVEL = _loss_weights_cfg_module_level.get('coordinates', 1.0)
OBJECTNESS_LOSS_WEIGHT_FPN_MODULE_LEVEL = _loss_weights_cfg_module_level.get('objectness', 1.0)
NO_OBJECT_LOSS_WEIGHT_FPN_MODULE_LEVEL = _loss_weights_cfg_module_level.get('no_object', 0.5)
CLASS_LOSS_WEIGHT_FPN_MODULE_LEVEL = _loss_weights_cfg_module_level.get('classification', 1.0)


# ------------------------------------------------------------------

@tf.function
def calculate_single_level_loss(y_true_level, y_pred_level,
                                num_classes_arg,  # Явно передаем
                                coord_weight_arg, obj_weight_arg,
                                no_obj_weight_arg, cls_weight_arg,  # Явно передаем веса
                                level_name_debug="UnknownLevel"):
    """
    Рассчитывает компоненты потерь для одного уровня FPN.
    Возвращает: objectness_loss, class_loss, box_loss, num_responsible_objects_for_level
    """
    shape_y_true = tf.shape(y_true_level)
    batch_size_dynamic = shape_y_true[0]
    expected_features_per_anchor = 4 + 1 + num_classes_arg
    M_total_predictions_level = tf.cast(tf.reduce_prod(shape_y_true[1:-1]), tf.float32)
    if M_total_predictions_level == 0: M_total_predictions_level = tf.constant(1.0, dtype=tf.float32)

    y_true_reshaped = tf.reshape(y_true_level, [batch_size_dynamic, -1, expected_features_per_anchor])
    y_pred_reshaped = tf.reshape(y_pred_level, [batch_size_dynamic, -1, expected_features_per_anchor])

    true_boxes_encoded = y_true_reshaped[..., 0:4]
    pred_boxes_encoded_raw = y_pred_reshaped[..., 0:4]
    true_objectness = y_true_reshaped[..., 4:5]
    pred_objectness_logits = y_pred_reshaped[..., 4:5]
    true_classes_one_hot = y_true_reshaped[..., 5:5 + num_classes_arg]
    pred_classes_logits = y_pred_reshaped[..., 5:5 + num_classes_arg]

    objectness_bce_loss_per_item = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_objectness, logits=pred_objectness_logits)
    loss_weights_obj_tensor = tf.where(tf.equal(true_objectness, 1.0), obj_weight_arg,
                                       no_obj_weight_arg)  # Используем переданные веса
    weighted_objectness_loss_per_item = objectness_bce_loss_per_item * loss_weights_obj_tensor
    total_objectness_loss_level = tf.reduce_sum(weighted_objectness_loss_per_item) / (
                tf.cast(batch_size_dynamic, tf.float32) * M_total_predictions_level + 1e-6)

    objectness_mask_for_responsible = tf.squeeze(tf.cast(true_objectness, tf.float32), axis=-1)
    num_responsible_objects_float = tf.maximum(tf.reduce_sum(objectness_mask_for_responsible), 1e-6)

    class_bce_loss_per_item_per_class = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_classes_one_hot, logits=pred_classes_logits)
    class_loss_per_prediction = tf.reduce_sum(class_bce_loss_per_item_per_class, axis=-1)
    masked_class_loss_per_item = class_loss_per_prediction * objectness_mask_for_responsible
    total_class_loss_level = (tf.reduce_sum(
        masked_class_loss_per_item) / num_responsible_objects_float) * cls_weight_arg  # Используем переданный вес

    delta_huber = 1.0;
    abs_error_coords = tf.abs(true_boxes_encoded - pred_boxes_encoded_raw)
    is_small_error_coords = abs_error_coords <= delta_huber
    squared_loss_coords = 0.5 * tf.square(abs_error_coords);
    linear_loss_coords = delta_huber * (abs_error_coords - 0.5 * delta_huber)
    coord_loss_individual_components = tf.where(is_small_error_coords, squared_loss_coords, linear_loss_coords)
    coord_loss_per_prediction = tf.reduce_sum(coord_loss_individual_components, axis=-1)
    masked_coord_loss_per_item = coord_loss_per_prediction * objectness_mask_for_responsible
    total_box_loss_level = (tf.reduce_sum(
        masked_coord_loss_per_item) / num_responsible_objects_float) * coord_weight_arg  # Используем переданный вес

    return total_objectness_loss_level, total_class_loss_level, total_box_loss_level, num_responsible_objects_float


@tf.function
def compute_detector_loss_v2_fpn(y_true_fpn_tuple, y_pred_fpn_list):
    """
    Рассчитывает общую потерю для FPN детектора.
    Использует глобальные переменные модуля для конфигурации и весов потерь.
    """
    # Используем глобальные переменные, определенные в начале этого модуля
    fpn_level_names_to_iterate = FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL
    num_classes_for_loss = NUM_CLASSES_LOSS_FPN_MODULE_LEVEL
    coord_w = COORD_LOSS_WEIGHT_FPN_MODULE_LEVEL
    obj_w = OBJECTNESS_LOSS_WEIGHT_FPN_MODULE_LEVEL
    no_obj_w = NO_OBJECT_LOSS_WEIGHT_FPN_MODULE_LEVEL
    cls_w = CLASS_LOSS_WEIGHT_FPN_MODULE_LEVEL

    num_fpn_levels = len(fpn_level_names_to_iterate)
    if len(y_true_fpn_tuple) != num_fpn_levels or len(y_pred_fpn_list) != num_fpn_levels:
        tf.print("ОШИБКА compute_loss_v2_fpn: Несоответствие уровней в y_true/y_pred.", output_stream=sys.stderr)
        return tf.constant(1e9, dtype=tf.float32)

    total_loss_agg = tf.constant(0.0, dtype=tf.float32)
    debug_losses_components = {}  # Для отладочного режима

    for i in range(num_fpn_levels):
        level_name = fpn_level_names_to_iterate[i]
        obj_l, cls_l, box_l, num_pos_l = calculate_single_level_loss(
            y_true_fpn_tuple[i], y_pred_fpn_list[i],
            num_classes_for_loss,
            coord_w, obj_w, no_obj_w, cls_w,
            level_name_debug=level_name
        )
        total_loss_agg += (obj_l + cls_l + box_l)  # Предполагаем, что веса уровней FPN равны 1.0

        if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ and os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] == "1":
            debug_losses_components[f"obj_loss_{level_name}"] = obj_l
            debug_losses_components[f"cls_loss_{level_name}"] = cls_l
            debug_losses_components[f"box_loss_{level_name}"] = box_l
            debug_losses_components[f"num_pos_anchors_{level_name}"] = num_pos_l

    if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ and os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] == "1":
        debug_losses_components["total_loss"] = total_loss_agg
        return debug_losses_components
    else:
        return total_loss_agg


# --- Блок if __name__ == '__main__': для тестирования этого файла ---
if __name__ == '__main__':
    print(f"--- Тестирование detection_losses.py (для FPN) ---")

    # Используем глобальные переменные модуля, которые уже были инициализированы из конфига или дефолтами
    if not CONFIG_LOAD_SUCCESS_MODULE_LEVEL:  # Этот флаг устанавливается при загрузке конфига в начале модуля
        print(
            "ОШИБКА: Конфигурационный файл detector_config.yaml не был загружен корректно при инициализации модуля detection_losses.py.")
        print("         Тестирование будет использовать жестко заданные дефолты, что может быть нерепрезентативно.")

    print(f"Используемые веса потерь (из конфигурации модуля):")
    print(f"  Координаты: {COORD_LOSS_WEIGHT_FPN_MODULE_LEVEL}");
    print(f"  Objectness (позитивные): {OBJECTNESS_LOSS_WEIGHT_FPN_MODULE_LEVEL}")
    print(f"  No Object (негативные): {NO_OBJECT_LOSS_WEIGHT_FPN_MODULE_LEVEL}");
    print(f"  Классификация: {CLASS_LOSS_WEIGHT_FPN_MODULE_LEVEL}")
    print(f"Количество классов для потерь (из конфигурации модуля): {NUM_CLASSES_LOSS_FPN_MODULE_LEVEL}")
    print(f"Уровни FPN для потерь (из конфигурации модуля): {FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL}")

    # --- Получаем конфигурацию FPN уровней для теста ---
    # Эта конфигурация должна быть ПОЛНОСТЬЮ определена (включая якоря, страйды, размеры сетки)
    # Она будет использоваться для создания y_true и y_pred правильных форм.
    # Мы можем попытаться загрузить ее еще раз здесь для __main__ или использовать ту, что уже есть в модуле.
    # Для простоты и консистентности, будем использовать FPN_LEVELS_CONFIG_GLOBAL,
    # который должен быть доступен, если этот файл импортируется (например, из detector_data_loader).
    # НО, если мы запускаем losses.py НАПРЯМУЮ, нам нужно самим его построить из DETECTOR_CONFIG_MODULE_LEVEL.

    # --- Начало блока построения тестовой конфигурации FPN уровней для __main__ ---
    _main_loss_fpn_params_test_local = DETECTOR_CONFIG_MODULE_LEVEL.get('fpn_detector_params', {})
    _main_loss_input_shape_list_test_local = _main_loss_fpn_params_test_local.get('input_shape', [416, 416, 3])

    main_loss_target_h_test_local = _main_loss_input_shape_list_test_local[0]
    main_loss_target_w_test_local = _main_loss_input_shape_list_test_local[1]

    # Используем NUM_CLASSES_LOSS_FPN_MODULE_LEVEL, определенный глобально в этом модуле
    main_loss_num_classes_for_test = NUM_CLASSES_LOSS_FPN_MODULE_LEVEL
    # Используем FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL, определенный глобально
    main_loss_fpn_level_names_for_test = FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL

    main_loss_fpn_strides_yaml_test_local = _main_loss_fpn_params_test_local.get('detector_fpn_strides', {})
    main_loss_fpn_anchors_yaml_test_local = _main_loss_fpn_params_test_local.get('detector_fpn_anchor_configs', {})
    main_loss_fpn_level_names_from_cfg = _main_loss_fpn_params_test_local.get('detector_fpn_levels', FPN_LEVEL_NAMES_ORDERED_LOSS_MODULE_LEVEL)

    FPN_CONFIGS_FOR_MAIN_TEST = {}  # Локальная переменная для этого __main__
    for ln_main_loss_local in main_loss_fpn_level_names_for_test:
        lc_main_loss_local = main_loss_fpn_anchors_yaml_test_local.get(ln_main_loss_local, {})
        # Используем _PLOT_UTILS_DEFAULT_FPN_CONFIGS_DICT (из plot_utils) как источник структуры дефолтов,
        # если он был бы здесь определен. Но лучше определить свои дефолты для losses.py.
        # Давайте определим минимальные дефолты прямо здесь, если в конфиге ничего нет.
        default_stride_main_loss = {'P3': 8, 'P4': 16, 'P5': 32}.get(ln_main_loss_local, 16)
        default_anchors_main_loss_list = [[0.1, 0.1]] * 3  # Дефолт 3 якоря
        default_num_anchors_main_loss = 3

        ls_main_loss_local = main_loss_fpn_strides_yaml_test_local.get(ln_main_loss_local, default_stride_main_loss)
        awh_main_list_from_cfg_loss_local = lc_main_loss_local.get('anchors_wh_normalized',
                                                                   default_anchors_main_loss_list)
        na_main_from_cfg_loss_local = lc_main_loss_local.get('num_anchors_this_level')

        current_anchors_np_main_loss_local = np.array(awh_main_list_from_cfg_loss_local, dtype=np.float32)
        if not (current_anchors_np_main_loss_local.ndim == 2 and \
                current_anchors_np_main_loss_local.shape[1] == 2 and \
                current_anchors_np_main_loss_local.shape[0] > 0):
            current_anchors_np_main_loss_local = np.array(default_anchors_main_loss_list, dtype=np.float32)

        na_main_final_loss_local = current_anchors_np_main_loss_local.shape[0]
        if na_main_from_cfg_loss_local is not None and na_main_from_cfg_loss_local != na_main_final_loss_local:
            pass  # Можно добавить print предупреждение

        FPN_CONFIGS_FOR_MAIN_TEST[ln_main_loss_local] = {
            'stride': ls_main_loss_local,
            'anchors_wh_normalized': current_anchors_np_main_loss_local,
            'num_anchors': na_main_final_loss_local,
            'grid_h': main_loss_target_h_test_local // ls_main_loss_local if ls_main_loss_local > 0 else 0,
            'grid_w': main_loss_target_w_test_local // ls_main_loss_local if ls_main_loss_local > 0 else 0
        }
    # --- Конец блока построения тестовой конфигурации FPN уровней ---

    batch_size_test_val_main, num_classes_test_val_main = 2, main_loss_num_classes_for_test  # Используем main_loss_num_classes_for_test

    # Инициализируем списки ДО цикла
    y_true_list_test_main_val = []
    y_pred_list_test_main_val = []

    for level_name_test_main_loop in main_loss_fpn_level_names_for_test:  # Используем main_loss_fpn_level_names_for_test
        level_cfg_test_main_loop = FPN_CONFIGS_FOR_MAIN_TEST.get(
            level_name_test_main_loop)  # Используем FPN_CONFIGS_FOR_MAIN_TEST
        if not level_cfg_test_main_loop:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ (__main__ losses): Нет конфига для уровня {level_name_test_main_loop} в FPN_CONFIGS_FOR_MAIN_TEST, пропускаем.")
            continue

        gh, gw, na = level_cfg_test_main_loop['grid_h'], level_cfg_test_main_loop['grid_w'], level_cfg_test_main_loop[
            'num_anchors']
        # Проверка на нулевые размеры сетки или якорей
        if gh == 0 or gw == 0 or na == 0:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ (__main__ losses): Нулевые размеры сетки или якорей для уровня {level_name_test_main_loop}. gh={gh}, gw={gw}, na={na}. Пропускаем этот уровень для теста.")
            # Добавляем заглушку, чтобы длина списков y_true/y_pred совпадала с количеством уровней
            dummy_shape = (batch_size_test_val_main, 1, 1, 1, 5 + num_classes_test_val_main)  # Минимальная форма
            y_true_list_test_main_val.append(tf.zeros(dummy_shape, dtype=tf.float32))
            y_pred_list_test_main_val.append(tf.zeros(dummy_shape, dtype=tf.float32))
            continue

        y_true_shape_level_main_loop = (batch_size_test_val_main, gh, gw, na, 5 + num_classes_test_val_main)
        y_true_level_np_main_loop = np.zeros(y_true_shape_level_main_loop, dtype=np.float32)

        if level_name_test_main_loop == "P4":  # Добавляем один "настоящий" объект на уровне P4
            anchor_w_val_test = level_cfg_test_main_loop['anchors_wh_normalized'][0, 0]
            anchor_h_val_test = level_cfg_test_main_loop['anchors_wh_normalized'][0, 1]
            # ... (остальная логика заполнения y_true_level_np_main_loop как была) ...
            y_true_level_np_main_loop[0, gh // 2, gw // 2, 0, 0:4] = [0.5, 0.5,
                                                                      np.log(0.2 / (anchor_w_val_test + 1e-9)),
                                                                      np.log(0.3 / (anchor_h_val_test + 1e-9))]
            y_true_level_np_main_loop[0, gh // 2, gw // 2, 0, 4] = 1.0
            y_true_level_np_main_loop[0, gh // 2, gw // 2, 0, 5 + 0] = 1.0
        y_true_list_test_main_val.append(tf.constant(y_true_level_np_main_loop, dtype=tf.float32))

        y_pred_level_logits_np_main_loop = np.random.randn(*y_true_shape_level_main_loop).astype(np.float32) * 0.1
        if level_name_test_main_loop == "P4":
            # ... (логика заполнения y_pred_level_logits_np_main_loop как была) ...
            y_pred_level_logits_np_main_loop[0, gh // 2, gw // 2, 0, 0:4] = y_true_level_np_main_loop[0, gh // 2,
                                                                            gw // 2, 0, 0:4] + np.random.randn(4) * 0.05
            y_pred_level_logits_np_main_loop[0, gh // 2, gw // 2, 0, 4] = 5.0
            pred_cls_obj_level_main_loop = np.full(num_classes_test_val_main, -5.0, dtype=np.float32);
            pred_cls_obj_level_main_loop[0] = 5.0
            y_pred_level_logits_np_main_loop[0, gh // 2, gw // 2, 0, 5:] = pred_cls_obj_level_main_loop
        y_pred_list_test_main_val.append(tf.constant(y_pred_level_logits_np_main_loop, dtype=np.float32))

    print("\nТестирование функции потерь FPN (compute_detector_loss_v2_fpn)...")
    os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] = "1"

    # Проверка, что списки не пустые перед вызовом потерь
    if not y_true_list_test_main_val or not y_pred_list_test_main_val or \
            len(y_true_list_test_main_val) != len(main_loss_fpn_level_names_from_cfg) or \
            len(y_pred_list_test_main_val) != len(main_loss_fpn_level_names_from_cfg):
        print("ОШИБКА: y_true или y_pred для теста потерь не были корректно сформированы для всех уровней FPN.")
    else:
        loss_output_main = compute_detector_loss_v2_fpn(tuple(y_true_list_test_main_val),
                                                        y_pred_list_test_main_val)  # <--- Используем y_true_list_test_main_val
        if isinstance(loss_output_main, dict):
            print(f"  Общая потеря FPN (случайные+1 объект): {loss_output_main['total_loss'].numpy():.4f}")
            for k_loss_main, v_loss_main in loss_output_main.items():
                if k_loss_main != 'total_loss': print(f"    {k_loss_main}: {v_loss_main.numpy():.4f}")
        else:
            print(f"  Общая потеря FPN (случайные+1 объект): {loss_output_main.numpy():.4f}")

        y_pred_perfect_list_test_main_val = []  # <--- Инициализируем здесь
        for y_true_level_tf_perfect_main_val in y_true_list_test_main_val:  # <--- Используем y_true_list_test_main_val
            # ... (остальная логика для y_pred_perfect_list_test_main_val как была) ...
            y_true_np_perf_main_val = y_true_level_tf_perfect_main_val.numpy();
            y_pred_lvl_perf_np_main_val = np.zeros_like(y_true_np_perf_main_val, dtype=np.float32)
            y_pred_lvl_perf_np_main_val[..., 0:4] = y_true_np_perf_main_val[..., 0:4]
            y_pred_lvl_perf_np_main_val[..., 4:5] = np.where(y_true_np_perf_main_val[..., 4:5] > 0.5, 10.0, -10.0)
            true_cls_one_hot_perf_lvl_main_val = y_true_np_perf_main_val[..., 5:]
            pred_cls_logits_perf_lvl_main_val = np.where(true_cls_one_hot_perf_lvl_main_val > 0.5, 10.0, -10.0)
            y_pred_lvl_perf_np_main_val[..., 5:] = pred_cls_logits_perf_lvl_main_val
            y_pred_perfect_list_test_main_val.append(tf.constant(y_pred_lvl_perf_np_main_val, dtype=np.float32))

        if y_true_list_test_main_val and y_pred_perfect_list_test_main_val:  # Проверка что не пустые
            loss_fpn_perfect_main = compute_detector_loss_v2_fpn(tuple(y_true_list_test_main_val),
                                                                 y_pred_perfect_list_test_main_val)  # <--- Используем y_true_list_test_main_val
            if isinstance(loss_fpn_perfect_main, dict):
                print(
                    f"  Общая потеря FPN (для 'идеальных' логитов): {loss_fpn_perfect_main['total_loss'].numpy():.4f}")
            else:
                print(f"  Общая потеря FPN (для 'идеальных' логитов): {loss_fpn_perfect_main.numpy():.4f}")

    if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ: del os.environ["DEBUG_TRAINING_LOOP_ACTIVE"]
    print("\n--- Тестирование detection_losses.py (FPN) завершено ---")