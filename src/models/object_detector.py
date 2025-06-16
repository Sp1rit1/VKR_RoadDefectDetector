# src/models/object_detector.py
import tensorflow as tf
import yaml
import os
import numpy as np
import time  # Добавим для тестового прогона

# --- Загрузка Конфигурации ---
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))
_detector_config_path = os.path.join(_project_root_dir, 'src', 'configs', 'detector_config.yaml')

DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS_MODEL = True
try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_MODEL = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_MODEL = False;
    print(f"ОШИБКА (object_detector.py): detector_config.yaml не найден: {_detector_config_path}")
except yaml.YAMLError as e:
    CONFIG_LOAD_SUCCESS_MODEL = False;
    print(f"ОШИБКА (object_detector.py): YAML в detector_config.yaml: {e}")

if not CONFIG_LOAD_SUCCESS_MODEL:
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка загрузки detector_config.yaml. Используются дефолты в object_detector.py.")
    DETECTOR_CONFIG.setdefault('backbone_name', 'MobileNetV2')
    DETECTOR_CONFIG.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG.setdefault('freeze_backbone', True)
    DETECTOR_CONFIG.setdefault('num_classes', 2)
    DETECTOR_CONFIG.setdefault('fpn_filters', 128)
    DETECTOR_CONFIG.setdefault('head_depth', 2)
    DETECTOR_CONFIG.setdefault('head_conv_filters', 128)
    DETECTOR_CONFIG.setdefault('leaky_relu_alpha', 0.1)
    DETECTOR_CONFIG.setdefault('fpn_anchor_configs', {
        'P3': {'num_anchors_this_level': 3}, 'P4': {'num_anchors_this_level': 3}, 'P5': {'num_anchors_this_level': 3}
    })
    DETECTOR_CONFIG.setdefault('l2_regularization', None)

INPUT_SHAPE_MODEL_CFG = tuple(DETECTOR_CONFIG.get('input_shape', [416, 416, 3]))
BACKBONE_NAME_MODEL_CFG = DETECTOR_CONFIG.get('backbone_name', 'MobileNetV2')
FREEZE_BACKBONE_INITIAL_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)
FPN_FILTERS_CFG = DETECTOR_CONFIG.get('fpn_filters', 128)
HEAD_DEPTH_CFG = DETECTOR_CONFIG.get('head_depth', 2)
HEAD_CONV_FILTERS_CFG = DETECTOR_CONFIG.get('head_conv_filters', 128)
LEAKY_RELU_ALPHA_CFG = DETECTOR_CONFIG.get('leaky_relu_alpha', 0.1)
L2_REG_CFG = DETECTOR_CONFIG.get('l2_regularization', None)
NUM_CLASSES_MODEL_CFG = DETECTOR_CONFIG.get('num_classes', 2)
FPN_ANCHOR_CONFIGS_CFG = DETECTOR_CONFIG.get('fpn_anchor_configs', {})
P3_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_CFG.get('P3', {}).get('num_anchors_this_level', 3)
P4_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_CFG.get('P4', {}).get('num_anchors_this_level', 3)
P5_ANCHORS_PER_LOC = FPN_ANCHOR_CONFIGS_CFG.get('P5', {}).get('num_anchors_this_level', 3)


def conv_bn_leaky(x, filters, kernel_size, strides=1, name_prefix=""):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG_CFG) if L2_REG_CFG else None, name=f"{name_prefix}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA_CFG, name=f"{name_prefix}_leakyrelu")(x)
    return x


def prediction_head(input_tensor, head_id_str, grid_h, grid_w,
                    num_anchors_at_this_level, num_classes, head_depth, head_filters):
    x = input_tensor
    for i in range(head_depth):
        x = conv_bn_leaky(x, filters=head_filters, kernel_size=3, name_prefix=f"head_{head_id_str}_body_conv{i + 1}")

    num_predictions_per_anchor = 4 + 1 + num_classes
    raw_predictions = tf.keras.layers.Conv2D(
        filters=num_anchors_at_this_level * num_predictions_per_anchor, kernel_size=1,
        padding='same', activation=None, name=f"head_{head_id_str}_raw_preds"
    )(x)

    # Для Reshape используем известные/вычисленные grid_h и grid_w
    output_shape_tuple = (grid_h, grid_w, num_anchors_at_this_level, num_predictions_per_anchor)
    reshaped_predictions = tf.keras.layers.Reshape(output_shape_tuple, name=f"head_{head_id_str}_predictions")(
        raw_predictions)
    return reshaped_predictions


def build_object_detector_v2_fpn():
    inputs = tf.keras.Input(shape=INPUT_SHAPE_MODEL_CFG, name="input_image_detector_fpn")

    if BACKBONE_NAME_MODEL_CFG == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            input_tensor=inputs, include_top=False, weights='imagenet'
        )
        try:
            c3_output_name = 'block_6_expand_relu'
            c4_output_name = 'block_13_expand_relu'
            c5_output_name = 'out_relu'
            c3 = base_model.get_layer(c3_output_name).output
            c4 = base_model.get_layer(c4_output_name).output
            c5 = base_model.get_layer(c5_output_name).output
            print(
                f"INFO: Слои для FPN из MobileNetV2: C3='{c3_output_name}', C4='{c4_output_name}', C5='{c5_output_name}'")
        except ValueError as e:
            print(f"ОШИБКА: Не удалось найти один из слоев backbone для FPN: {e}")
            print(
                "Пожалуйста, проверьте имена слоев в MobileNetV2 для вашего input_shape с помощью base_model.summary().")
            # Попытка взять слои по индексам (менее надежно и требует проверки summary)
            # Эти индексы могут сильно варьироваться! Это лишь ПРИМЕР для MobileNetV2 (224x224)
            # Для входа 416x416 индексы будут другие.
            # print("Попытка взять слои по индексам (требует проверки model.summary()):")
            # mobilenet_temp = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE_MODEL_CFG, include_top=False, weights=None) # Без весов, только для summary
            # mobilenet_temp.summary()
            # c3 = base_model.layers[54].output # Примерный индекс для stride 8
            # c4 = base_model.layers[116].output # Примерный индекс для stride 16
            # c5 = base_model.layers[-1].output # Обычно последний слой
            # print(f"INFO: (Попытка) Слои для FPN из MobileNetV2 по индексам: C3='{c3.name}', C4='{c4.name}', C5='{c5.name}'")
            raise e  # Перевыбрасываем ошибку, так как без правильных слоев FPN не построить

        # Создаем модель backbone на основе уже переданного `inputs` в `base_model`
        backbone_model_fpn = tf.keras.Model(inputs=inputs, outputs=[c3, c4, c5], name="Backbone_MobileNetV2_FPN")

    else:
        raise ValueError(f"Неподдерживаемый backbone для FPN: {BACKBONE_NAME_MODEL_CFG}")

    if FREEZE_BACKBONE_INITIAL_CFG:
        backbone_model_fpn.trainable = False
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' ЗАМОРОЖЕН.")
    else:
        backbone_model_fpn.trainable = True
        print(f"INFO (FPN Model): Backbone '{BACKBONE_NAME_MODEL_CFG}' РАЗМОРОЖЕН.")

    c3_features, c4_features, c5_features = backbone_model_fpn.outputs

    # FPN - Шея
    p5_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c5_to_p5_in')(c5_features)
    p4_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c4_to_p4_in')(c4_features)
    p3_in = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=1, padding='same', name='fpn_c3_to_p3_in')(c3_features)

    p5_up = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p5_upsampled')(p5_in)
    p4_merged = tf.keras.layers.Add(name='fpn_p4_merged')([p5_up, p4_in])

    p4_up = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p4_upsampled')(p4_merged)
    p3_merged = tf.keras.layers.Add(name='fpn_p3_merged')([p4_up, p3_in])

    p5 = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p5_output')(
        p5_in)  # Используем p5_in для P5
    p4 = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p4_output')(p4_merged)
    p3 = tf.keras.layers.Conv2D(FPN_FILTERS_CFG, kernel_size=3, padding='same', name='fpn_p3_output')(p3_merged)

    # Prediction Heads
    grid_h_p3, grid_w_p3 = INPUT_SHAPE_MODEL_CFG[0] // 8, INPUT_SHAPE_MODEL_CFG[1] // 8
    grid_h_p4, grid_w_p4 = INPUT_SHAPE_MODEL_CFG[0] // 16, INPUT_SHAPE_MODEL_CFG[1] // 16
    grid_h_p5, grid_w_p5 = INPUT_SHAPE_MODEL_CFG[0] // 32, INPUT_SHAPE_MODEL_CFG[1] // 32

    predictions_p3 = prediction_head(p3, "P3", grid_h_p3, grid_w_p3, P3_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)
    predictions_p4 = prediction_head(p4, "P4", grid_h_p4, grid_w_p4, P4_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)
    predictions_p5 = prediction_head(p5, "P5", grid_h_p5, grid_w_p5, P5_ANCHORS_PER_LOC, NUM_CLASSES_MODEL_CFG,
                                     HEAD_DEPTH_CFG, HEAD_CONV_FILTERS_CFG)

    model_outputs = [predictions_p3, predictions_p4, predictions_p5]
    model = tf.keras.Model(inputs=inputs, outputs=model_outputs, name=f"{BACKBONE_NAME_MODEL_CFG}_FPN_Detector_v2")
    return model


if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Версия FPN) ---")
    try:
        with open(_detector_config_path, 'r', encoding='utf-8') as f:
            DETECTOR_CONFIG_TEST = yaml.safe_load(f)
        input_shape_test = tuple(DETECTOR_CONFIG_TEST.get('input_shape', [416, 416, 3]))
        num_classes_test = DETECTOR_CONFIG_TEST.get('num_classes', 2)
        fpn_anchor_cfgs_test = DETECTOR_CONFIG_TEST.get('fpn_anchor_configs', {})
        p3_anchors_test = fpn_anchor_cfgs_test.get('P3', {}).get('num_anchors_this_level', 3)
        p4_anchors_test = fpn_anchor_cfgs_test.get('P4', {}).get('num_anchors_this_level', 3)
        p5_anchors_test = fpn_anchor_cfgs_test.get('P5', {}).get('num_anchors_this_level', 3)
        print(f"Конфигурация входа для теста: {input_shape_test}")
        print(f"Количество классов: {num_classes_test}")
    except Exception as e:
        print(f"Ошибка загрузки конфига для теста: {e}. Используем глобальные переменные модуля.")
        input_shape_test = INPUT_SHAPE_MODEL_CFG;
        num_classes_test = NUM_CLASSES_MODEL_CFG
        p3_anchors_test = P3_ANCHORS_PER_LOC;
        p4_anchors_test = P4_ANCHORS_PER_LOC;
        p5_anchors_test = P5_ANCHORS_PER_LOC
    try:
        detector_model_fpn = build_object_detector_v2_fpn()
        print("\nСтруктура модели детектора с FPN:");
        detector_model_fpn.summary(line_length=150)
        print("\nТестовый прогон модели FPN (проверка форм выходов):")
        batch_size_for_test = 1
        dummy_input_shape_test_fpn = (
        batch_size_for_test, input_shape_test[0], input_shape_test[1], input_shape_test[2])
        dummy_input_fpn = tf.random.normal(dummy_input_shape_test_fpn)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input_fpn.shape}")
        start_time = time.time();
        predictions_list = detector_model_fpn(dummy_input_fpn, training=False);
        end_time = time.time()
        print(f"  Время инференса на фиктивном входе: {end_time - start_time:.4f} сек.")
        if isinstance(predictions_list, list) and len(predictions_list) == 3:
            print(f"\n  Модель вернула список из {len(predictions_list)} тензоров (P3, P4, P5):")
            expected_grid_h_p3, expected_grid_w_p3 = input_shape_test[0] // 8, input_shape_test[1] // 8
            expected_grid_h_p4, expected_grid_w_p4 = input_shape_test[0] // 16, input_shape_test[1] // 16
            expected_grid_h_p5, expected_grid_w_p5 = input_shape_test[0] // 32, input_shape_test[1] // 32
            expected_preds_per_anchor = 4 + 1 + num_classes_test
            expected_shapes = [
                (batch_size_for_test, expected_grid_h_p3, expected_grid_w_p3, p3_anchors_test,
                 expected_preds_per_anchor),
                (batch_size_for_test, expected_grid_h_p4, expected_grid_w_p4, p4_anchors_test,
                 expected_preds_per_anchor),
                (batch_size_for_test, expected_grid_h_p5, expected_grid_w_p5, p5_anchors_test,
                 expected_preds_per_anchor)]
            level_names = ["P3", "P4", "P5"];
            all_shapes_match = True
            for i, pred_tensor in enumerate(predictions_list):
                print(f"    Форма выхода {level_names[i]}: {pred_tensor.shape}, Ожидаемая: {expected_shapes[i]}")
                # Сравниваем формы как списки, чтобы избежать проблем с None в TensorShape
                if pred_tensor.shape.as_list() != list(expected_shapes[i]):
                    all_shapes_match = False;
                    print(f"      ОШИБКА: Форма выхода {level_names[i]} НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ!")
            if all_shapes_match:
                print("\n  Все формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
            else:
                print("\n  ОШИБКА: НЕ ВСЕ формы выходных тензоров FPN СООТВЕТСТВУЮТ ОЖИДАЕМЫМ.")
        else:
            print(f"  ОШИБКА: Модель вернула не список из 3х тензоров, а: {type(predictions_list)}")
    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели FPN детектора: {e}");
        import traceback;

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py (FPN) завершено ---")