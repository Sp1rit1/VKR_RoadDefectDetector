# src/models/object_detector.py
import tensorflow as tf
import yaml
import os

# --- Загрузка Конфигурации ---
_current_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
_detector_config_path = os.path.join(_current_dir, '..', 'configs', 'detector_config.yaml')
_base_config_path = os.path.join(_current_dir, '..', 'configs',
                                 'base_config.yaml')  # Может понадобиться для input_shape

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    with open(_base_config_path, 'r', encoding='utf-8') as f:  # Загружаем и базовый для общих параметров
        BASE_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ОШИБКА: Файл конфигурации детектора или базовый не найден (object_detector.py).")
    # Задаем дефолты, чтобы модуль хотя бы импортировался
    DETECTOR_CONFIG = {'backbone_name': 'MobileNetV2',
                       'freeze_backbone': True,
                       'num_classes': 2,
                       'head_conv_filters': 256,  # Увеличим немного фильтры в голове
                       'num_anchors_per_location': 3,
                       'train_params': {'batch_size': 1}}
    BASE_CONFIG = {'model_params': {'target_height': 416, 'target_width': 416}}
    print("ПРЕДУПРЕЖДЕНИЕ: Файлы конфигурации не найдены, используются значения по умолчанию для object_detector.py.")
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл: {e} (object_detector.py)")
    DETECTOR_CONFIG = {'backbone_name': 'MobileNetV2',
                       'freeze_backbone': True,
                       'num_classes': 2,
                       'head_conv_filters': 256,
                       'num_anchors_per_location': 3,
                       'train_params': {'batch_size': 1}}
    BASE_CONFIG = {'model_params': {'target_height': 416, 'target_width': 416}}
    print("ПРЕДУПРЕЖДЕНИЕ: Ошибка чтения YAML, используются значения по умолчанию (object_detector.py).")

# Используем input_shape из detector_config, если есть, иначе из base_config
if 'input_shape' in DETECTOR_CONFIG:
    INPUT_SHAPE_CFG = tuple(DETECTOR_CONFIG['input_shape'])
elif 'model_params' in BASE_CONFIG and 'input_shape' in BASE_CONFIG['model_params']:
    INPUT_SHAPE_CFG = tuple(BASE_CONFIG['model_params']['input_shape'])
elif 'input_shape' in BASE_CONFIG.get('model_params', {}):  # Еще одна попытка для старого base_config
    INPUT_SHAPE_CFG = tuple(BASE_CONFIG['model_params']['input_shape'])
else:  # Если нигде нет, ставим дефолт, совпадающий с target_height/width
    INPUT_SHAPE_CFG = (BASE_CONFIG.get('model_params', {}).get('target_height', 416),
                       BASE_CONFIG.get('model_params', {}).get('target_width', 416),
                       3)

NUM_CLASSES_CFG = DETECTOR_CONFIG.get('num_classes', 2)
BACKBONE_NAME_CFG = DETECTOR_CONFIG.get('backbone_name', 'MobileNetV2')
FREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('freeze_backbone', True)
HEAD_CONV_FILTERS_CFG = DETECTOR_CONFIG.get('head_conv_filters', 256)  # Увеличили дефолт
NUM_ANCHORS_CFG = DETECTOR_CONFIG.get('num_anchors_per_location', 3)


def build_object_detector_v1():  # Назовем v1, так как это первая "правильная" версия
    """
    Строит модель детектора объектов.
    Backbone (MobileNetV2) -> Несколько сверточных слоев -> Голова предсказаний.
    Голова предсказывает для сетки и якорей: objectness, координаты, классы.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE_CFG, name="input_image_detector")

    # 1. Backbone
    if BACKBONE_NAME_CFG == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE_CFG,
                                                       include_top=False,
                                                       weights='imagenet')
        # Архитектурное решение: Берем выход слоя, который уменьшает размер в 16 раз.
        # Для MobileNetV2 с входом 224x224, 'block_13_expand_relu' имеет выход 7x7.
        # Если вход 416x416, то 416/16 = 26, значит выход будет ~26x26.
        # Если вход 320x320, то 320/16 = 20, значит выход будет ~20x20.
        # Нам нужно проверить имя слоя для выбранного INPUT_SHAPE_CFG.
        # Для MobileNetV2 'block_13_expand_relu' (индекс -28 у модели без include_top) часто хороший выбор.
        # Или можно взять более глубокий 'out_relu' (последний слой перед GlobalAveragePooling, если бы он был),
        # но его разрешение будет меньше (например, /32).
        # Давайте для начала возьмем 'block_13_expand_relu' для большего разрешения (меньший stride).
        # Если его нет для данного input_shape, MobileNetV2 вернет ошибку при base_model.get_layer.
        # Более надежно брать по индексу, если знаем структуру, или последний слой.
        # Для MobileNetV2, если include_top=False, последний слой это 'out_relu'.
        # Но 'block_13_expand_relu' дает большее разрешение (stride 16).
        # Если INPUT_SHAPE_CFG сильно отличается от стандартных, имена слоев могут быть другие.
        # Для простоты, если конкретный слой не найдется, возьмем просто выход базовой модели.
        try:
            # Попытка взять слой, дающий разрешение /16
            # Для входа (224,224,3) 'block_13_expand_relu' имеет выход (None, 14, 14, 96) - нет, это /16 от 224. 224/16 = 14
            # Для входа (416,416,3) этот слой даст (None, 26, 26, 96)
            # Для входа (320,320,3) этот слой даст (None, 20, 20, 96)
            # В MobileNetV2 слои с expand_relu это выходы bottleneck блоков.
            # block_13_expand_relu (если считать с 1) соответствует stride 16.
            # block_16_project_BN (последний слой перед финальной сверткой и пулингом) соответствует stride 32.
            # Мы хотим разрешение побольше для детекции мелких объектов.
            # `layer.name` 'block_13_expand_relu'
            feature_extraction_layer_name = 'block_13_expand_relu'
            backbone_output_tensor = base_model.get_layer(feature_extraction_layer_name).output
            print(f"INFO: Используется выход слоя '{feature_extraction_layer_name}' из MobileNetV2.")
        except ValueError:
            print(
                f"WARNING: Слой '{feature_extraction_layer_name}' не найден в MobileNetV2 для input_shape {INPUT_SHAPE_CFG}.")
            print("         Будет использован последний выходной слой базовой модели.")
            backbone_output_tensor = base_model.output

        backbone = tf.keras.Model(inputs=base_model.input, outputs=backbone_output_tensor, name="Backbone_MobileNetV2")

    else:  # Добавь сюда другие варианты backbone, если будешь экспериментировать
        raise ValueError(f"Неподдерживаемый backbone: {BACKBONE_NAME_CFG}")

    if FREEZE_BACKBONE_CFG:
        backbone.trainable = False
        print(f"INFO: Backbone '{BACKBONE_NAME_CFG}' ЗАМОРОЖЕН.")
    else:
        backbone.trainable = True
        print(f"INFO: Backbone '{BACKBONE_NAME_CFG}' РАЗМОРОЖЕН.")

    # Применяем backbone к нашим входам
    x = backbone(inputs, training=(not FREEZE_BACKBONE_CFG))
    # training флаг важен для BatchNormalization, если backbone разморожен

    # 2. Дополнительные сверточные слои ("тело" детектора)
    # Эти слои помогут модели изучить более специфичные признаки для задачи детекции
    # поверх общих признаков из backbone.
    x = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG, kernel_size=3, padding='same',
                               activation='relu', name="detector_body_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="detector_body_bn1")(x)
    x = tf.keras.layers.Conv2D(filters=HEAD_CONV_FILTERS_CFG // 2, kernel_size=1, padding='same',
                               activation='relu', name="detector_body_conv2_1x1")(x)  # 1x1 для смешивания каналов
    x = tf.keras.layers.BatchNormalization(name="detector_body_bn2")(x)
    # feature_map_for_head = x # Это наша финальная карта признаков для "головы"

    # 3. Голова предсказаний (Prediction Head)
    # Она будет предсказывать для каждой ячейки на feature_map_for_head и для каждого якоря:
    # - Координаты рамки (4 значения: tx, ty, tw, th - закодированные)
    # - Objectness score (1 значение: уверенность в наличии объекта)
    # - Вероятности классов (NUM_CLASSES_CFG значений)
    num_predictions_per_anchor = 4 + 1 + NUM_CLASSES_CFG
    total_output_filters = NUM_ANCHORS_CFG * num_predictions_per_anchor

    # Финальный сверточный слой без активации (логиты)
    raw_head_output = tf.keras.layers.Conv2D(
        filters=total_output_filters,
        kernel_size=1,  # 1x1 свертка для предсказаний
        padding='same',
        activation=None,  # Логиты, активации (sigmoid/softmax) будут применяться в функции потерь или при инференсе
        name="detection_head_predictions"
    )(x)  # feature_map_for_head теперь просто x

    # Решейпим выход в понятный формат: (batch_size, grid_height, grid_width, num_anchors, num_predictions_per_anchor)
    # grid_height и grid_width будут такими же, как у `x` (выхода detector_body_bn2)
    # tf.shape(x)[1] это grid_height, tf.shape(x)[2] это grid_width
    # Но для Reshape лучше использовать известные измерения, если они есть, или -1 для автоматического вывода

    # Получаем пространственные размеры карты признаков 'x'
    # Эти размеры будут зависеть от INPUT_SHAPE_CFG и архитектуры backbone/тела
    # Например, если x имеет форму (None, 13, 13, 128), то grid_h=13, grid_w=13
    # Для Reshape нужно знать эти размеры или использовать -1, если одно из измерений может быть выведено.
    # Мы можем не указывать их явно в Reshape, если Keras сможет их вывести из `total_output_filters`
    # и `NUM_ANCHORS_CFG` и `num_predictions_per_anchor`.
    # Предположим, что Keras справится, если мы укажем последние два измерения.
    # Выход raw_head_output будет (batch_size, grid_h, grid_w, total_output_filters)

    # Правильный Reshape:
    # Сначала получаем grid_h, grid_w динамически, если возможно, или из конфига, если мы их знаем заранее
    # Для нашего примера с MobileNetV2 и выходом /16:
    # grid_h = INPUT_SHAPE_CFG[0] // 16
    # grid_w = INPUT_SHAPE_CFG[1] // 16
    # Но лучше, если слой Reshape сам это определит, если мы правильно зададим остальные измерения.

    # Мы хотим (batch, H, W, A, Preds_per_A)
    # raw_head_output.shape = (batch, H, W, A * Preds_per_A)
    # Значит, нужно (-1, H_из_x, W_из_x, NUM_ANCHORS_CFG, num_predictions_per_anchor)
    # где H_из_x и W_из_x - это tf.shape(x)[1] и tf.shape(x)[2]

    # Используем слой Reshape. Он должен знать H и W.
    # Если мы не передаем H и W явно, то одно из измерений в target_shape может быть -1.
    # В нашем случае, если Conv2D выдает (B, H, W, A*P), то Reshape((H, W, A, P))
    # Keras должен справиться с этим, если H и W не меняются динамически от батча к батчу (что обычно так).

    predictions_reshaped = tf.keras.layers.Reshape(
        (-1, NUM_ANCHORS_CFG, num_predictions_per_anchor),  # -1 здесь означает, что это измерение будет вычислено (H*W)
        # но это может не сработать, если H и W не фиксированы при построении графа
        name="predictions_reshaped_raw"
    )(raw_head_output)

    # Более надежный Reshape, если мы знаем H и W (или можем их получить из x.shape)
    # Это будет работать лучше, если пространственные размеры x известны на момент построения модели
    # (что обычно так, если input_shape фиксирован).
    # Например, если x.shape = (None, 13, 13, 128), то:
    # predictions_reshaped = tf.keras.layers.Reshape(
    #     (x.shape[1], x.shape[2], NUM_ANCHORS_CFG, num_predictions_per_anchor),
    #     name="predictions_reshaped"
    # )(raw_head_output)
    # Однако, x.shape[1] и x.shape[2] могут быть None на этапе построения графа.
    # В этом случае, мы можем либо положиться на вывод формы через -1,
    # либо вычислить grid_h, grid_w из INPUT_SHAPE_CFG и stride сети.

    # YOLO часто делает так: выход (B, H, W, A*(5+C)). Затем при необходимости можно разделить на части.
    # Это и есть наш raw_head_output. Для простоты, пусть модель возвращает его.
    # А Reshape в (B, H, W, A, 5+C) сделаем либо здесь, либо при пост-обработке/в лоссах.
    # Давай сделаем Reshape здесь для ясности.
    # Нам нужно знать H и W сетки. Они зависят от входного размера и stride сети.
    # Если вход 416, stride 16, то H=W=26. Если stride 32, то H=W=13.
    # MobileNetV2 с выходом 'block_13_expand_relu' дает stride 16.
    grid_h = INPUT_SHAPE_CFG[0] // 16
    grid_w = INPUT_SHAPE_CFG[1] // 16

    final_output_shape = (grid_h, grid_w, NUM_ANCHORS_CFG, num_predictions_per_anchor)
    predictions_final_reshaped = tf.keras.layers.Reshape(
        final_output_shape, name="detector_predictions"
    )(raw_head_output)

    model = tf.keras.Model(inputs=inputs, outputs=predictions_final_reshaped,
                           name=f"{BACKBONE_NAME_CFG}_ObjectDetector_v1")
    return model


if __name__ == '__main__':
    print(f"--- Тестирование object_detector.py (Версия с сеткой и якорями) ---")
    print(f"Конфигурация входа: {INPUT_SHAPE_CFG}")
    print(f"Количество классов: {NUM_CLASSES_CFG}")
    print(f"Количество якорей на ячейку: {NUM_ANCHORS_CFG}")

    try:
        detector_model = build_object_detector_v1()
        print("\nСтруктура модели детектора:")
        detector_model.summary(line_length=120)

        print("\nТестовый прогон модели (проверка формы выхода):")
        # Используем batch_size из detector_config или дефолт
        batch_size_for_test = DETECTOR_CONFIG.get('train_params', {}).get('batch_size', 1)

        dummy_input_shape_test = (
            batch_size_for_test,
            INPUT_SHAPE_CFG[0],
            INPUT_SHAPE_CFG[1],
            INPUT_SHAPE_CFG[2]
        )
        dummy_input = tf.random.normal(dummy_input_shape_test)
        print(f"  Создан фиктивный входной тензор с формой: {dummy_input.shape}")

        predictions = detector_model(dummy_input)

        # Ожидаемая форма выхода
        expected_grid_h = INPUT_SHAPE_CFG[0] // 16  # Зависит от stride backbone и "тела"
        expected_grid_w = INPUT_SHAPE_CFG[1] // 16
        expected_output_features_per_anchor = 4 + 1 + NUM_CLASSES_CFG

        print(f"  Форма предсказаний: {predictions.shape}")
        print(
            f"  Ожидаемая форма: ({batch_size_for_test}, {expected_grid_h}, {expected_grid_w}, {NUM_ANCHORS_CFG}, {expected_output_features_per_anchor})")

        if predictions.shape == (
        batch_size_for_test, expected_grid_h, expected_grid_w, NUM_ANCHORS_CFG, expected_output_features_per_anchor):
            print("  Форма выхода СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")
        else:
            print("  ОШИБКА: Форма выхода НЕ СООТВЕТСТВУЕТ ОЖИДАЕМОЙ.")

    except Exception as e:
        print(f"ОШИБКА при создании или тестировании модели детектора: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование object_detector.py завершено ---")