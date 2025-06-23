import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path
import sys

# --- Настройка путей ---
try:
    # Этот путь будет работать, если скрипт лежит в корне или в подпапке
    current_path = Path(__file__).resolve()
    PROJECT_ROOT = current_path
    while not (PROJECT_ROOT / 'src').exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
        if PROJECT_ROOT == PROJECT_ROOT.parent: raise FileNotFoundError
except (NameError, FileNotFoundError):
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Импорты из нашего проекта ---
try:
    from src.datasets.data_loader_v3_standard import generate_all_anchors
except ImportError:
    # Для случая, когда этот файл используется отдельно, и data_loader недоступен
    # Мы можем определить generate_all_anchors прямо здесь, если нужно.
    # Но для теста лучше положиться на общий код.
    print("Не удалось импортировать generate_all_anchors. Тест может не сработать.")
    generate_all_anchors = None


# --- Функции Постобработки ---

def decode_predictions(raw_preds, all_anchors, detector_config):
    """
    Декодирует "сырые" выходы модели (смещения) в реальные координаты боксов.
    """
    num_classes = detector_config['num_classes']
    num_levels = len(raw_preds) // 2

    # Шаг 1: Конкатенация. Это более явно и чисто, чем if/else.
    # Мы всегда ожидаем список из 6 тензоров.
    y_pred_reg_flat = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in raw_preds[:num_levels]], axis=1)
    y_pred_cls_flat = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, num_classes]) for p in raw_preds[num_levels:]],
                                axis=1)

    # Шаг 2: Декодирование. Нет необходимости в expand_dims, TensorFlow сам справится с "вещанием" (broadcasting).
    # all_anchors (N, 4) будет "вещаться" на y_pred_reg_flat (B, N, 4).
    anchor_w = all_anchors[:, 2] - all_anchors[:, 0]
    anchor_h = all_anchors[:, 3] - all_anchors[:, 1]
    anchor_cx = all_anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = all_anchors[:, 1] + 0.5 * anchor_h

    tx, ty, tw, th = tf.unstack(y_pred_reg_flat, axis=-1)

    pred_cx = tx * anchor_w + anchor_cx
    pred_cy = ty * anchor_h + anchor_cy
    pred_w = tf.exp(tw) * anchor_w
    pred_h = tf.exp(th) * anchor_h

    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h

    # Формат [y1, x1, y2, x2] для NMS - это правильно.
    decoded_boxes = tf.stack([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1)

    decoded_scores = tf.nn.sigmoid(y_pred_cls_flat)

    return decoded_boxes, decoded_scores


def perform_nms(decoded_boxes, decoded_scores, predict_config):
    """
    Выполняет Non-Maximum Suppression.
    """
    # Шаг 3: Безопасное получение параметров с ПРАВИЛЬНЫМИ типами.
    score_threshold = float(predict_config.get('default_conf_thresh', 0.3))
    iou_threshold = float(predict_config.get('default_iou_thresh', 0.45))
    # `max_detections` ДОЛЖЕН быть ЦЕЛЫМ числом.
    max_detections = int(predict_config.get('default_max_dets', 100))

    return tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(decoded_boxes, axis=2),
        scores=decoded_scores,
        max_output_size_per_class=max_detections,
        max_total_size=max_detections,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        clip_boxes=False
    )

# --- Тестовый блок ---
if __name__ == '__main__':
    print("--- Тестирование функций постобработки ---")

    # [ИСПРАВЛЕНИЕ] Загружаем оба конфига, как в реальном приложении
    try:
        detector_config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(detector_config_path, 'r', encoding='utf-8') as f:
            detector_config = yaml.safe_load(f)

        predict_config_path = PROJECT_ROOT / "src" / "configs" / "predict_config.yaml"
        with open(predict_config_path, 'r', encoding='utf-8') as f:
            predict_config = yaml.safe_load(f)
        print("Конфиги успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки конфигов: {e}");
        sys.exit(1)

    if generate_all_anchors is None:
        print("Тест не может быть выполнен без функции generate_all_anchors.");
        sys.exit(1)

    # --- Генерация фиктивных данных ---
    all_anchors = generate_all_anchors(
        detector_config['input_shape'], [8, 16, 32],
        detector_config['anchor_scales'], detector_config['anchor_ratios']
    )
    all_anchors = tf.constant(all_anchors, dtype=tf.float32)
    num_total_anchors = all_anchors.shape[0]
    print(f"Сгенерировано {num_total_anchors} якорей.")

    # Создаем "сырой" выход модели
    batch_size = 1
    num_classes = detector_config['num_classes']
    # Создаем один "сильный" предикт для класса 0 и якоря 5000
    raw_cls_preds = np.random.randn(batch_size, num_total_anchors, num_classes).astype(np.float32) * 0.1
    raw_cls_preds[0, 5000, 0] = 5.0  # Сильный логит ~0.99 после sigmoid

    raw_reg_preds = np.random.randn(batch_size, num_total_anchors, 4).astype(np.float32) * 0.01

    # Разбиваем на 6 тензоров, как это делает модель
    # (упрощенная логика для теста)
    reg_p3, reg_p4, reg_p5 = tf.split(raw_reg_preds, 3, axis=1)
    cls_p3, cls_p4, cls_p5 = tf.split(raw_cls_preds, 3, axis=1)
    raw_preds_list = [reg_p3, reg_p4, reg_p5, cls_p3, cls_p4, cls_p5]

    # --- Тест decode_predictions ---
    print("\n--- Тест decode_predictions ---")
    decoded_boxes, decoded_scores = decode_predictions(raw_preds_list, all_anchors, detector_config)
    print(f"Форма декодированных боксов: {decoded_boxes.shape}")
    print(f"Форма декодированных скоров: {decoded_scores.shape}")
    print(f"Декодированный скор для якоря 5000, класс 0: {decoded_scores[0, 5000, 0].numpy():.4f}")
    assert decoded_scores[0, 5000, 0].numpy() > 0.99

    # --- Тест perform_nms ---
    print("\n--- Тест perform_nms ---")
    # Используем predict_config, как и положено
    nms_boxes, nms_scores, nms_classes, valid_detections = perform_nms(decoded_boxes, decoded_scores, predict_config)
    print(f"Форма nms_boxes: {nms_boxes.shape}")
    print(f"Форма nms_scores: {nms_scores.shape}")
    print(f"Форма nms_classes: {nms_classes.shape}")
    print(f"Количество найденных объектов (valid_detections): {valid_detections.numpy()[0]}")

    assert valid_detections.numpy()[0] >= 1, "Должен был найтись хотя бы один объект"
    print("Найденный класс:", detector_config['class_names'][int(nms_classes.numpy()[0, 0])])
    print("Уверенность:", nms_scores.numpy()[0, 0])

    print("\n[SUCCESS] Все тесты постобработки пройдены.")