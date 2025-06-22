import tensorflow as tf
import numpy as np
import sys
from pathlib import Path
import yaml

# --- Настройка путей для импорта ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import generate_all_anchors
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)


# --- Функции Постобработки ---

def decode_predictions(y_pred, all_anchors, config):
    """Декодирует "сырые" предсказания модели в реальные координаты рамок и скоры."""
    num_classes = config['num_classes']

    if isinstance(y_pred, (list, tuple)):
        num_levels = len(y_pred) // 2
        pred_reg = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in y_pred[:num_levels]], axis=1)
        pred_cls = tf.concat([tf.reshape(p, [tf.shape(p)[0], -1, num_classes]) for p in y_pred[num_levels:]], axis=1)
    else:
        pred_reg, pred_cls = y_pred

    scores = tf.nn.sigmoid(pred_cls)

    # ===> ИСПРАВЛЕНИЕ ЗДЕСЬ 1 <===
    # Добавляем новую ось к якорям, чтобы они были совместимы с (batch, num_anchors, 4)
    all_anchors = tf.expand_dims(all_anchors, axis=0)  # (1, num_anchors, 4)

    anchor_w = all_anchors[..., 2] - all_anchors[..., 0]
    anchor_h = all_anchors[..., 3] - all_anchors[..., 1]
    anchor_cx = all_anchors[..., 0] + 0.5 * anchor_w
    anchor_cy = all_anchors[..., 1] + 0.5 * anchor_h

    pred_cx = pred_reg[..., 0] * anchor_w + anchor_cx
    pred_cy = pred_reg[..., 1] * anchor_h + anchor_cy
    pred_w = tf.exp(pred_reg[..., 2]) * anchor_w
    pred_h = tf.exp(pred_reg[..., 3]) * anchor_h

    pred_ymin = pred_cy - pred_h / 2.
    pred_xmin = pred_cx - pred_w / 2.
    pred_ymax = pred_cy + pred_h / 2.
    pred_xmax = pred_cx + pred_w / 2.

    boxes = tf.stack([pred_ymin, pred_xmin, pred_ymax, pred_xmax], axis=-1)

    return boxes, scores


def perform_nms(decoded_boxes, scores, predict_config):
    """
    Выполняет Non-Maximum Suppression. Использует параметры из predict_config.
    """
    score_threshold = predict_config['default_conf_thresh']
    iou_threshold = predict_config['default_iou_thresh']
    max_total_detections = predict_config['default_max_dets']

    if len(tf.shape(decoded_boxes)) == 2:
        decoded_boxes = tf.expand_dims(decoded_boxes, axis=0)
        scores = tf.expand_dims(scores, axis=0)

    return tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(decoded_boxes, axis=2),
        scores=scores,
        max_output_size_per_class=max_total_detections,
        max_total_size=max_total_detections,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

# --- Тестовый блок ---
if __name__ == '__main__':
    print("--- Тестирование функций постобработки ---")

    try:
        config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            test_config = yaml.safe_load(f)
        print("Конфиг успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
        exit()

    all_anchors_np = generate_all_anchors(
        test_config['input_shape'], [8, 16, 32],
        test_config['anchor_scales'], test_config['anchor_ratios']
    )
    all_anchors_tf = tf.constant(all_anchors_np, dtype=tf.float32)
    num_total_anchors = all_anchors_tf.shape[0]
    print(f"Сгенерировано {num_total_anchors} якорей.")

    BATCH_SIZE = 1
    NUM_CLASSES = test_config['num_classes']
    pred_reg = np.zeros((BATCH_SIZE, num_total_anchors, 4), dtype=np.float32)
    pred_cls = np.full((BATCH_SIZE, num_total_anchors, NUM_CLASSES), -5.0, dtype=np.float32)

    pred_reg[0, 5000, :] = [0.1, 0.1, 0.2, 0.2]
    pred_cls[0, 5000, 0] = 5.0
    pred_reg[0, 5001, :] = [0.11, 0.11, 0.19, 0.19]
    pred_cls[0, 5001, 0] = 4.0
    pred_reg[0, 40000, :] = [-0.05, -0.05, 0.0, 0.0]
    pred_cls[0, 40000, 1] = 6.0

    pred_reg_tf = tf.constant(pred_reg)
    pred_cls_tf = tf.constant(pred_cls)

    print("\n--- Тест decode_predictions ---")
    decoded_boxes, decoded_scores = decode_predictions([pred_reg_tf, pred_cls_tf], all_anchors_tf, test_config)
    print(f"Форма декодированных боксов: {decoded_boxes.shape}")
    print(f"Форма декодированных скоров: {decoded_scores.shape}")

    box_5000_decoded = decoded_boxes[0, 5000].numpy()
    score_5000_decoded = decoded_scores[0, 5000].numpy()
    print(f"Декодированный скор для якоря 5000, класс 0: {score_5000_decoded[0]:.4f}")

    print("\n--- Тест perform_nms ---")
    # Передаем весь батч
    nms_boxes, nms_scores, nms_classes, valid_detections = perform_nms(decoded_boxes, decoded_scores, test_config)

    # Результат NMS - это тензор для всего батча, берем срез [0] для первого элемента
    valid_detections = valid_detections[0]
    print(f"Найдено валидных детекций после NMS: {valid_detections.numpy()}")

    for i in range(valid_detections.numpy()):
        print(f"\nДетекция #{i + 1}:")
        box = nms_boxes[0, i].numpy()  # Берем срез [0] по батчу
        score = nms_scores[0, i].numpy()
        class_id = int(nms_classes[0, i].numpy())
        print(f"  - Класс: {test_config['class_names'][class_id]} (id: {class_id})")
        print(f"  - Уверенность: {score:.4f}")
        print(f"  - Рамка [ymin, xmin, ymax, xmax]: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")

    assert valid_detections.numpy() == 2, "Должно остаться 2 детекции после NMS"
    print("\n[SUCCESS] Тесты постобработки пройдены.")