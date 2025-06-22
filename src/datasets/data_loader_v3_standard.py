import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from . import augmentations


# --- Блок 1: Функции-утилиты ---
# (Без изменений)
def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, class_names = [], []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin, ymin = int(float(bndbox.find('xmin').text)), int(float(bndbox.find('ymin').text))
        xmax, ymax = int(float(bndbox.find('xmax').text)), int(float(bndbox.find('ymax').text))
        boxes.append([xmin, ymin, xmax, ymax])
        class_names.append(class_name)
    return boxes, class_names


def generate_all_anchors(input_shape, fpn_strides, anchor_scales, anchor_ratios):
    all_anchors = []
    image_height, image_width = input_shape[0], input_shape[1]
    for stride in fpn_strides:
        base_anchor_size = stride * 4
        feature_map_height, feature_map_width = image_height // stride, image_width // stride
        for y_fm in range(feature_map_height):
            for x_fm in range(feature_map_width):
                cx, cy = (x_fm + 0.5) * stride, (y_fm + 0.5) * stride
                for scale in anchor_scales:
                    for ratio in anchor_ratios:
                        h, w = base_anchor_size * scale * np.sqrt(ratio), base_anchor_size * scale / np.sqrt(ratio)
                        xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                        all_anchors.append([xmin, ymin, xmax, ymax])
    all_anchors_np = np.array(all_anchors, dtype=np.float32)
    all_anchors_np[:, [0, 2]] /= image_width
    all_anchors_np[:, [1, 3]] /= image_height
    return np.clip(all_anchors_np, 0.0, 1.0)


def calculate_iou_matrix(boxes1, boxes2):
    boxes1_np = np.array(boxes1)
    boxes2_np = np.array(boxes2)
    boxes1_ext = boxes1_np[:, np.newaxis, :]
    boxes2_ext = boxes2_np[np.newaxis, :, :]
    x1 = np.maximum(boxes1_ext[..., 0], boxes2_ext[..., 0])
    y1 = np.maximum(boxes1_ext[..., 1], boxes2_ext[..., 1])
    x2 = np.minimum(boxes1_ext[..., 2], boxes2_ext[..., 2])
    y2 = np.minimum(boxes1_ext[..., 3], boxes2_ext[..., 3])
    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h
    area1 = (boxes1_ext[..., 2] - boxes1_ext[..., 0]) * (boxes1_ext[..., 3] - boxes1_ext[..., 1])
    area2 = (boxes2_ext[..., 2] - boxes2_ext[..., 0]) * (boxes2_ext[..., 3] - boxes2_ext[..., 1])
    union_area = area1 + area2 - intersection_area
    return intersection_area / (union_area + 1e-7)


# --- Блок 2: Основная логика ---
# (Без изменений)
def assign_gt_to_anchors(gt_boxes_norm, all_anchors_norm, pos_iou_thresh, neg_iou_thresh):
    num_anchors = all_anchors_norm.shape[0]
    num_gt = gt_boxes_norm.shape[0]
    if num_gt == 0:
        anchor_labels = np.full((num_anchors,), -1, dtype=np.int32)
        matched_gt_boxes = np.zeros((num_anchors, 4), dtype=np.float32)
        max_iou_per_anchor = np.zeros((num_anchors,), dtype=np.float32)
        return anchor_labels, matched_gt_boxes, max_iou_per_anchor

    iou_matrix = calculate_iou_matrix(gt_boxes_norm, all_anchors_norm)
    max_iou_per_anchor = np.max(iou_matrix, axis=0)
    matched_gt_idx_per_anchor = np.argmax(iou_matrix, axis=0)

    anchor_labels = np.full((num_anchors,), -1, dtype=np.int32)
    anchor_labels[(max_iou_per_anchor >= neg_iou_thresh) & (max_iou_per_anchor < pos_iou_thresh)] = 0
    anchor_labels[max_iou_per_anchor >= pos_iou_thresh] = 1

    best_anchor_idx_per_gt = np.argmax(iou_matrix, axis=1)
    anchor_labels[best_anchor_idx_per_gt] = 1

    matched_gt_boxes = gt_boxes_norm[matched_gt_idx_per_anchor]
    return anchor_labels, matched_gt_boxes, max_iou_per_anchor


def encode_box_targets(anchors, matched_gt_boxes):
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
    gt_cx = matched_gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = matched_gt_boxes[:, 1] + 0.5 * gt_h
    anchor_w[anchor_w == 0] = 1e-7
    anchor_h[anchor_h == 0] = 1e-7
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)
    return np.stack([tx, ty, tw, th], axis=1)


# --- Блок 3: Класс-генератор и создание датасета ---

class DataGenerator:
    def __init__(self, config, all_anchors, is_training=True, debug_mode=False):
        self.config = config
        self.is_training = is_training
        self.debug_mode = debug_mode
        dataset_path = Path(config['dataset_path'])
        if is_training:
            self.image_dir = dataset_path / config['train_images_subdir']
            self.annot_dir = dataset_path / config['train_annotations_subdir']
        else:
            self.image_dir = dataset_path / config['val_images_subdir']
            self.annot_dir = dataset_path / config['val_annotations_subdir']
        self.image_paths = sorted([p for p in self.image_dir.glob('*.jpg')])
        self.annot_paths = sorted([p for p in self.annot_dir.glob('*.xml')])
        self.input_shape = config['input_shape']
        self.class_mapping = {name: i for i, name in enumerate(config['class_names'])}
        self.num_classes = config['num_classes']
        self.pos_iou_thresh = config['anchor_positive_iou_threshold']
        self.neg_iou_thresh = config['anchor_ignore_iou_threshold']
        self.all_anchors = all_anchors
        if self.is_training and config['use_augmentation']:
            self.augmenter = augmentations.get_detector_train_augmentations(*self.input_shape[:2])
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.image_paths)

    def __call__(self):
        for i in range(len(self)):
            image_path = self.image_paths[i]
            annot_path = self.annot_paths[i]
            image = cv2.imread(str(image_path))
            image_original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h_orig, image_w_orig, _ = image_original_rgb.shape
            gt_boxes_pixels, gt_class_names = parse_voc_xml(annot_path)

            image_resized = cv2.resize(image_original_rgb, (self.input_shape[1], self.input_shape[0]))

            gt_boxes_resized = []
            if gt_boxes_pixels:
                for box in gt_boxes_pixels:
                    xmin, ymin, xmax, ymax = box
                    xmin_r, ymin_r = (xmin / image_w_orig) * self.input_shape[1], (ymin / image_h_orig) * \
                                     self.input_shape[0]
                    xmax_r, ymax_r = (xmax / image_w_orig) * self.input_shape[1], (ymax / image_h_orig) * \
                                     self.input_shape[0]
                    gt_boxes_resized.append([xmin_r, ymin_r, xmax_r, ymax_r])

            if self.augmenter:
                augmented = self.augmenter(image=image_resized, bboxes=gt_boxes_resized,
                                           class_labels_for_albumentations=gt_class_names)
                image_final = augmented['image']
                gt_boxes_final_pixels = augmented['bboxes']
                gt_class_names_final = augmented['class_labels_for_albumentations']
            else:
                image_final = image_resized
                gt_boxes_final_pixels = gt_boxes_resized
                gt_class_names_final = gt_class_names

            image_final_norm = image_final.astype(np.float32) / 255.0

            if gt_boxes_final_pixels:
                gt_boxes_norm = np.array(gt_boxes_final_pixels, dtype=np.float32) / np.array(
                    [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]])
                gt_class_ids = np.array([self.class_mapping[name] for name in gt_class_names_final], dtype=np.int32)
            else:
                gt_boxes_norm = np.zeros((0, 4), dtype=np.float32)
                gt_class_ids = np.zeros((0,), dtype=np.int32)

            anchor_labels, matched_gt_boxes, max_iou_per_anchor = assign_gt_to_anchors(
                gt_boxes_norm, self.all_anchors, self.pos_iou_thresh, self.neg_iou_thresh
            )

            y_true_cls = np.zeros((len(self.all_anchors), self.num_classes), dtype=np.float32)
            y_true_reg = np.zeros((len(self.all_anchors), 4), dtype=np.float32)

            positive_indices = np.where(anchor_labels == 1)[0]
            if len(positive_indices) > 0:
                iou_pos_anchors_vs_gt = calculate_iou_matrix(self.all_anchors[positive_indices], gt_boxes_norm)
                best_gt_indices = np.argmax(iou_pos_anchors_vs_gt, axis=1)
                positive_gt_class_ids = gt_class_ids[best_gt_indices]
                y_true_cls[positive_indices] = tf.keras.utils.to_categorical(positive_gt_class_ids,
                                                                             num_classes=self.num_classes)
                y_true_reg[positive_indices] = encode_box_targets(self.all_anchors[positive_indices],
                                                                  matched_gt_boxes[positive_indices])

            ignore_indices = np.where(anchor_labels == 0)[0]
            y_true_cls[ignore_indices] = -1.0

            if not self.debug_mode:
                yield image_final_norm, (y_true_reg, y_true_cls)
            else:
                debug_info = {
                    "image_path": str(image_path),
                    "image_original": image_original_rgb,
                    # ===> ИЗМЕНЕНИЕ ЗДЕСЬ <===
                    "gt_boxes_original": np.array(gt_boxes_pixels, dtype=np.int32).reshape(-1, 4),
                    "gt_class_names_original": np.array(gt_class_names, dtype=str),
                    "image_augmented": image_final,
                    "gt_boxes_augmented": np.array(gt_boxes_final_pixels, dtype=np.float32).reshape(-1, 4),
                    "gt_class_names_augmented": np.array(gt_class_names_final, dtype=str),
                    "all_anchors_norm": self.all_anchors,
                    "anchor_labels": anchor_labels,
                    "max_iou_per_anchor": max_iou_per_anchor,
                }
                yield image_final_norm, (y_true_reg, y_true_cls), debug_info


def create_dataset(config, is_training=True, batch_size=8, debug_mode=False):
    all_anchors = generate_all_anchors(
        config['input_shape'], [8, 16, 32], config['anchor_scales'], config['anchor_ratios']
    )
    generator = DataGenerator(config, all_anchors, is_training, debug_mode=debug_mode)

    if not debug_mode:
        output_signature = (
            tf.TensorSpec(shape=config['input_shape'], dtype=tf.float32),
            (tf.TensorSpec(shape=(len(all_anchors), 4), dtype=tf.float32),
             tf.TensorSpec(shape=(len(all_anchors), config['num_classes']), dtype=tf.float32))
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=config['input_shape'], dtype=tf.float32),
            (tf.TensorSpec(shape=(len(all_anchors), 4), dtype=tf.float32),
             tf.TensorSpec(shape=(len(all_anchors), config['num_classes']), dtype=tf.float32)),
            {
                "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
                "image_original": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                "gt_boxes_original": tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
                "gt_class_names_original": tf.TensorSpec(shape=(None,), dtype=tf.string),
                "image_augmented": tf.TensorSpec(shape=config['input_shape'], dtype=tf.uint8),
                "gt_boxes_augmented": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "gt_class_names_augmented": tf.TensorSpec(shape=(None,), dtype=tf.string),
                "all_anchors_norm": tf.TensorSpec(shape=(len(all_anchors), 4), dtype=tf.float32),
                "anchor_labels": tf.TensorSpec(shape=(len(all_anchors),), dtype=tf.int32),
                "max_iou_per_anchor": tf.TensorSpec(shape=(len(all_anchors),), dtype=tf.float32),
            }
        )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if not debug_mode:
        if is_training:
            dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Тестовый запуск ---
if __name__ == '__main__':

    print("--- Финальное тестирование data_loader_v3_standard.py ---")
    from pathlib import Path
    import yaml

    try:
        config_path = Path(__file__).parent.parent / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Конфигурационный файл успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
        exit()
    print("\nСоздание тренировочного датасета (1 батч)...")
    train_dataset = create_dataset(config, is_training=True, batch_size=4)  # Уменьшил batch_size
    try:
        for images, (y_reg, y_cls) in train_dataset.take(1):
            print("\n--- Проверка форм одного батча ---")
            print(f"Форма батча изображений: {images.shape}")
            print(f"Форма y_true для регрессии: {y_reg.shape}")
            print(f"Форма y_true для классификации: {y_cls.shape}")

            num_positives = tf.reduce_sum(tf.cast(tf.reduce_any(y_cls > 0, axis=-1), tf.int32)).numpy()
            num_ignored = tf.reduce_sum(tf.cast(tf.reduce_all(y_cls == -1, axis=-1), tf.int32)).numpy()
            print(f"\nВ первом батче найдено:")
            print(f"  - Позитивных якорей: {num_positives}")
            print(f"  - Игнорируемых якорей: {num_ignored}")

            assert images.shape[0] == 4
            assert y_reg.shape[0] == 4
            assert y_cls.shape[0] == 4
        print("\n[SUCCESS] Базовый тест create_dataset пройден. Датасет генерирует батчи правильной формы.")
    except Exception as e:
        print(f"\n[ERROR] Ошибка при итерации по датасету: {e}")
        import traceback

        traceback.print_exc()