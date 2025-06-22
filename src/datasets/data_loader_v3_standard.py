import logging

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from . import augmentations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
def assign_gt_to_anchors(gt_boxes_norm, gt_class_ids, all_anchors_norm, pos_iou_thresh, neg_iou_thresh):
    """
    Назначает GT-боксы якорям согласно правилам RetinaNet.
    Теперь также возвращает ID класса сопоставленного GT для каждого якоря.

    Args:
        gt_boxes_norm (np.ndarray): Массив GT боксов (M, 4), нормализованных [xmin,ymin,xmax,ymax].
        gt_class_ids (np.ndarray): Массив ID классов для GT боксов (M,).
        all_anchors_norm (np.ndarray): Массив всех якорей (N, 4), нормализованных [xmin,ymin,xmax,ymax].
        pos_iou_thresh (float): Порог IoU для позитивного назначения.
        neg_iou_thresh (float): Порог IoU для негативного назначения.

    Returns:
        tuple: (anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor)
               - anchor_labels (np.ndarray): (N,) массив с метками (1:pos, 0:ign, -1:neg).
               - matched_gt_boxes_for_anchors (np.ndarray): (N, 4) GT-бокс, сопоставленный с каждым якорем.
               - matched_gt_class_ids_for_anchors (np.ndarray): (N,) ID класса GT, сопоставленного с каждым якорем.
               - max_iou_per_anchor (np.ndarray): (N,) максимальное IoU для каждого якоря.
    """
    num_anchors = all_anchors_norm.shape[0]
    num_gt = gt_boxes_norm.shape[0]

    # Инициализируем массивы для возврата
    anchor_labels = np.full((num_anchors,), -1, dtype=np.int32)  # -1: negative по умолчанию
    # Для не-позитивных якорей эти значения не будут использоваться в loss, но должны быть
    matched_gt_boxes_for_anchors = np.zeros((num_anchors, 4), dtype=np.float32)
    matched_gt_class_ids_for_anchors = np.zeros((num_anchors,),
                                                dtype=np.int32)  # Заполним нулями (например, первым классом)
    max_iou_per_anchor = np.zeros((num_anchors,), dtype=np.float32)

    if num_gt == 0:  # Если нет GT объектов, все якоря негативные
        return anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor

    # 1. Вычисляем матрицу IoU (num_gt, num_anchors)
    iou_matrix = calculate_iou_matrix(gt_boxes_norm, all_anchors_norm)  # Важно: первый аргумент GT, второй - якоря

    # 2. Находим максимальное IoU для каждого якоря и индекс GT, с которым оно достигнуто.
    # axis=0 означает, что мы ищем максимум по столбцам (т.е. для каждого якоря ищем лучший GT)
    max_iou_per_anchor[:] = np.max(iou_matrix, axis=0)  # (num_anchors,)
    matched_gt_idx_per_anchor = np.argmax(iou_matrix, axis=0)  # (num_anchors,) Индекс лучшего GT для каждого якоря

    # 3. Шаг 1 назначения: по порогу IoU
    anchor_labels[max_iou_per_anchor >= pos_iou_thresh] = 1  # 1: positive
    anchor_labels[(max_iou_per_anchor >= neg_iou_thresh) & (max_iou_per_anchor < pos_iou_thresh)] = 0  # 0: ignored

    # 4. Шаг 2 назначения: "Гарантированное назначение" для каждого GT
    # Для каждого GT бокса находим якорь с лучшим IoU
    # axis=1 означает, что мы ищем максимум по строкам (т.е. для каждого GT ищем лучший якорь)
    best_anchor_idx_per_gt = np.argmax(iou_matrix, axis=1)  # (num_gt,) Индекс лучшего якоря для каждого GT

    # Принудительно делаем эти якоря позитивными
    anchor_labels[best_anchor_idx_per_gt] = 1

    # ===> КЛЮЧЕВОЕ ИЗМЕНЕНИЕ GPT (адаптированное): Обновляем сопоставление для этих "гарантированных" якорей <===
    # Если якорь был сделан позитивным на этом шаге, убедимся, что он сопоставлен с ПРАВИЛЬНЫМ GT.
    # matched_gt_idx_per_anchor УЖЕ содержит лучший GT для каждого якоря по IoU,
    # но для "гарантированных" якорей мы должны использовать тот GT, который их "выбрал".
    for gt_idx_loop in range(num_gt):
        anchor_to_update_idx = best_anchor_idx_per_gt[gt_idx_loop]
        # Обновляем, какой GT сопоставлен с этим "гарантированно позитивным" якорем
        matched_gt_idx_per_anchor[anchor_to_update_idx] = gt_idx_loop
        # Также обновим max_iou_per_anchor для этого якоря, чтобы он отражал IoU с "его" GT
        max_iou_per_anchor[anchor_to_update_idx] = iou_matrix[gt_idx_loop, anchor_to_update_idx]

    # 5. Собираем matched_gt_boxes и matched_gt_class_ids для ВСЕХ якорей
    # на основе обновленного matched_gt_idx_per_anchor.
    # Для негативных и игнорируемых якорей здесь будут какие-то GT и классы,
    # но мы их не будем использовать в loss регрессии или для positive_gt_class_ids.
    matched_gt_boxes_for_anchors = gt_boxes_norm[matched_gt_idx_per_anchor]
    matched_gt_class_ids_for_anchors = gt_class_ids[matched_gt_idx_per_anchor]

    return anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor


def encode_box_targets(anchors, matched_gt_boxes):  # anchors - это all_anchors_np[positive_indices]
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
    gt_cx = matched_gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = matched_gt_boxes[:, 1] + 0.5 * gt_h

    epsilon = 1e-7
    anchor_w = np.maximum(anchor_w, epsilon)
    anchor_h = np.maximum(anchor_h, epsilon)
    gt_w = np.maximum(gt_w, epsilon)
    gt_h = np.maximum(gt_h, epsilon)

    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)

    # Клиппинг значений (очень важно)
    bbox_xform_clip = np.log(1000. / 16.)  # ~4.135
    xy_clip = 2.5  # Ограничим смещение центра (можно подбирать)

    tx = np.clip(tx, -xy_clip, xy_clip)
    ty = np.clip(ty, -xy_clip, xy_clip)
    tw = np.clip(tw, -bbox_xform_clip, bbox_xform_clip)
    th = np.clip(th, -bbox_xform_clip, bbox_xform_clip)

    targets = np.stack([tx, ty, tw, th], axis=1)
    return targets


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

        self.image_paths = sorted([p for p in self.image_dir.glob('*.jpg')])  # Добавил .jpg
        self.annot_paths = sorted([p for p in self.annot_dir.glob('*.xml')])

        # Проверка на соответствие количества изображений и аннотаций
        if len(self.image_paths) != len(self.annot_paths) or not self.image_paths:
            logger.warning(f"Несоответствие файлов или пустые папки в {self.image_dir} / {self.annot_dir}")
            # Можно здесь выбросить исключение или обработать как-то иначе
            # Для теста оставим возможность работы с пустым списком, __len__ вернет 0
            self.image_paths = []
            self.annot_paths = []

        self.input_shape = config['input_shape']
        self.class_mapping = {name: i for i, name in enumerate(config['class_names'])}
        self.num_classes = config['num_classes']
        self.pos_iou_thresh = config['anchor_positive_iou_threshold']
        self.neg_iou_thresh = config['anchor_ignore_iou_threshold']  # <--- Убедимся, что он здесь есть

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
            if image is None:  # Проверка на случай, если изображение не загрузилось
                logger.error(f"Не удалось загрузить изображение: {image_path}")
                continue
            image_original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h_orig, image_w_orig, _ = image_original_rgb.shape

            gt_boxes_pixels, gt_class_names_original = parse_voc_xml(annot_path)

            image_resized = cv2.resize(image_original_rgb, (self.input_shape[1], self.input_shape[0]))

            gt_boxes_resized_pixels = []
            if gt_boxes_pixels:
                for box in gt_boxes_pixels:
                    xmin, ymin, xmax, ymax = box
                    xmin_r = (xmin / image_w_orig) * self.input_shape[1]
                    ymin_r = (ymin / image_h_orig) * self.input_shape[0]
                    xmax_r = (xmax / image_w_orig) * self.input_shape[1]
                    ymax_r = (ymax / image_h_orig) * self.input_shape[0]
                    gt_boxes_resized_pixels.append([xmin_r, ymin_r, xmax_r, ymax_r])

            # Данные для аугментации
            image_to_augment = image_resized
            boxes_to_augment = gt_boxes_resized_pixels
            class_labels_for_aug = gt_class_names_original  # Используем имена классов

            if self.augmenter:
                augmented = self.augmenter(image=image_to_augment, bboxes=boxes_to_augment,
                                           class_labels_for_albumentations=class_labels_for_aug)
                image_final_uint8 = augmented['image']
                gt_boxes_final_pixels = augmented['bboxes']
                gt_class_names_final = augmented['class_labels_for_albumentations']
            else:
                image_final_uint8 = image_to_augment
                gt_boxes_final_pixels = boxes_to_augment
                gt_class_names_final = class_labels_for_aug

            image_final_norm = image_final_uint8.astype(np.float32) / 255.0

            gt_boxes_norm = np.array(gt_boxes_final_pixels, dtype=np.float32) / np.array(
                [self.input_shape[1], self.input_shape[0], self.input_shape[1],
                 self.input_shape[0]]) if gt_boxes_final_pixels else np.empty((0, 4), dtype=np.float32)
            gt_class_ids = np.array([self.class_mapping[name] for name in gt_class_names_final],
                                    dtype=np.int32) if gt_class_names_final else np.empty((0,), dtype=np.int32)

            # ===> ИСПРАВЛЕННЫЙ ВЫЗОВ assign_gt_to_anchors <===
            anchor_labels, matched_gt_boxes, matched_gt_class_ids, max_iou_per_anchor = assign_gt_to_anchors(
                gt_boxes_norm,
                gt_class_ids,  # Передаем ID классов GT
                self.all_anchors,
                self.pos_iou_thresh,
                self.neg_iou_thresh  # <--- Добавлен недостающий аргумент
            )

            y_true_cls = np.zeros((len(self.all_anchors), self.num_classes), dtype=np.float32)
            y_true_reg = np.zeros((len(self.all_anchors), 4), dtype=np.float32)

            positive_indices = np.where(anchor_labels == 1)[0]
            if len(positive_indices) > 0:
                # Используем ID классов, сопоставленные на этапе ASSIGN_GT
                positive_actual_gt_class_ids = matched_gt_class_ids[positive_indices]
                y_true_cls[positive_indices] = tf.keras.utils.to_categorical(positive_actual_gt_class_ids,
                                                                             num_classes=self.num_classes)

                # Для регрессии используем те GT боксы, которые были сопоставлены
                y_true_reg[positive_indices] = encode_box_targets(
                    self.all_anchors[positive_indices],
                    matched_gt_boxes[positive_indices]  # Передаем правильные matched_gt_boxes
                )

            ignore_indices = np.where(anchor_labels == 0)[0]
            y_true_cls[ignore_indices] = -1.0

            if not self.debug_mode:
                yield image_final_norm, (y_true_reg, y_true_cls)
            else:
                debug_info = {
                    "image_path": str(image_path),
                    "image_original": image_original_rgb,  # uint8
                    "gt_boxes_original": np.array(gt_boxes_pixels, dtype=np.int32).reshape(-1, 4),
                    "gt_class_names_original": np.array(gt_class_names_original, dtype=str),
                    "image_augmented": image_final_uint8,  # uint8
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
    print("--- Финальное тестирование data_loader_v3_standard.py (с исправленным assign_gt) ---")

    from pathlib import Path
    import yaml

    try:
        # Загружаем основной конфиг
        _project_root = Path(__file__).parent.parent.parent.resolve()  # Корень проекта
        config_path = _project_root / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Конфигурационный файл успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
        exit()

    print("\n1. Тест создания тренировочного датасета (1 батч, обычный режим):")
    try:
        train_dataset = create_dataset(config, is_training=True, batch_size=2, debug_mode=False)
        for images, (y_reg, y_cls) in train_dataset.take(1):
            print(f"  - Форма батча изображений: {images.shape}")
            print(f"  - Форма y_true для регрессии: {y_reg.shape}")
            print(f"  - Форма y_true для классификации: {y_cls.shape}")
            assert images.shape[0] == 2
            num_total_anchors = generate_all_anchors(config['input_shape'], [8, 16, 32], config['anchor_scales'],
                                                     config['anchor_ratios']).shape[0]
            assert y_reg.shape[1] == num_total_anchors
            assert y_cls.shape[1] == num_total_anchors
        print("  - [SUCCESS] Тест 1 пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 1: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Тест создания валидационного датасета (1 батч, debug_mode=True):")
    try:
        val_dataset_debug = create_dataset(config, is_training=False, batch_size=1, debug_mode=True)
        for images_dbg, (y_reg_dbg, y_cls_dbg), debug_info_dict in val_dataset_debug.take(1):
            print(f"  - Форма изображения (debug): {images_dbg.shape}")
            print(f"  - Форма y_reg (debug): {y_reg_dbg.shape}")
            print(f"  - Форма y_cls (debug): {y_cls_dbg.shape}")
            print(f"  - Ключи в debug_info: {list(debug_info_dict.keys())}")
            assert images_dbg.shape[0] == 1  # batch_size = 1
            assert "image_path" in debug_info_dict
            assert "anchor_labels" in debug_info_dict
        print("  - [SUCCESS] Тест 2 пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 2: {e}")
        import traceback

        traceback.print_exc()