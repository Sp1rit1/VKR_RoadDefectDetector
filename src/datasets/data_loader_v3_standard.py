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
    """
    Вычисляет матрицу IoU (Intersection over Union) между двумя наборами боксов.
    Использует векторизацию numpy для высокой производительности.

    Args:
        boxes1 (np.ndarray): Массив боксов формы (N, 4) в формате [xmin, ymin, xmax, ymax].
        boxes2 (np.ndarray): Массив боксов формы (M, 4) в формате [xmin, ymin, xmax, ymax].

    Returns:
        np.ndarray: Матрица IoU формы (N, M), где каждый элемент (i, j) - это
                    IoU между i-м боксом из boxes1 и j-м боксом из boxes2.
    """
    # Преобразуем в numpy массивы для надежности
    boxes1_np = np.array(boxes1)
    boxes2_np = np.array(boxes2)

    # Расширяем размерности, чтобы каждый бокс из boxes1 сравнивался с каждым из boxes2
    # boxes1_ext: (N, 1, 4)
    # boxes2_ext: (1, M, 4)
    boxes1_ext = boxes1_np[:, np.newaxis, :]
    boxes2_ext = boxes2_np[np.newaxis, :, :]

    # --- Вычисление координат пересечения (intersection) ---
    # Находим левую верхнюю точку (x1, y1) прямоугольника пересечения
    # Это максимум из левых верхних точек исходных боксов
    inter_x1 = np.maximum(boxes1_ext[..., 0], boxes2_ext[..., 0])
    inter_y1 = np.maximum(boxes1_ext[..., 1], boxes2_ext[..., 1])

    # Находим правую нижнюю точку (x2, y2) прямоугольника пересечения
    # Это минимум из правых нижних точек исходных боксов
    inter_x2 = np.minimum(boxes1_ext[..., 2], boxes2_ext[..., 2])
    inter_y2 = np.minimum(boxes1_ext[..., 3], boxes2_ext[..., 3]) # <--- ЗДЕСЬ БЫЛА ОШИБКА, ИСПРАВЛЕНО

    # Вычисляем ширину и высоту пересечения
    # np.maximum(0.0, ...) гарантирует, что если боксы не пересекаются, размер будет 0, а не отрицательным
    intersection_w = np.maximum(0.0, inter_x2 - inter_x1)
    intersection_h = np.maximum(0.0, inter_y2 - inter_y1)

    # Вычисляем площадь пересечения
    intersection_area = intersection_w * intersection_h

    # --- Вычисление площади объединения (union) ---
    # Площадь каждого бокса из первого набора
    area1 = (boxes1_ext[..., 2] - boxes1_ext[..., 0]) * (boxes1_ext[..., 3] - boxes1_ext[..., 1])
    # Площадь каждого бокса из второго набора
    area2 = (boxes2_ext[..., 2] - boxes2_ext[..., 0]) * (boxes2_ext[..., 3] - boxes2_ext[..., 1])

    # Площадь объединения = Площадь1 + Площадь2 - ПлощадьПересечения
    union_area = area1 + area2 - intersection_area

    # --- Финальный расчет IoU ---
    # Добавляем маленький эпсилон в знаменатель, чтобы избежать деления на ноль
    iou = intersection_area / (union_area + 1e-7)

    return iou


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

        # --- Настройка путей к данным ---
        dataset_path = Path(config['dataset_path'])
        img_subdir = config['train_images_subdir'] if is_training else config['val_images_subdir']
        ann_subdir = config['train_annotations_subdir'] if is_training else config['val_annotations_subdir']
        self.image_dir = dataset_path / img_subdir
        self.annot_dir = dataset_path / ann_subdir

        # --- Сопоставление файлов ---
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_image_files = [p for ext in image_extensions for p in self.image_dir.glob(ext)]
        all_annot_files = list(self.annot_dir.glob('*.xml'))
        image_file_dict = {p.stem: p for p in all_image_files}
        annot_file_dict = {p.stem: p for p in all_annot_files}
        matched_basenames = sorted(list(image_file_dict.keys() & annot_file_dict.keys()))
        self.image_paths = [image_file_dict[basename] for basename in matched_basenames]
        self.annot_paths = [annot_file_dict[basename] for basename in matched_basenames]

        if not self.image_paths:
            logging.error(f"Не найдено совпадающих пар изображений/аннотаций в {self.image_dir} и {self.annot_dir}")

        # --- Загрузка параметров из конфига ---
        self.input_shape = config['input_shape']
        self.class_mapping = {name: i for i, name in enumerate(config['class_names'])}
        self.num_classes = config['num_classes']
        self.pos_iou_thresh = config['anchor_positive_iou_threshold']
        self.neg_iou_thresh = config['anchor_ignore_iou_threshold']
        self.all_anchors = all_anchors

        # --- [ВАЖНО] Предрасчет размеров для разбиения y_true ---
        self.anchor_counts_per_level = []
        self.output_shapes_per_level = []
        H, W = self.input_shape[:2]
        num_base_anchors = config['num_anchors_per_level']  # Одно число, например 21
        self.fpn_strides = [8, 16, 32]
        for stride in self.fpn_strides:
            fh, fw = H // stride, W // stride
            self.anchor_counts_per_level.append(fh * fw * num_base_anchors)
            self.output_shapes_per_level.append((fh, fw, num_base_anchors))

        # --- Настройка аугментации ---
        if self.is_training and config.get('use_augmentation', True):
            self.augmenter = augmentations.get_detector_train_augmentations(*self.input_shape[:2])
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.image_paths)

    def __call__(self):
        for i in range(self.__len__()):
            image_path, annot_path = self.image_paths[i], self.annot_paths[i]

            # 1. Загрузка и ресайз
            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning(f"Не удалось прочитать изображение: {image_path}, пропускаем.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h_orig, w_orig, _ = image_rgb.shape
            gt_boxes_pixels, gt_class_names = parse_voc_xml(annot_path)

            h_target, w_target = self.input_shape[:2]
            image_resized = cv2.resize(image_rgb, (w_target, h_target))

            gt_boxes_resized_pixels = []
            if gt_boxes_pixels:
                for b in gt_boxes_pixels:
                    x1, y1, x2, y2 = b
                    gt_boxes_resized_pixels.append(
                        [(x1 / w_orig) * w_target, (y1 / h_orig) * h_target, (x2 / w_orig) * w_target,
                         (y2 / h_orig) * h_target])

            # 2. Аугментация
            if self.augmenter:
                augmented = self.augmenter(image=image_resized, bboxes=gt_boxes_resized_pixels,
                                           class_labels_for_albumentations=gt_class_names)
                image_final = augmented['image']
                gt_boxes_final_pixels = augmented['bboxes']
                gt_class_names_final = augmented['class_labels_for_albumentations']
            else:
                image_final = image_resized
                gt_boxes_final_pixels = gt_boxes_resized_pixels
                gt_class_names_final = gt_class_names

            # 3. Нормализация данных для y_true
            image_for_model = image_final.astype(np.float32) / 255.0

            gt_boxes_norm = np.array(gt_boxes_final_pixels, dtype=np.float32) / np.array(
                [w_target, h_target, w_target, h_target]) if gt_boxes_final_pixels else np.empty((0, 4))
            gt_class_ids = np.array([self.class_mapping[name] for name in gt_class_names_final],
                                    dtype=np.int32) if gt_class_names_final else np.empty((0,), dtype=np.int32)

            # 4. Назначение якорей и создание "плоских" y_true
            anchor_labels, matched_gt_boxes, matched_gt_class_ids, _ = assign_gt_to_anchors(gt_boxes_norm, gt_class_ids,
                                                                                            self.all_anchors,
                                                                                            self.pos_iou_thresh,
                                                                                            self.neg_iou_thresh)

            y_true_cls_flat = np.zeros((self.all_anchors.shape[0], self.num_classes), dtype=np.float32)
            y_true_reg_flat = np.zeros((self.all_anchors.shape[0], 4), dtype=np.float32)

            positive_indices = np.where(anchor_labels == 1)[0]
            if len(positive_indices) > 0:
                y_true_cls_flat[positive_indices] = tf.keras.utils.to_categorical(
                    matched_gt_class_ids[positive_indices], num_classes=self.num_classes)
                y_true_reg_flat[positive_indices] = encode_box_targets(self.all_anchors[positive_indices],
                                                                       matched_gt_boxes[positive_indices])

            y_true_cls_flat[np.where(anchor_labels == 0)[0]] = -1.0  # Метка для игнорируемых

            # 5. Разбиение плоских y_true на 6 частей, как у модели
            y_reg_split_by_level = tf.split(y_true_reg_flat, self.anchor_counts_per_level, axis=0)
            y_cls_split_by_level = tf.split(y_true_cls_flat, self.anchor_counts_per_level, axis=0)

            # Собираем в список для удобства
            y_true_final_list = []
            # Сначала регрессия
            for i in range(len(self.fpn_strides)):
                shape = (*self.output_shapes_per_level[i], 4)
                y_true_final_list.append(tf.reshape(y_reg_split_by_level[i], shape))
            # Затем классификация
            for i in range(len(self.fpn_strides)):
                shape = (*self.output_shapes_per_level[i], self.num_classes)
                y_true_final_list.append(tf.reshape(y_cls_split_by_level[i], shape))

            # [ИСПРАВЛЕНИЕ] Превращаем список в кортеж перед yield
            y_true_final_tuple = tuple(y_true_final_list)

            # Возвращаем данные в формате (изображение, КОРТЕЖ_из_6_тензоров)
            if not self.debug_mode:
                yield image_for_model, y_true_final_tuple
            else:
                # В debug режиме возвращаем то же самое, но без словаря, чтобы не усложнять
                # и чтобы сигнатура совпадала с основным режимом
                yield image_for_model, y_true_final_tuple


def create_dataset(config, is_training=True, batch_size=8, debug_mode=False):
    """
    Создает tf.data.Dataset для обучения или валидации.
    """
    # Генерируем якоря один раз
    all_anchors = generate_all_anchors(
        config['input_shape'], [8, 16, 32], config['anchor_scales'], config['anchor_ratios']
    )
    # Создаем экземпляр генератора
    generator = DataGenerator(config, all_anchors, is_training, debug_mode=debug_mode)

    # Создаем правильную output_signature
    fh_p3, fw_p3, na = generator.output_shapes_per_level[0]
    fh_p4, fw_p4, _ = generator.output_shapes_per_level[1]
    fh_p5, fw_p5, _ = generator.output_shapes_per_level[2]

    output_signature = (
        tf.TensorSpec(shape=config['input_shape'], dtype=tf.float32),
        # Кортеж из 6 тензоров
        (
            # Регрессия
            tf.TensorSpec(shape=(fh_p3, fw_p3, na, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(fh_p4, fw_p4, na, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(fh_p5, fw_p5, na, 4), dtype=tf.float32),
            # Классификация
            tf.TensorSpec(shape=(fh_p3, fw_p3, na, config['num_classes']), dtype=tf.float32),
            tf.TensorSpec(shape=(fh_p4, fw_p4, na, config['num_classes']), dtype=tf.float32),
            tf.TensorSpec(shape=(fh_p5, fw_p5, na, config['num_classes']), dtype=tf.float32),
        )
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # [ИЗМЕНЕНИЕ] Добавляем .repeat()
    if not debug_mode:
        # Зацикливаем датасет, чтобы он никогда не заканчивался
        dataset = dataset.repeat()  # <--- ГЛАВНОЕ ИЗМЕНЕНИЕ

        if is_training:
            # Перемешиваем только тренировочный датасет
            buffer_size = min(500, len(generator))
            if buffer_size > 0:
                dataset = dataset.shuffle(buffer_size=buffer_size)

        # Батчим после перемешивания и зацикливания
        dataset = dataset.batch(batch_size, drop_remainder=is_training)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Тестовый запуск ---
if __name__ == '__main__':
    print("--- Финальное тестирование data_loader_v3_standard.py ---")

    from pathlib import Path
    import yaml

    try:
        # Загружаем основной конфиг
        _project_root = Path(__file__).parent.parent.parent.resolve()
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

        # [ИСПРАВЛЕНИЕ] Распаковываем y_true как один объект (кортеж)
        for images, y_true_tuple in train_dataset.take(1):
            print(f"  - Форма батча изображений: {images.shape}")
            print(f"  - Тип y_true: {type(y_true_tuple)}")
            print(f"  - Длина y_true (ожидаем 6): {len(y_true_tuple)}")

            # Проверяем формы каждого из 6 тензоров в y_true
            y_reg_p3, y_reg_p4, y_reg_p5, y_cls_p3, y_cls_p4, y_cls_p5 = y_true_tuple
            print(f"  - Форма y_true_reg_p3: {y_reg_p3.shape}")
            print(f"  - Форма y_true_cls_p3: {y_cls_p3.shape}")

            assert images.shape[0] == 2
            assert len(y_true_tuple) == 6
            # Простая проверка формы первого регрессионного тензора
            assert len(y_reg_p3.shape) == 5  # (batch, H, W, num_anchors, 4)

        print("  - [SUCCESS] Тест 1 пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 1: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Тест создания валидационного датасета (1 батч, debug_mode=True):")
    try:
        # [ИСПРАВЛЕНИЕ] В debug_mode теперь тоже 2 элемента
        val_dataset_debug = create_dataset(config, is_training=False, batch_size=1, debug_mode=True)

        # [ИСПРАВЛЕНИЕ] Распаковываем только 2 элемента
        for image_dbg, y_true_tuple_dbg in val_dataset_debug.take(1):
            print(f"  - Форма изображения (debug): {image_dbg.shape}")
            print(f"  - Длина y_true (debug): {len(y_true_tuple_dbg)}")

            assert image_dbg.shape == tuple(config['input_shape'])
            assert len(y_true_tuple_dbg) == 6

        print("  - [SUCCESS] Тест 2 пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 2: {e}")
        import traceback

        traceback.print_exc()