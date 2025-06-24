# RoadDefectDetector/src/datasets/data_loader_v3_standard.py

import logging

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from . import augmentations # Убедитесь, что augmentations.py находится в той же папке или доступен

logger = logging.getLogger(__name__)
# Установим уровень только если он не был установлен ранее (для избежания дублирования)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Блок 1: Функции-утилиты ---
# (Эти функции не меняются, так как их логика корректна)

def parse_voc_xml(xml_path: Path):
    """Парсит XML-файл аннотации в формате PASCAL VOC."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, class_names = [], []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # Убедимся, что класс входит в наш список классов, если нужно фильтровать
        # if class_name not in config['class_names']: continue
        bndbox = obj.find('bndbox')
        # Используем float для промежуточных вычислений, int для финального результата
        xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
        xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax]) # Возвращаем float здесь, чтобы избежать ошибок точности
        class_names.append(class_name)
    return boxes, class_names


def generate_all_anchors(input_shape, fpn_strides, anchor_scales, anchor_ratios):
    """
    Генерирует все якоря для всех уровней FPN на основе Scale/Ratio подхода (RetinaNet-like).
    Якоря возвращаются в нормализованных координатах [xmin, ymin, xmax, ymax].

    Args:
        input_shape (list): Форма входного изображения [высота, ширина, каналы].
        fpn_strides (list): Список шагов (strides) для каждого уровня FPN (например, [8, 16, 32]).
        anchor_scales (list): Список скалярных множителей масштаба якорей (например, [1.0, 1.26, 1.59]).
        anchor_ratios (list): Список соотношений сторон (ширина/высота) якорей (например, [0.5, 1.0, 2.0]).

    Returns:
        np.ndarray: Массив всех сгенерированных якорей формы (TotalAnchors, 4),
                    в нормализованных координатах [xmin, ymin, xmax, ymax].
    """
    all_anchors = []
    image_height, image_width = input_shape[0], input_shape[1]
    # Базовый размер якоря на уровне 1 (stride 1)
    # В RetinaNet часто base_anchor_size=32 для P3 (stride 8), что эквивалентно stride*4.
    # Придерживаемся stride*4 для каждого уровня.
    anchor_base_size_per_stride = {stride: stride * 4 for stride in fpn_strides}

    for stride in fpn_strides:
        base_anchor_size = anchor_base_size_per_stride[stride] # Размер базового якоря для текущего страйда
        feature_map_height, feature_map_width = image_height // stride, image_width // stride

        # Генерируем якоря для каждой ячейки сетки на текущем уровне
        for y_fm in range(feature_map_height):
            for x_fm in range(feature_map_width):
                # Центр текущей ячейки в пикселях оригинального изображения
                cx, cy = (x_fm + 0.5) * stride, (y_fm + 0.5) * stride

                # Генерируем якоря для этой ячейки, комбинируя scales и ratios
                for scale in anchor_scales:
                    for ratio in anchor_ratios:
                        # Вычисляем ширину и высоту якоря в пикселях
                        # W/H = ratio => W = ratio * H
                        # W * H = base_anchor_size^2 * scale^2
                        # (ratio * H) * H = base_anchor_size^2 * scale^2
                        # H^2 = (base_anchor_size^2 * scale^2) / ratio
                        # H = base_anchor_size * scale / sqrt(ratio)
                        # W = base_anchor_size * scale * sqrt(ratio)
                        h = base_anchor_size * scale / np.sqrt(ratio)
                        w = base_anchor_size * scale * np.sqrt(ratio)


                        # Вычисляем координаты углов якоря в пикселях
                        xmin, ymin = cx - w / 2., cy - h / 2.
                        xmax, ymax = cx + w / 2., cy + h / 2.

                        all_anchors.append([xmin, ymin, xmax, ymax])

    # Преобразуем список в numpy массив и нормализуем координаты в диапазон [0, 1]
    all_anchors_np = np.array(all_anchors, dtype=np.float32)
    all_anchors_np[:, [0, 2]] /= image_width
    all_anchors_np[:, [1, 3]] /= image_height

    # Обрезаем координаты до диапазона [0, 1] на всякий случай
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
    inter_x1 = np.maximum(boxes1_ext[..., 0], boxes2_ext[..., 0])
    inter_y1 = np.maximum(boxes1_ext[..., 1], boxes2_ext[..., 1])
    inter_x2 = np.minimum(boxes1_ext[..., 2], boxes2_ext[..., 2])
    inter_y2 = np.minimum(boxes1_ext[..., 3], boxes2_ext[..., 3])

    # Вычисляем ширину и высоту пересечения
    intersection_w = np.maximum(0.0, inter_x2 - inter_x1)
    intersection_h = np.maximum(0.0, inter_y2 - inter_y1)

    # Вычисляем площадь пересечения
    intersection_area = intersection_w * intersection_h

    # --- Вычисление площади объединения (union) ---
    area1 = (boxes1_ext[..., 2] - boxes1_ext[..., 0]) * (boxes1_ext[..., 3] - boxes1_ext[..., 1])
    area2 = (boxes2_ext[..., 2] - boxes2_ext[..., 0]) * (boxes2_ext[..., 3] - boxes2_ext[..., 1])

    # Площадь объединения = Площадь1 + Площадь2 - ПлощадьПересечения
    union_area = area1 + area2 - intersection_area

    # --- Финальный расчет IoU ---
    # Добавляем маленький эпсилон в знаменатель, чтобы избежать деления на ноль
    iou = intersection_area / (union_area + 1e-7)

    return iou


def assign_gt_to_anchors(gt_boxes_norm, gt_class_ids, all_anchors_norm, pos_iou_thresh, neg_iou_thresh):
    """
    Назначает GT-боксы якорям согласно правилам RetinaNet.
    Возвращает метки якорей (позитивный/негативный/игнорируемый),
    а также сопоставленные GT-боксы и ID классов для позитивных якорей.

    Args:
        gt_boxes_norm (np.ndarray): Массив GT боксов (M, 4), нормализованных [xmin,ymin,xmax,ymax].
        gt_class_ids (np.ndarray): Массив ID классов для GT боксов (M,).
        all_anchors_norm (np.ndarray): Массив всех якорей (N, 4), нормализованных [xmin,ymin,xmax,ymax].
        pos_iou_thresh (float): Порог IoU для позитивного назначения (>=).
        neg_iou_thresh (float): Порог IoU для негативного назначения (<).
                                 Якоря с IoU в диапазоне [neg_iou_thresh, pos_iou_thresh) игнорируются.

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
    # -1: negative по умолчанию
    anchor_labels = np.full((num_anchors,), -1, dtype=np.int32)
    # Заполним нулями (например, первым классом) и нулями для боксов.
    # Для не-позитивных якорей эти значения не будут использоваться в loss, но должны быть.
    matched_gt_boxes_for_anchors = np.zeros((num_anchors, 4), dtype=np.float32)
    matched_gt_class_ids_for_anchors = np.zeros((num_anchors,), dtype=np.int32)
    max_iou_per_anchor = np.zeros((num_anchors,), dtype=np.float32)

    if num_gt == 0:  # Если нет GT объектов, все якоря негативные
        # anchor_labels уже -1 по умолчанию, остальные массивы уже нули.
        return anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor

    # 1. Вычисляем матрицу IoU (num_gt, num_anchors)
    # iou_matrix[i, j] = IoU между i-м GT и j-м якорем
    iou_matrix = calculate_iou_matrix(gt_boxes_norm, all_anchors_norm)

    # 2. Находим для каждого якоря: максимальное IoU и индекс GT, с которым оно достигнуто.
    # axis=0 -> максимум по столбцам (для каждого якоря)
    max_iou_per_anchor[:] = np.max(iou_matrix, axis=0)
    matched_gt_idx_per_anchor = np.argmax(iou_matrix, axis=0) # Индекс лучшего GT для каждого якоря

    # 3. Назначаем якоря на основе порогов IoU с ЛУЧШИМ GT для каждого якоря
    # Positive: IoU >= pos_iou_thresh (с ЛЮБЫМ GT, но мы используем max_iou_per_anchor, что equivalentно лучшему)
    anchor_labels[max_iou_per_anchor >= pos_iou_thresh] = 1

    # Ignored: neg_iou_thresh <= IoU < pos_iou_thresh
    anchor_labels[(max_iou_per_anchor >= neg_iou_thresh) & (max_iou_per_anchor < pos_iou_thresh)] = 0

    # Negative: IoU < neg_iou_thresh (остаются -1 по умолчанию)


    # 4. Шаг "Гарантированного назначения": Для каждого GT-бокса, якорь с НАИЛУЧШИМ IoU (даже если < pos_iou_thresh) становится позитивным.
    # Это предотвращает ситуации, когда ни один якорь не был назначен конкретному GT.
    # axis=1 -> максимум по строкам (для каждого GT)
    best_anchor_idx_per_gt = np.argmax(iou_matrix, axis=1) # Индекс лучшего якоря для каждого GT

    # Принудительно делаем эти якоря позитивными.
    # Используем np.unique на всякий случай, если разные GT выбрали один и тот же якорь как лучший.
    anchor_labels[np.unique(best_anchor_idx_per_gt)] = 1


    # 5. Собираем matched_gt_boxes и matched_gt_class_ids для ВСЕХ якорей.
    # Для позитивных якорей (как назначенных по порогу, так и "гарантированных"),
    # мы используем тот GT, который их сделал позитивными (то есть, тот, который дал им max_iou_per_anchor),
    # ИЛИ тот GT, который их "выбрал" при гарантированном назначении.
    # Важно, чтобы matched_gt_idx_per_anchor для "гарантированных" якорей указывал на ПРАВИЛЬНЫЙ GT.
    # Наш код уже делает это благодаря циклу в предыдущей версии, где мы обновляли matched_gt_idx_per_anchor
    # для best_anchor_idx_per_gt. Здесь мы сохраним эту логику.

    # Создаем временный массив для отслеживания, какой GT действительно сопоставлен с каждым якорем
    # Изначально используем индекс лучшего GT по IoU
    final_matched_gt_idx = np.copy(matched_gt_idx_per_anchor)

    # Для каждого GT, найдем его "лучший" якорь и обновим final_matched_gt_idx для этого якоря,
    # указывая на текущий gt_idx_loop. Это переписывает сопоставление по max_iou_per_anchor,
    # гарантируя, что "гарантированный" якорь сопоставлен с GT, который его выбрал.
    for gt_idx_loop in range(num_gt):
        anchor_to_update_idx = best_anchor_idx_per_gt[gt_idx_loop]
        final_matched_gt_idx[anchor_to_update_idx] = gt_idx_loop
        # Для этого "гарантированного" якоря, его метка уже была установлена в 1 на шаге 4.
        # Его max_iou_per_anchor также должен отражать IoU с этим "его" GT.
        max_iou_per_anchor[anchor_to_update_idx] = iou_matrix[gt_idx_loop, anchor_to_update_idx]


    # Теперь используем final_matched_gt_idx для получения правильных GT боксов и классов
    matched_gt_boxes_for_anchors = gt_boxes_norm[final_matched_gt_idx]
    matched_gt_class_ids_for_anchors = gt_class_ids[final_matched_gt_idx]

    # === Проверка на всякий случай ===
    # Убедимся, что для позитивных якорей, полученный matched_gt_idx действительно указывает на GT, который сделал его позитивным.
    # Этот шаг в продакшн коде не нужен, но полезен для отладки.
    # positive_indices_check = np.where(anchor_labels == 1)[0]
    # for idx in positive_indices_check:
    #      gt_idx = final_matched_gt_idx[idx]
    #      actual_iou = iou_matrix[gt_idx, idx]
    #      if not np.isclose(actual_iou, max_iou_per_anchor[idx]):
    #          logger.warning(f"IoU mismatch for positive anchor {idx}: expected {max_iou_per_anchor[idx]:.4f}, got {actual_iou:.4f} with matched GT {gt_idx}")
    # ================================

    return anchor_labels, matched_gt_boxes_for_anchors, matched_gt_class_ids_for_anchors, max_iou_per_anchor


def encode_box_targets(anchors, matched_gt_boxes):
    """
    Кодирует смещения от якорей до сопоставленных GT-боксов в формат (tx, ty, tw, th).

    Args:
        anchors (np.ndarray): Массив якорей (N, 4) в формате [xmin, ymin, xmax, ymax].
                              Обычно это подмножество якорей (например, только позитивные).
        matched_gt_boxes (np.ndarray): Массив GT-боксов (N, 4) в формате [xmin, ymin, xmax, ymax],
                                       сопоставленных с каждым якорем в `anchors`.

    Returns:
        np.ndarray: Массив закодированных целей регрессии (N, 4) в формате (tx, ty, tw, th).
    """
    # Вычисляем ширину, высоту и центры якорей
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h

    # Вычисляем ширину, высоту и центры сопоставленных GT-боксов
    gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
    gt_cx = matched_gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = matched_gt_boxes[:, 1] + 0.5 * gt_h

    # Добавляем маленький эпсилон для стабильности (избежать деления на ноль или логарифма от нуля)
    epsilon = 1e-7
    anchor_w = np.maximum(anchor_w, epsilon)
    anchor_h = np.maximum(anchor_h, epsilon)
    gt_w = np.maximum(gt_w, epsilon)
    gt_h = np.maximum(gt_h, epsilon)

    # Вычисляем закодированные смещения
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)


    # === [ИСПРАВЛЕНО] Добавляем масштабирование смещений для балансировки градиентов ===
    # Эти множители должны быть согласованы с обратным преобразованием в _decode_boxes
    tx *= 0.1
    ty *= 0.1
    tw *= 0.2
    th *= 0.2


    # --- Клиппинг значений (очень важно для стабильности регрессии) ---
    # Этот клиппинг остается полезным даже после масштабирования, чтобы избежать
    # экстремальных значений, если якорь и GT очень сильно отличаются.
    # log(1000/16) ≈ 4.135. Это стандартное значение, ограничивающее максимальное изменение размера в 62.5 раз.
    bbox_xform_clip = np.log(1000. / 16.) # Или можно взять из конфига

    # Ограничиваем максимальное смещение центра относительно размера якоря.
    # xy_clip = 2.5 означает, что центр GT не может быть дальше, чем на 2.5 размера якоря по любой оси.
    xy_clip = 2.5 # Можно взять из конфига

    tx = np.clip(tx, -xy_clip, xy_clip)
    ty = np.clip(ty, -xy_clip, xy_clip)
    tw = np.clip(tw, -bbox_xform_clip, bbox_xform_clip)
    th = np.clip(th, -bbox_xform_clip, bbox_xform_clip)

    # Собираем закодированные цели в массив
    targets = np.stack([tx, ty, tw, th], axis=1)
    return targets


# --- Блок 3: Класс-генератор и создание датасета ---

class DataGenerator:
    def __init__(self, config, all_anchors, is_training=True, debug_mode=False):
        """
        Инициализирует генератор данных.

        Args:
            config (dict): Конфигурационный словарь.
            all_anchors (np.ndarray): Массив всех сгенерированных якорей.
            is_training (bool): True для тренировочного датасета.
            debug_mode (bool): Если True, __call__ возвращает доп. debug_info.
        """
        self.config = config
        self.is_training = is_training
        self.debug_mode = debug_mode # Этот режим теперь влияет только на то, что возвращает __call__

        # --- Настройка путей к данным ---
        dataset_path = Path(config['dataset_path'])
        img_subdir = config['train_images_subdir'] if is_training else config['val_images_subdir']
        ann_subdir = config['train_annotations_subdir'] if is_training else config['val_annotations_subdir']
        self.image_dir = dataset_path / img_subdir
        self.annot_dir = dataset_path / ann_subdir

        # --- Сопоставление файлов ---
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(list(self.image_dir.glob(ext)))

        all_annot_files = list(self.annot_dir.glob('*.xml'))

        image_file_dict = {p.stem: p for p in all_image_files}
        annot_file_dict = {p.stem: p for p in all_annot_files}

        # Находим общие имена файлов (без расширения) и сортируем их
        matched_basenames = sorted(list(image_file_dict.keys() & annot_file_dict.keys()))

        # Собираем полные пути к изображениям и аннотациям для совпадающих файлов
        self.image_paths = [image_file_dict[basename] for basename in matched_basenames]
        self.annot_paths = [annot_file_dict[basename] for basename in matched_basenames]

        if not self.image_paths:
            logging.error(f"Не найдено совпадающих пар изображений/аннотаций в {self.image_dir} и {self.annot_dir}")
        else:
             logging.info(f"Найдено {len(self.image_paths)} пар изображений/аннотаций для {'тренировки' if is_training else 'валидации'}.")


        # --- Загрузка параметров из конфига ---
        self.input_shape = config['input_shape']
        self.class_mapping = {name: i for i, name in enumerate(config['class_names'])}
        self.num_classes = config['num_classes']
        self.pos_iou_thresh = config['anchor_positive_iou_threshold']
        self.neg_iou_thresh = config['anchor_ignore_iou_threshold']
        self.all_anchors = all_anchors # Якоря уже сгенерированы в create_dataset и переданы сюда


        # --- [ИСПРАВЛЕНО] Предрасчет размеров и проверка согласованности num_anchors ---
        self.fpn_strides = [8, 16, 32] # Получаем страйды

        # Вычисляем ожидаемое количество якорей на ячейку на основе scales и ratios
        expected_na_per_level = len(config['anchor_scales']) * len(config['anchor_ratios'])

        # Если значение в конфиге не совпадает с расчетным, логируем предупреждение и ИСПРАВЛЯЕМ его
        if config['num_anchors_per_level'] != expected_na_per_level:
            logging.warning(
                f"num_anchors_per_level={config['num_anchors_per_level']} в конфиге, "
                f"но по расчёту len(scales)*len(ratios) должно быть {expected_na_per_level}. "
                f"Использую вычислённое значение."
            )
            # Исправляем значение в конфиге, чтобы все последующие расчеты были верными
            self.config['num_anchors_per_level'] = expected_na_per_level

        # Используем (возможно, исправленное) значение
        num_base_anchors = self.config['num_anchors_per_level']


        self.total_anchors_count = 0
        self.level_anchor_counts = [] # Количество якорей НА УРОВНЕ
        self.level_output_shapes = [] # Форма (H, W, num_base_anchors) НА УРОВНЕ

        H, W = self.input_shape[:2]

        for stride in self.fpn_strides:
            fh, fw = H // stride, W // stride
            level_anchor_count = fh * fw * num_base_anchors
            self.level_anchor_counts.append(level_anchor_count)
            self.level_output_shapes.append((fh, fw, num_base_anchors))
            self.total_anchors_count += level_anchor_count


        # --- [ИСПРАВЛЕНО] Убираем tf.debugging.assert_equal и используем обычный assert ---
        # Эта проверка выполняется один раз при создании генератора, а не в графе TensorFlow
        assert self.all_anchors.shape[0] == self.total_anchors_count, \
            f"Количество сгенерированных якорей all_anchors ({self.all_anchors.shape[0]}) не совпадает с ожидаемым по конфигу ({self.total_anchors_count})!"
        logging.info(f"Ожидаемое и фактическое количество якорей совпадает: {self.total_anchors_count}")


        # --- Настройка аугментации ---
        if self.is_training and self.config.get('use_augmentation', True):
            logging.info("Аугментация ВКЛЮЧЕНА для тренировки.")
            self.augmenter = augmentations.get_detector_train_augmentations(
                self.input_shape[0],  # img_height
                self.input_shape[1]  # img_width
            )
        else:
            logging.info("Аугментация ВЫКЛЮЧЕНА.")
            self.augmenter = None

    def __len__(self):
        """Возвращает общее количество изображений в датасете."""
        return len(self.image_paths)

    def __call__(self):
        """Генератор данных для tf.data.Dataset."""
        # Перемешиваем пути к файлам в начале каждой эпохи для тренировочного датасета
        if self.is_training:
            # Используем numpy.random для перемешивания, т.к. __call__ вызывается в контексте numpy/python, не tf.graph
            # Сид numpy.random установлен глобально перед запуском тренировки
            shuffled_indices = np.random.permutation(len(self.image_paths))
            image_paths = [self.image_paths[i] for i in shuffled_indices]
            annot_paths = [self.annot_paths[i] for i in shuffled_indices]
        else:
            image_paths = self.image_paths
            annot_paths = self.annot_paths

        for i in range(len(image_paths)):
            image_path, annot_path = image_paths[i], annot_paths[i]

            try: # try-except для обработки ошибок на уровне одного изображения
                # 1. Загрузка и ресайз изображения
                image = cv2.imread(str(image_path))
                if image is None:
                    logging.warning(f"Не удалось прочитать изображение: {image_path}, пропускаем.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations ожидает RGB uint8
                h_orig, w_orig, _ = image_rgb.shape

                h_target, w_target = self.input_shape[:2]
                # Ресайз изображения до целевого размера
                image_resized = cv2.resize(image_rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR)


                # 2. Парсинг аннотаций и ресайз рамок до целевого размера изображения (в пикселях)
                gt_boxes_pixels_orig, gt_class_names_orig = parse_voc_xml(annot_path)

                gt_boxes_resized_pixels = []
                gt_classes_resized = []
                if gt_boxes_pixels_orig:
                    for box, class_name in zip(gt_boxes_pixels_orig, gt_class_names_orig):
                        x1, y1, x2, y2 = box
                        # Ресайз координат пропорционально изменению размера изображения
                        px1 = (x1 / w_orig) * w_target
                        py1 = (y1 / h_orig) * h_target
                        px2 = (x2 / w_orig) * w_target
                        py2 = (y2 / h_orig) * h_target

                        # === [ВАЖНО] Клиппинг рамок ДО аугментации/фильтрации ===
                        # Убедимся, что рамки не выходят за границы изображения после ресайза
                        px1, py1, px2, py2 = np.clip([px1, py1, px2, py2], 0, max(h_target, w_target)) # Клиппинг по максимальной стороне

                        # Простая проверка на валидность рамки (ширина и высота > 0) после клиппинга
                        if px2 > px1 and py2 > py1:
                            gt_boxes_resized_pixels.append([px1, py1, px2, py2])
                            gt_classes_resized.append(class_name)
                        else:
                            logging.debug(f"Отброшена невалидная рамка после ресайза/клиппинга в {image_path.name}: [{px1}, {py1}, {px2}, {py2}] (orig: {box})")


                # --- Подготовка данных для аугментации ---
                # Albumentations ожидает рамки в формате pascal_voc (xmin, ymin, xmax, ymax)
                image_for_aug = image_resized.copy() # Работаем с копией
                bboxes_for_aug = [list(b) for b in gt_boxes_resized_pixels] # Копируем список и рамки
                class_labels_for_aug = list(gt_classes_resized) # Копируем список меток


                # --- Применение Аугментации ---
                image_augmented_uint8 = image_for_aug # Инициализация на случай, если аугментация выключена или не применяется
                gt_boxes_augmented_pixels = bboxes_for_aug
                gt_class_names_augmented = class_labels_for_aug


                if self.augmenter: # Проверяем, есть ли augmenter
                    if bboxes_for_aug: # Применяем аугментацию только если есть что аугментировать
                        try:
                            augmented = self.augmenter(
                                image=image_for_aug, # Albumentations работает с uint8 RGB
                                bboxes=bboxes_for_aug,
                                class_labels_for_albumentations=class_labels_for_aug
                            )
                            image_aug_temp = augmented['image']
                            gt_boxes_aug_temp = augmented['bboxes'] # Рамки в пикселях после аугментации
                            gt_class_names_aug_temp = augmented['class_labels_for_albumentations'] # Имена классов после аугментации (некоторые могут быть отброшены)

                            # === [ВАЖНО] Защита от полной потери GT-боксов после аугментации ===
                            # Если после аугментации не осталось боксов, а ИЗНАЧАЛЬНО они были (до аугментации),
                            # то используем данные до аугментации, чтобы не терять позитивные примеры.
                            if not gt_boxes_aug_temp or len(gt_boxes_aug_temp) == 0:
                                if len(bboxes_for_aug) > 0: # Проверяем, были ли GT до аугментации
                                    logging.warning(f"!!! ВСЕ GT БОКСЫ ПОТЕРЯНЫ ПОСЛЕ АУГМЕНТАЦИИ для {image_path.name}, ИСПОЛЬЗУЕТСЯ ОРИГИНАЛ (ресайзнутый+клиппинг) !!!")
                                    # В этом случае image_augmented_uint8 и gt_boxes_augmented_pixels
                                    # должны остаться равны image_for_aug и bboxes_for_aug
                                else:
                                     # Если GT не было и до аугментации, просто используем результат аугментации (пустые GT)
                                     image_augmented_uint8 = image_aug_temp
                                     gt_boxes_augmented_pixels = gt_boxes_aug_temp
                                     gt_class_names_augmented = gt_class_names_aug_temp
                            else:
                                # Если после аугментации GT остались, используем аугментированные данные
                                image_augmented_uint8 = image_aug_temp
                                gt_boxes_augmented_pixels = gt_boxes_aug_temp
                                gt_class_names_augmented = gt_class_names_aug_temp

                        except Exception as e:
                            logging.error(f"Ошибка при применении аугментации для {image_path.name}: {e}. Используется оригинал.")
                            # В случае ошибки, используем данные до аугментации
                            image_augmented_uint8 = image_for_aug
                            gt_boxes_augmented_pixels = bboxes_for_aug
                            gt_class_names_augmented = class_labels_for_aug # Возвращаем метки до аугментации


                    # Если self.augmenter есть, но bboxes_for_aug пустой, то image_augmented_uint8 и др.
                    # должны остаться равны image_for_aug и bboxes_for_aug (данным до аугментации)
                    # Этот случай уже обрабатывается в инициализации image_augmented_uint8 и др. перед блоком if self.augmenter:
                    pass # Nothing more to do if no GT for aug
                # Если self.augmenter is None, image_augmented_uint8 и др. также остаются image_for_aug и bboxes_for_aug


                # 4. Финальная нормализация изображения (для подачи в модель)
                # image_augmented_uint8 - это uint8 RGB после аугментации или ресайза

                if self.config.get('image_normalization_method') == 'imagenet':
                     # Нормализация для EfficientNet (диапазон [-1, 1])
                     image_for_model = tf.keras.applications.efficientnet.preprocess_input(
                         image_augmented_uint8.astype(np.float32) # EfficientNet ожидает float32
                     )
                else:
                     # Простая нормализация в диапазон [0, 1]
                     logging.debug(f"Для {image_path.name} используется нормализация делением на 255.0")
                     image_for_model = image_augmented_uint8.astype(np.float32) / 255.0


                # 5. Обработка финализированных GT-боксов (после аугментации/отката)
                # Преобразуем пиксельные рамки GT в нормализованные [0, 1]
                gt_boxes_final_norm = np.array(gt_boxes_augmented_pixels, dtype=np.float32)
                if gt_boxes_final_norm.shape[0] > 0:
                     gt_boxes_final_norm /= np.array([w_target, h_target, w_target, h_target], dtype=np.float32)
                     gt_boxes_final_norm = np.clip(gt_boxes_final_norm, 0.0, 1.0)

                     # === [ВАЖНО] Финальная фильтрация невалидных рамок после аугментации/клиппинга ===
                     # Проверяем, что ширина и высота рамки положительны после всех преобразований
                     valid_indices = (gt_boxes_final_norm[:, 2] > gt_boxes_final_norm[:, 0]) & \
                                     (gt_boxes_final_norm[:, 3] > gt_boxes_final_norm[:, 1])

                     gt_boxes_norm = gt_boxes_final_norm[valid_indices]
                     # Применяем ту же фильтрацию к именам классов перед преобразованием в ID
                     gt_class_names_final_valid = [gt_class_names_augmented[i] for i, valid in enumerate(valid_indices) if valid]

                     # Преобразуем имена классов в числовые ID
                     # Используем .get() с None на случай, если имя класса по какой-то причине не в class_mapping
                     gt_class_ids = np.array([self.class_mapping.get(name) for name in gt_class_names_final_valid if self.class_mapping.get(name) is not None], dtype=np.int32)
                     gt_boxes_norm = gt_boxes_norm[:len(gt_class_ids)] # Обрежем боксы, если какие-то классы были None

                else:
                    # Если gt_boxes_augmented_pixels был пуст, то и gt_boxes_norm пуст
                    gt_boxes_norm = np.empty((0, 4), dtype=np.float32)
                    gt_class_ids = np.empty((0,), dtype=np.int32)


                # === [ВАЖНО] Проверка на пустые GT после всех фильтраций ===
                # Если после всех этапов не осталось GT объектов, мы не можем назначить якоря.
                # В этом случае, возвращаем изображение и пустые y_true.
                if gt_boxes_norm.shape[0] == 0:
                     # logging.warning(f"Нет валидных GT боксов для {image_path.name} после обработки. Возвращаем пустые y_true.")
                     # Возвращаем пустые y_true плоские тензоры правильного размера
                     y_true_reg_flat = np.zeros((self.total_anchors_count, 4), dtype=np.float32)
                     y_true_cls_flat = np.full((self.total_anchors_count, self.num_classes), -1.0, dtype=np.float32) # Все якоря негативные (-1)

                     if not self.debug_mode:
                         yield image_for_model, (y_true_reg_flat, y_true_cls_flat)
                     # else: # В debug_mode нужно вернуть все info, но это усложнит код. Отключим debug_mode для этого генератора.
                     #     pass # В этом режиме не поддерживаем пустые GT пока

                     continue # Пропускаем оставшуюся логику и переходим к следующему изображению


                # 6. Назначение GT якорям и кодирование регрессионных целей
                # all_anchors_norm уже нормализованы и хранятся в self.all_anchors

                # === [ВАЖНО] Проверка: количество GT объектов должно быть >= 1 для assign_gt_to_anchors ===
                # Эта проверка уже сделана выше (gt_boxes_norm.shape[0] == 0)

                anchor_labels, matched_gt_boxes, matched_gt_class_ids, max_iou_per_anchor = assign_gt_to_anchors(
                    gt_boxes_norm,           # Нормализованные GT рамки после обработки
                    gt_class_ids,            # ID классов GT после обработки
                    self.all_anchors,        # ВСЕ нормализованные якоря (плоский список)
                    self.pos_iou_thresh,     # Порог IoU для позитивных
                    self.neg_iou_thresh      # Порог IoU для игнорируемых
                )

                # 7. Формирование плоских тензоров y_true
                # Эти тензоры имеют форму (TotalAnchors, ...)
                y_true_cls_flat = np.zeros((self.total_anchors_count, self.num_classes), dtype=np.float32)
                y_true_reg_flat = np.zeros((self.total_anchors_count, 4), dtype=np.float32)

                # Обработка позитивных якорей
                positive_indices = np.where(anchor_labels == 1)[0]
                if len(positive_indices) > 0:
                    # Убедимся, что индексы не выходят за границы matched_gt_class_ids / matched_gt_boxes
                    valid_pos_indices = positive_indices[positive_indices < len(matched_gt_class_ids)]

                    y_true_cls_flat[valid_pos_indices] = tf.keras.utils.to_categorical(
                        matched_gt_class_ids[valid_pos_indices], # Используем ID классов сопоставленных GT
                        num_classes=self.num_classes
                    )

                    y_true_reg_flat[valid_pos_indices] = encode_box_targets(
                        self.all_anchors[valid_pos_indices], # Якоря для позитивных индексов
                        matched_gt_boxes[valid_pos_indices]  # Сопоставленные GT боксы для позитивных индексов
                    )

                # Обработка игнорируемых якорей (метка -1)
                ignore_indices = np.where(anchor_labels == 0)[0]
                y_true_cls_flat[ignore_indices] = -1.0

                # Негативные якоря (-1) остаются нулями в y_true_cls_flat (что и ожидается Focal Loss), и нулями в y_true_reg_flat


                # === [ВАЖНО] Возвращаем ИЗОБРАЖЕНИЕ и ДВА ПЛОСКИХ y_true тензора ===
                # Dataset.batch() добавит батч-размерность позже.
                # DetectorLoss будет отвечать за "разворачивание" этих плоских тензоров по уровням FPN.
                if not self.debug_mode:
                    # В обычном режиме возвращаем только изображение и y_true (плоские)
                    yield image_for_model, (y_true_reg_flat, y_true_cls_flat)
                else:
                    # В отладочном режиме возвращаем доп. информацию для plot_utils и анализа
                    debug_info = {
                        "image_path": str(image_path),
                        "image_original": image_rgb.copy(), # uint8, RGB
                        "gt_boxes_original": np.array(gt_boxes_pixels_orig, dtype=np.int32),
                        "gt_class_names_original": np.array(gt_class_names_orig, dtype=str),
                        "image_augmented": image_augmented_uint8.copy(), # uint8, RGB
                        "gt_boxes_augmented": np.array(gt_boxes_augmented_pixels, dtype=np.float32),
                        "gt_class_names_augmented": np.array(gt_class_names_augmented, dtype=str),
                        "all_anchors_norm": self.all_anchors,
                        "anchor_labels": anchor_labels, # Метки (-1, 0, 1)
                        "max_iou_per_anchor": max_iou_per_anchor, # Макс IoU для каждого якоря
                        # y_true_reg и y_true_cls плоские также доступны, если нужны для отладки
                        # "y_true_reg_flat": y_true_reg_flat,
                        # "y_true_cls_flat": y_true_cls_flat,
                    }
                    # В debug_mode возвращаем изображение, y_true (плоские), и debug_info
                    yield image_for_model, (y_true_reg_flat, y_true_cls_flat), debug_info


            except Exception as e: # Ловим любые оставшиеся ошибки при обработке изображения
                logging.error(f"Критическая ошибка при обработке {image_path.name}: {e}", exc_info=True) # exc_info=True выведет traceback
                # Пропускаем этот пример и продолжаем с генерацией следующего
                continue


def create_dataset(config, is_training=True, batch_size=8, debug_mode=False) -> tf.data.Dataset:
    """
    Создает tf.data.Dataset для обучения или валидации.

    Args:
        config (dict): Конфигурационный словарь.
        is_training (bool): True для тренировочного датасета, False для валидационного.
        batch_size (int): Размер батча.
        debug_mode (bool): Если True, генератор возвращает доп. debug_info.

    Returns:
        tf.data.Dataset: Готовый датасет.
    """
    logging.info(f"Начало создания tf.data.Dataset (is_training={is_training}, batch_size={batch_size}, debug_mode={debug_mode})")

    # 1. Генерируем все якоря ОДИН РАЗ
    # generate_all_anchors ожидает shape, strides, scales, ratios
    fpn_strides = [8, 16, 32] # Эти страйды должны соответствовать модели
    all_anchors = generate_all_anchors(
        config['input_shape'],
        fpn_strides,
        config['anchor_scales'], # Скалярные множители (например, [1.0, 1.26, 1.59])
        config['anchor_ratios'] # Соотношения сторон (например, [0.25, 0.5, 1.0, 2.0, 4.0])
    )
    logging.info(f"Всего сгенерировано якорей: {all_anchors.shape[0]}")


    # 2. Создаем экземпляр класса DataGenerator
    # all_anchors передаются в __init__
    generator_instance = DataGenerator(config, all_anchors, is_training, debug_mode=debug_mode)

    # 3. Определяем выходную сигнатуру датасета
    # Это форма и тип данных ОДНОГО примера, который выдает генератор (__call__).
    # Генератор теперь выдает: изображение (Tensor), кортеж (y_true_reg_flat, y_true_cls_flat)
    # y_true_reg_flat: (TotalAnchors, 4)
    # y_true_cls_flat: (TotalAnchors, num_classes)

    total_anchors_count = generator_instance.total_anchors_count # Используем предрасчет из генератора

    output_signature = (
        tf.TensorSpec(shape=config['input_shape'], dtype=tf.float32), # Изображение
        # Кортеж y_true
        (
            tf.TensorSpec(shape=(total_anchors_count, 4), dtype=tf.float32), # y_true_reg_flat
            tf.TensorSpec(shape=(total_anchors_count, config['num_classes']), dtype=tf.float32) # y_true_cls_flat
        )
        # Debug info добавляется только если debug_mode=True
        # Сигнатура debug_info остается сложной и требует, чтобы debug_info содержал только TensorFlow-совместимые типы с определенными формами
        # Для надежности, в debug_mode=True, лучше не использовать .batch() и .prefetch()
    )

    # Если debug_mode включен, выходная сигнатура включает debug_info.
    # Важно: tf.data.Dataset.batch не должен использоваться в debug_mode=True,
    # т.к. debug_info содержит python объекты/numpy массивы с нефиксированными формами.
    if debug_mode:
        logging.warning("debug_mode=True включен. Dataset не будет батчиться и перемешиваться через Dataset API.")
        # В этом режиме возвращается тройной выход (image, y_true_tuple, debug_info)
        # Сигнатура должна это отражать. Используем нестрогую сигнатуру для debug_info.
        output_signature = (
             tf.TensorSpec(shape=config['input_shape'], dtype=tf.float32),
             (
                 tf.TensorSpec(shape=(total_anchors_count, 4), dtype=tf.float32),
                 tf.TensorSpec(shape=(total_anchors_count, config['num_classes']), dtype=tf.float32)
             ),
             # Сигнатура для debug_info - делаем ее нестрогой для гибкости в debug_mode.
             # Предполагаем, что debug_info - это словарь с нестрогими формами.
             tf.nest.map_structure(lambda spec: tf.TensorSpec(shape=None, dtype=spec.dtype) if isinstance(spec, tf.TensorSpec) else spec, {
                  "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
                  "image_original": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                  "gt_boxes_original": tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
                  "gt_class_names_original": tf.TensorSpec(shape=(None,), dtype=tf.string),
                  "image_augmented": tf.TensorSpec(shape=config['input_shape'], dtype=tf.uint8),
                  "gt_boxes_augmented": tf.TensorSpec(shape=(None, 4), dtype=tf.float32), # Рамки после аугментации, могут быть пустыми
                  "gt_class_names_augmented": tf.TensorSpec(shape=(None,), dtype=tf.string), # Метки после аугментации, могут быть пустыми
                  "all_anchors_norm": tf.TensorSpec(shape=(total_anchors_count, 4), dtype=tf.float32),
                  "anchor_labels": tf.TensorSpec(shape=(total_anchors_count,), dtype=tf.int32),
                  "max_iou_per_anchor": tf.TensorSpec(shape=(total_anchors_count,), dtype=tf.float32),
            })
        )
        # Создаем Dataset без batch/shuffle/prefetch в debug mode
        dataset = tf.data.Dataset.from_generator(generator_instance, output_signature=output_signature)
        return dataset # Возвращаем небатченный, неперемешанный датасет для отладки


    # --- Настройка пайплайна для обучения/валидации (batching, shuffling, prefetching) ---
    # В этом блоке debug_mode=False
    dataset = tf.data.Dataset.from_generator(generator_instance, output_signature=output_signature)

    # Зацикливаем датасет для обучения
    if is_training:
        # [ИСПРАВЛЕНО] УДАЛЕН .repeat()
        # dataset = dataset.repeat() # Зацикливаем тренировочный датасет

        # Перемешиваем
        # [ИСПРАВЛЕНО] Буфер перемешивания не может быть меньше 2.
        # Используем max(2, ...)
        buffer_size = max(2, min(len(generator_instance), batch_size * 3))
        if len(generator_instance) > 1: # Перемешиваем только если есть что перемешивать
             dataset = dataset.shuffle(buffer_size=buffer_size)
             logging.info(f"Перемешивание тренировочного датасета включено с buffer_size={buffer_size}.")
        else:
             logging.warning("Размер тренировочного датасета <= 1, перемешивание пропущено.")


    # Батчим данные.
    # [ИСПРАВЛЕНО] drop_remainder=False для всех случаев, как в предложенном коде.
    # Это предотвращает потерю примеров в последнем батче, особенно на валидации.
    dataset = dataset.batch(batch_size, drop_remainder=False)
    logging.info(f"Batching настроен: batch_size={batch_size}, drop_remainder=False")


    # Prefetching для параллельной работы CPU и GPU (или CPU и CPU)
    # [ИСПРАВЛЕНИЕ] Добавляем num_parallel_calls в map (для ускорения CPU)
    # Это изменение не в этой функции, а в DataGenerator.__call__
    # Здесь добавляем .prefetch()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logging.info("Создание tf.data.Dataset завершено.")
    return dataset


# --- Тестовый запуск ---
if __name__ == '__main__':
    print("--- Тестирование data_loader_v3_standard.py ---")

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

    # [ИСПРАВЛЕНИЕ] Убедимся, что num_anchors_per_level в конфиге совпадает с ожидаемым
    expected_na = len(config['anchor_scales']) * len(config['anchor_ratios'])
    if expected_na != config['num_anchors_per_level']:
        print(f"ПРЕДУПРЕЖДЕНИЕ: num_anchors_per_level в конфиге ({config['num_anchors_per_level']}) не совпадает с расчетом по scales*ratios ({expected_na}).")
        print(f"Используется значение из конфига.")
        # Можно здесь обновить конфиг: config['num_anchors_per_level'] = expected_na


    print("\n1. Тест создания тренировочного датасета (1 батч, обычный режим):")
    try:
        # Используем достаточно большой батч, чтобы проверить drop_remainder
        train_dataset = create_dataset(config, is_training=True, batch_size=config.get('batch_size', 8), debug_mode=False)

        # [ИСПРАВЛЕНИЕ] Ожидаем 2 плоских тензора в y_true_tuple
        # Пройдемся по нескольким батчам, чтобы проверить зацикливание/перемешивание
        for batch_idx, (images, y_true_tuple) in enumerate(train_dataset.take(3)): # Возьмем 3 батча
            print(f"  - Батч #{batch_idx + 1}")
            print(f"    - Форма батча изображений: {images.shape}")
            print(f"    - Тип y_true: {type(y_true_tuple)}")
            print(f"    - Длина y_true (ожидаем 2): {len(y_true_tuple)}")

            # Проверяем формы каждого из 2 плоских тензоров в y_true
            y_true_reg_flat, y_true_cls_flat = y_true_tuple
            print(f"    - Форма y_true_reg_flat: {y_true_reg_flat.shape}")
            print(f"    - Форма y_true_cls_flat: {y_true_cls_flat.shape}")

            # Простая проверка формы
            assert len(images.shape) == 4 # (batch, H, W, C)
            assert images.shape[0] == config.get('batch_size', 8) # Размер батча

            assert len(y_true_reg_flat.shape) == 3 # (batch, TotalAnchors, 4)
            assert len(y_true_cls_flat.shape) == 3 # (batch, TotalAnchors, num_classes)

            # Убедимся, что общее количество якорей соответствует ожидаемому
            # Генератор для подсчета создается только для получения total_anchors_count
            generator_for_count = DataGenerator(config, generate_all_anchors(config['input_shape'], [8, 16, 32], config['anchor_scales'], config['anchor_ratios']), is_training=True, debug_mode=False)
            expected_total_anchors = generator_for_count.total_anchors_count
            assert y_true_reg_flat.shape[1] == expected_total_anchors
            assert y_true_cls_flat.shape[1] == expected_total_anchors

        print("  - [SUCCESS] Тест 1 (обычный режим) пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 1: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Тест создания валидационного датасета (1 батч, debug_mode=False):")
    try:
        # Валидационный датасет не зацикливается и не перемешивается.
        # Возьмем только один батч.
        val_dataset = create_dataset(config, is_training=False, batch_size=config.get('batch_size', 8), debug_mode=False)

        for batch_idx, (images_val, y_true_tuple_val) in enumerate(val_dataset.take(1)):
            print(f"  - Батч #{batch_idx + 1}")
            print(f"    - Форма батча изображений: {images_val.shape}")
            print(f"    - Тип y_true: {type(y_true_tuple_val)}")
            print(f"    - Длина y_true (ожидаем 2): {len(y_true_tuple_val)}")

            y_true_reg_flat_val, y_true_cls_flat_val = y_true_tuple_val
            print(f"    - Форма y_true_reg_flat: {y_true_reg_flat_val.shape}")
            print(f"    - Форма y_true_cls_flat: {y_true_cls_flat_val.shape}")

            assert len(images_val.shape) == 4
            # Для валидации drop_remainder=False, размер батча может быть меньше
            # assert images_val.shape[0] <= config.get('batch_size', 8)
            assert len(y_true_tuple_val) == 2

        print("  - [SUCCESS] Тест 2 (валидация) пройден.")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 2: {e}")
        import traceback

        traceback.print_exc()

    print("\n3. Тест получения одного примера в debug_mode=True (НЕ ЧЕРЕЗ TF.DATA.DATASET.BATCH):")
    try:
        # В debug_mode=True, рекомендуется итерироваться по экземпляру DataGenerator напрямую,
        # чтобы избежать проблем с tf.data и динамическими/numpy типами в debug_info.

        # [ИСПРАВЛЕНИЕ] Генерируем якоря здесь для debug-генератора
        all_anchors_for_debug = generate_all_anchors(config['input_shape'], [8, 16, 32], config['anchor_scales'], config['anchor_ratios'])

        debug_generator_instance = DataGenerator(
             config,
             all_anchors_for_debug, # Передаем сгенерированные якоря
             is_training=False, # Обычно отладка на валидационных данных
             debug_mode=True # Включаем debug_mode
        )

        # [ИСПРАВЛЕНИЕ] Вызываем генератор как функцию, чтобы получить итератор
        print("Попытка получить один пример напрямую из DataGenerator в debug_mode=True...")
        generator_object = debug_generator_instance() # Вызываем __call__()

        # Получаем первый элемент из итератора. Если генератор пуст, next() выбросит StopIteration.
        single_example = next(generator_object)


        # В debug_mode, __call__ Yield'ит: image_for_model, (y_true_reg_flat, y_true_cls_flat), debug_info
        # [ИСПРАВЛЕНИЕ] Распаковываем 3 элемента, как и обещано в debug_mode
        image_single, y_true_tuple_single, debug_info_single = single_example


        print(f"  - Форма изображения (single): {image_single.shape}")
        print(f"  - Тип y_true: {type(y_true_tuple_single)}, Длина: {len(y_true_tuple_single)}")

        y_reg_flat_single, y_cls_flat_single = y_true_tuple_single
        print(f"  - Форма y_true_reg_flat (single): {y_reg_flat_single.shape}")
        print(f"  - Форма y_true_cls_flat (single): {y_cls_flat_single.shape}")

        print(f"  - Тип debug_info: {type(debug_info_single)}")
        print(f"  - Ключи в debug_info: {list(debug_info_single.keys())}")
        print(f"  - Пример image_path из debug_info: {debug_info_single.get('image_path', 'N/A')}")
        # Проверяем тип и форму numpy массива из debug_info
        anchor_labels_from_debug = debug_info_single.get('anchor_labels', np.array([]))
        print(f"  - Пример anchor_labels (numpy) из debug_info: {type(anchor_labels_from_debug)}, форма: {anchor_labels_from_debug.shape}")

        # [ИСПРАВЛЕНИЕ] Убедимся, что общее количество якорей соответствует ожидаемому
        # Используем предрасчет из генератора
        expected_total_anchors_debug = debug_generator_instance.total_anchors_count
        assert y_reg_flat_single.shape[0] == expected_total_anchors_debug # Форма (TotalAnchors, 4) без батча
        assert y_cls_flat_single.shape[0] == expected_total_anchors_debug # Форма (TotalAnchors, num_classes) без батча
        assert anchor_labels_from_debug.shape[0] == expected_total_anchors_debug # Форма (TotalAnchors,)


        assert len(image_single.shape) == 3 # Ожидаем (H, W, C) без батча
        assert len(y_true_tuple_single) == 2
        assert len(y_reg_flat_single.shape) == 2 # Ожидаем (TotalAnchors, 4) без батча
        assert len(y_cls_flat_single.shape) == 2 # Ожидаем (TotalAnchors, num_classes) без батча
        assert isinstance(debug_info_single, dict)

        print("  - [SUCCESS] Тест 3 (debug_mode) пройден.")
    except StopIteration:
         print("  - [INFO] Тест 3: Генератор пуст (нет изображений).")
    except Exception as e:
        print(f"  - [ERROR] Ошибка в тесте 3 (debug_mode): {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Тестирование data_loader_v3_standard.py завершено ---")