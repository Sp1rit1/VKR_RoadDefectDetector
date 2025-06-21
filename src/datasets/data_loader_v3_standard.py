# RoadDefectDetector/src/datasets/data_loader_v3_standard.py

import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import math
import yaml
from pathlib import Path
import logging
import random # Для инициализации сида random (если используется)
import matplotlib.pyplot as plt # Для визуализации в тестовом блоке
import matplotlib.patches as patches # Также может пригодиться для отрисовки вручную

# Импорт Albumentations
try:
    import albumentations as A
except ImportError:
    print("Ошибка импорта albumentations. Установите: pip install albumentations opencv-python")
    # Предоставляем заглушку, если Albumentations не установлен
    class DummyA:
        # В новых версиях Albumentations сид управляется через numpy.random
        # Заглушка A.set_seed не нужна, т.к. мы будем вызывать np.random.seed
        @staticmethod
        def Compose(transforms, bbox_params=None):
             logger.warning("Albumentations не найден, используется заглушка для A.Compose.")
             # Возвращает функцию, которая просто возвращает входные данные в нужном формате словаря
             return lambda image, bboxes, class_labels_for_albumentations: {'image': image, 'bboxes': bboxes, 'class_labels_for_albumentations': class_labels_for_albumentations}

    A = DummyA()

# Import augmentations from your existing file
try:
    # Предполагаем структуру src/datasets/augmentations.py
    # Используем абсолютный импорт, если структура проекта позволяет, иначе относительный
    try:
        from src.datasets.augmentations import get_detector_train_augmentations
    except ImportError:
        from .augmentations import get_detector_train_augmentations
except ImportError:
    print("Ошибка импорта augmentations.py. Убедитесь, что он находится в src/datasets/")
    # Предоставляем заглушку функции аугментации, если импорт не удался
    def get_detector_train_augmentations(img_height, img_width):
        logger.warning("augmentations.py не найден, используется заглушка аугментации.")
        # Возвращает функцию, которая просто возвращает входные данные
        # Albumentations ожидает, что функция вернет словарь с ключами 'image', 'bboxes', 'class_labels_for_albumentations'
        return lambda image, bboxes, class_labels_for_albumentations: {'image': image, 'bboxes': bboxes, 'class_labels_for_albumentations': class_labels_for_albumentations}

# Импорт утилит визуализации
try:
    # Предполагаем структуру src/utils/plot_utils.py
    # Используем абсолютный импорт
    try:
        from src.utils import plot_utils # Импортируем как модуль
    except ImportError:
        # Попытка относительного импорта
        from ..utils import plot_utils # Предполагаем, что data_loader в src/datasets, plot_utils в src/utils
except ImportError:
    print("Ошибка импорта plot_utils.py. Убедитесь, что он находится в src/utils/")
    # Предоставляем заглушки для функций отрисовки, если импорт не удался
    class DummyPlotUtils:
        def set_global_seed(self, seed): pass
        def plot_image(self, ax, image_np, title=""): print(f"Dummy plot_image: {title}")
        def plot_boxes_on_image(self, ax, boxes, labels=None, scores=None, extra_info=None, color_index_base=None, linewidth=2, fontsize=8): print(f"Dummy plot_boxes_on_image: {len(boxes)} boxes")
        def plot_original_gt(self, ax, image_np_original, gt_objects): self.plot_image(ax, image_np_original, "Dummy Orig GT"); self.plot_boxes_on_image(ax, [obj['bbox'] for obj in gt_objects])
        def plot_augmented_gt(self, ax, image_np_augmented, augmented_gt_objects): self.plot_image(ax, image_np_augmented, "Dummy Aug GT"); self.plot_boxes_on_image(ax, [obj['bbox'] for obj in augmented_gt_objects])
        def plot_specific_anchors_on_image(self, ax, image_np_augmented, anchors_info_list, title="Dummy Anchors"): self.plot_image(ax, image_np_augmented, title); self.plot_boxes_on_image(ax, [info['bbox'] for info in anchors_info_list])
        def show_plot(self): print("Dummy show_plot")
        def save_plot(self, fig, filepath): print(f"Dummy save_plot to {filepath}")
    plot_utils = DummyPlotUtils()


# --- Настройка логирования ---
# Настраиваем базовый уровень логирования для файла
# Уровень DEBUG для очень детального вывода, INFO для общего прогресса
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Уровень логирования для этого файла. В отладочном режиме полезен DEBUG.
# По умолчанию оставляем INFO, DEBUG включается в if __name__ == '__main__':
# logger.setLevel(logging.INFO)


# --- Утилита для загрузки конфига ---
def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            logger.error(f"Содержимое файла конфига {config_path} не является словарем.")
            return None
        # logger.debug(f"Конфиг успешно загружен из {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Файл конфига не найден по пути: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML файла конфига {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при загрузке конфига {config_path}: {e}")
        return None


# --- Константы для типов якорей ---
POSITIVE_ANCHOR = 1
IGNORED_ANCHOR = -1 # Используем -1 для игнорируемого индекса
NEGATIVE_ANCHOR = 0


# --- Утилиты для работы с рамками ---

def calculate_iou(box1, box2):
    """
    Рассчитывает IoU между двумя рамками.
    Формат рамок: [x_min, y_min, x_max, y_max] в пикселях.
    """
    # Определяем координаты пересечения
    x_min_intersect = max(box1[0], box2[0])
    y_min_intersect = max(box1[1], box2[1])
    x_max_intersect = min(box1[2], box2[2])
    y_max_intersect = min(box1[3], box2[3])

    # Вычисляем площадь пересечения
    # Убедимся, что пересечение положительное (рамки пересекаются)
    intersect_width = max(0.0, x_max_intersect - x_min_intersect)
    intersect_height = max(0.0, y_max_intersect - y_min_intersect)
    intersection_area = intersect_width * intersect_height

    # Вычисляем площади обеих рамок
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Вычисляем площадь объединения
    # Избегаем отрицательных площадей
    union_area = max(0.0, box1_area + box2_area - intersection_area)

    # Избегаем деления на ноль
    if union_area == 0:
        # Если объединение нулевое (например, рамки точечные/линейные или не пересекаются), IoU равно 0.0.
        return 0.0

    return intersection_area / union_area

def convert_pascal_voc_to_cxcywh(bbox_pascal_voc):
    """Конвертирует [x_min, y_min, x_max, y_max] в [cx, cy, w, h]."""
    xmin, ymin, xmax, ymax = bbox_pascal_voc
    width = xmax - xmin
    height = ymax - ymin
    # Убедимся, что размеры неотрицательные
    width = max(0.0, width)
    height = max(0.0, height)
    cx = xmin + width / 2.0
    cy = ymin + height / 2.0
    return [cx, cy, width, height]

# Эта функция не нужна для data loader'а, но полезна для постобработки
# def convert_cxcywh_to_pascal_voc(bbox_cxcywh):
#     """Конвертирует [cx, cy, w, h] в [x_min, y_min, x_max, y_max]."""
#     cx, cy, width, height = bbox_cxcywh
#     xmin = cx - width / 2.0
#     ymin = cy - height / 2.0
#     xmax = cx + width / 2.0
#     ymax = cy + height / 2.0
#     return [xmin, ymin, xmax, ymax]


def encode_box(anchor_cxcywh, gt_cxcywh):
    """
    Кодирует дельты GT рамки относительно якоря в формат (tx, ty, tw, th).
    Согласно стандартной параметризации:
    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = log(gw / aw)
    th = log(gh / ah)
    """
    acx, acy, aw, ah = anchor_cxcywh
    gcx, gcy, gw, gh = gt_cxcywh

    # Добавляем epsilon к размерам якоря и GT, чтобы избежать деления на ноль или логарифма от нуля/отрицательного числа
    epsilon = 1e-5
    aw = max(aw, epsilon)
    ah = max(ah, epsilon)
    gw = max(gw, epsilon)
    gh = max(gh, epsilon)

    tx = (gcx - acx) / aw
    ty = (gcy - acy) / ah
    tw = math.log(gw / aw)
    th = math.log(gh / ah)

    return [tx, ty, tw, th]

# def decode_box(anchor_cxcywh, encoded_box):
#     """Декодирует предсказанные дельты обратно в координаты рамки [cx, cy, w, h]."""
#     # Эта функция нужна для постобработки
#     pass


# --- Загрузка и парсинг аннотаций ---

def parse_pascal_voc_xml(xml_path, img_width, img_height, class_names):
    """
    Парсит XML файл аннотаций PASCAL VOC и возвращает список объектов.
    Каждый объект: {'bbox': [xmin, ymin, xmax, ymax], 'class': 'имя_класса', 'class_id': id_класса}.
    Координаты в пикселях исходного изображения.
    """
    gt_objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Можно использовать размеры из XML, но надежнее брать размеры загруженного изображения
        # size = root.find('size')
        # xml_width = int(size.find('width').text)
        # xml_height = int(size.find('height').text)

        for obj in root.findall('object'):
            obj_class_name_elem = obj.find('name')
            # Проверяем, существует ли элемент 'name' и его текст
            if obj_class_name_elem is None or obj_class_name_elem.text is None:
                 logger.debug(f"Пропущен объект без имени класса в файле {xml_path}")
                 continue
            obj_class_name = obj_class_name_elem.text.strip() # Убираем пробелы

            # Пропускаем объекты с именами классов, которых нет в нашем конфиге
            if obj_class_name not in class_names:
                logger.debug(f"Пропущен объект с неизвестным классом '{obj_class_name}' в файле {xml_path}")
                continue

            obj_class_id = class_names.index(obj_class_name)

            bndbox = obj.find('bndbox')
            if bndbox is None:
                logger.warning(f"Пропущен объект '{obj_class_name}' без bndbox в файле {xml_path}")
                continue

            # Проверяем наличие всех координат
            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')

            if any(e is None or e.text is None for e in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                 logger.warning(f"Пропущен объект '{obj_class_name}' с неполными координатами в bndbox в файле {xml_path}")
                 continue

            try:
                # Используем float для точности, затем конвертируем в int при необходимости отрисовки
                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
            except ValueError:
                logger.warning(f"Некорректные числовые значения координат для объекта '{obj_class_name}' в файле {xml_path}. Пропускаем.")
                continue


            # Убедимся, что координаты в пределах исходного изображения и xmax > xmin, ymax > ymin
            # Приводим координаты к границам изображения (от 0 до width-1 / height-1)
            # Pascal VOC часто использует 1-индексацию, но библиотеки типа OpenCV/Pillow используют 0-индексацию.
            # Координаты в XML могут быть 1-индексированы (например, xmin=1 - это первый пиксель).
            # При чтении XML, xmin/ymin часто уменьшают на 1, чтобы получить 0-индексацию.
            # Проверим, как ваш create_data_splits.py и convert_voc_to_yolo.py обрабатывали это.
            # Если они просто читали int и не вычитали 1, то ваши рамки уже 0-индексированы
            # (т.е. xmin=0 - первый пиксель). Если вычитали 1, то они тоже 0-индексированы.
            # Давайте считать, что координаты в XML уже 0-индексированы или после парсинга приведены к 0-индексации.
            # Тогда границы - от 0 до img_width-1 и от 0 до img_height-1.
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(img_width - 1), xmax) # Граница по ширине
            ymax = min(float(img_height - 1), ymax) # Граница по высоте

            # Проверяем корректность рамки после приведения к границам
            # Минимальный размер рамки может быть 1x1 пиксель
            # Проверка xmax > xmin и ymax > ymin отсекает точечные или линейные рамки
            if xmax > xmin and ymax > ymin:
                 # Можно добавить проверку на минимальную площадь GT, если нужно
                 # (хотя Albumentations bbox_params уже фильтрует по area/visibility после аугментации)
                 # min_gt_area = 1.0 # Например, минимум 1x1 пиксель
                 # if (xmax - xmin) * (ymax - ymin) >= min_gt_area:
                     gt_objects.append({
                        'bbox': [xmin, ymin, xmax, ymax], # Координаты в пикселях исходного изображения
                        'class': obj_class_name,
                        'class_id': obj_class_id
                    })
                 # else:
                 #    logger.debug(f"Пропущен GT объект с площадью ниже порога в {xml_path}: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")
            else:
                # logger.warning(f"Пропущен некорректный GT bbox в {xml_path}: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}] после приведения к границам. Исходные: [{float(xmin_elem.text)}, {float(ymin_elem.text)}, {float(xmax_elem.text)}, {float(ymax_elem.text)}]. Возможно, GT был точкой или линией, или вне изображения.")
                logger.warning(f"Пропущен некорректный GT bbox ({obj_class_name}) в {xml_path}: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}] после приведения к границам. Исходные: [{xmin_elem.text}, {ymin_elem.text}, {xmax_elem.text}, {ymax_elem.text}]")


    except FileNotFoundError:
        # logger.error(f"Файл аннотации не найден: {xml_path}") # Это может быть слишком много логов
        return [] # Возвращаем пустой список, если файл не найден
    except ET.ParseError as e:
        logger.error(f"Ошибка парсинга XML файла {xml_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при парсинге XML файла {xml_path}: {e}", exc_info=True)
        return []

    # logger.debug(f"Успешно спарсено {len(gt_objects)} объектов из {xml_path}")
    return gt_objects

# --- Генерация якорей ---

def generate_all_anchors(image_shape, config):
    """
    Генерирует все якоря для всех ячеек сетки на каждом уровне FPN.
    Якоря генерируются для размера image_shape.

    Args:
        image_shape (tuple): Форма изображения (height, width, channels).
        config (dict): Словарь конфигурации.

    Returns:
        tuple: (anchors_by_level, anchors_info_by_level)
           anchors_by_level (dict): Словарь {level_name: np.ndarray(shape=(TotalAnchorsOnLevel, 4)), ...}
                                    координаты [xmin, ymin, xmax, ymax] в пикселях (плоский список якорей).
           anchors_info_by_level (dict): Словарь {level_name: list_of_dicts, ...} с доп инфо по каждому якорю
                                         в том же плоском порядке.
    """
    img_height, img_width = image_shape[:2]
    # anchor_scales: Словарь {P3: [[w_norm,h_norm], ...], ...} - нормализованные [0,1] WxH пары
    # anchor_ratios: Словарь {P3: [ratios], ...}
    # num_anchors_per_level: Словарь {P3: num, ...} - количество якорей на ячейку = scales * ratios
    anchor_scales_config = config.get('anchor_scales')
    anchor_ratios_config = config.get('anchor_ratios')
    num_anchors_per_level_config = config.get('num_anchors_per_level')

    # Стандартные шаги для FPN P3, P4, P5
    fpn_strides = {'P3': 8, 'P4': 16, 'P5': 32}


    if not anchor_scales_config or not anchor_ratios_config or not num_anchors_per_level_config:
         logger.error("Отсутствуют ключи конфигурации якорей ('anchor_scales', 'anchor_ratios' или 'num_anchors_per_level'). Невозможно выполнить генерацию якорей.")
         return {}, {}


    anchors_by_level = {}
    anchors_info_by_level = {}

    for level_name, stride in fpn_strides.items():
        # Проверяем наличие конфигурации для текущего уровня
        if level_name not in anchor_scales_config or level_name not in anchor_ratios_config or level_name not in num_anchors_per_level_config:
             # logger.warning(f"Конфигурация якорей отсутствует для уровня FPN {level_name}. Пропускаем генерацию якорей для этого уровня.")
             continue

        scales_wh_norm = np.array(anchor_scales_config[level_name]) # (K, 2) нормализованные [w, h]
        ratios = np.array(anchor_ratios_config[level_name]) # (R,)
        expected_num_anchors_in_cell = num_anchors_per_level_config[level_name] # Ожидаемое K * R

        # Проверка соответствия конфига: K * R должно быть равно num_anchors_per_level
        num_scales = scales_wh_norm.shape[0]
        num_ratios = ratios.shape[0]
        actual_anchors_in_cell = num_scales * num_ratios

        if actual_anchors_in_cell != expected_num_anchors_in_cell:
             logger.error(f"Несоответствие в конфиге для уровня {level_name}: num_scales ({num_scales}) * num_ratios ({num_ratios}) = {actual_anchors_in_cell}, но num_anchors_per_level указан как {expected_num_anchors_in_cell}. Проверьте конфиг якорей! Используется фактическое значение {actual_anchors_in_cell}.")
             # Продолжаем с фактическим количеством из scales * ratios
             num_anchors_this_level = actual_anchors_in_cell
        else:
             num_anchors_this_level = expected_num_anchors_in_cell

        if num_anchors_this_level == 0:
             logger.warning(f"Нет scales или ratios для уровня {level_name}. Пропускаем генерацию.")
             continue


        level_anchors_pascal_voc = []
        level_anchors_info_list = [] # Список словарей для информации о якорях


        # Размеры карты признаков для этого уровня FPN
        featmap_height = math.ceil(img_height / stride)
        featmap_width = math.ceil(img_width / stride)

        # logger.debug(f"Уровень FPN {level_name} (шаг {stride}): размер карты признаков {featmap_height}x{featmap_width}. Ожидается {num_anchors_this_level} якорей на ячейку.")


        # Генерируем якоря для каждой ячейки в карте признаков
        for r in range(featmap_height):
            for c in range(featmap_width):
                # Центр ячейки в координатах исходного изображения (пиксели)
                center_x_pixel = (c + 0.5) * stride
                center_y_pixel = (r + 0.5) * stride

                anchor_in_cell_counter = 0 # Счетчик якорей внутри текущей ячейки

                # Генерируем якоря для этой ячейки, комбинируя scales и ratios
                for scale_idx, (base_w_norm, base_h_norm) in enumerate(scales_wh_norm):
                    base_w_pixel = base_w_norm * img_width
                    base_h_pixel = base_h_norm * img_height

                    for ratio_idx, ratio in enumerate(ratios):
                        # Применяем соотношение сторон, сохраняя площадь базового якоря
                        # h' = sqrt(Area / r), w' = h' * r
                        base_area = base_w_pixel * base_h_pixel
                        epsilon = 1e-6 # Маленькое число для избежания деления/логарифма от нуля

                        if base_area < epsilon or ratio < epsilon: # Обработка очень маленькой площади или соотношения
                             # Если площадь или отношение близки к нулю, используем базовые размеры с небольшим множителем отношения (альтернатива сохранению площади)
                             anchor_w_pixel = base_w_pixel * math.sqrt(max(ratio, epsilon)) # Используем max с epsilon для ratio
                             anchor_h_pixel = base_h_pixel / math.sqrt(max(ratio, epsilon))
                             # logger.debug(f"Низкая площадь или отношение при расчете якоря на уровне {level_name}, ячейка ({r},{c}), масштаб {scale_idx}, отношение {ratio_idx}. Исходная площадь {base_area:.4f}, отношение {ratio:.4f}. Используются упрощенные размеры: w={anchor_w_pixel:.2f}, h={anchor_h_pixel:.2f}")
                        else:
                             try:
                                 h_prime = math.sqrt(base_area / ratio)
                                 w_prime = h_prime * ratio
                                 anchor_w_pixel = w_prime
                                 anchor_h_pixel = h_prime
                             except ValueError as e: # Например, если ratio стало отрицательным (хотя и маловероятно)
                                 logger.error(f"Математическая ошибка ({e}) при расчете якоря на уровне {level_name}, ячейка ({r},{c}), масштаб {scale_idx}, отношение {ratio_idx}. Исходная площадь {base_area}, отношение {ratio}. Используются базовые размеры.", exc_info=True)
                                 anchor_w_pixel = base_w_pixel # Возврат к базовым
                                 anchor_h_pixel = base_h_pixel


                        # Координаты якоря [xmin, ymin, xmax, ymax] в пикселях
                        xmin = center_x_pixel - anchor_w_pixel / 2.0
                        ymin = center_y_pixel - anchor_h_pixel / 2.0
                        xmax = center_x_pixel + anchor_w_pixel / 2.0
                        ymax = center_y_pixel + anchor_h_pixel / 2.0

                        # Сохраняем якорь и информацию о нем
                        level_anchors_pascal_voc.append([xmin, ymin, xmax, ymax])
                        # Информация о якоре для отладки
                        level_anchors_info_list.append({
                            'level': level_name,
                            'cell_yx': (r, c), # Строка, Колонка в карте признаков
                            'anchor_in_cell_idx': anchor_in_cell_counter, # Индекс якоря внутри ячейки (от 0 до num_anchors_this_level-1)
                            'scale_idx': scale_idx,
                            'ratio_idx': ratio_idx,
                            'base_wh_norm': (float(base_w_norm), float(base_h_norm)), # Нормализованные WxH базового якоря (конвертируем в float для надежности)
                            'ratio': float(ratio), # Примененное отношение (конвертируем в float)
                            'bbox_pixel_raw': [float(xmin), float(ymin), float(xmax), float(ymax)] # Пиксельные координаты якоря (конвертируем в float)
                            # 'type', 'iou', 'assigned_gt_idx', 'encoded_box', 'class_id' будут добавлены позже в assign_gt_to_anchors
                        })
                        anchor_in_cell_counter += 1

                # Проверка: количество сгенерированных якорей в ячейке должно совпадать с ожидаемым
                if anchor_in_cell_counter != num_anchors_this_level:
                     logger.warning(f"Несоответствие фактического ({anchor_in_cell_counter}) и ожидаемого ({num_anchors_this_level}) количества якорей в ячейке ({r},{c}) уровня {level_name}. Проверьте логику генерации и конфиг.")


        # Преобразуем список координат в numpy массив и сохраняем
        # reshape(-1, 4) сделает плоский список всех якорей для этого уровня
        anchors_by_level[level_name] = np.array(level_anchors_pascal_voc, dtype=np.float32).reshape(-1, 4)
        anchors_info_by_level[level_name] = level_anchors_info_list
        # logger.debug(f"Сгенерировано {len(level_anchors_info_list)} якорей для уровня FPN {level_name}.")


    return anchors_by_level, anchors_info_by_level


# --- Назначение Ground Truth якорям ---

def assign_gt_to_anchors(image_shape, gt_objects, anchors_by_level, anchors_info_by_level, config):
    """
    Назначает ground truth объекты якорям на основе IoU и масштаба.
    В этой реализации якорь назначается положительным ТОЛЬКО одному GT объекту, с которым у него наибольший IoU > positive_threshold.
    Якорь становится игнорируемым, если он не назначен положительным НИ ОДНОМУ GT, но имеет IoU >= ignore_threshold хотя бы с ОДНИМ GT.
    Все остальные - отрицательные.
    Включает логику "гарантированного назначения" лучшего якоря для каждого GT, если не нашлось positive по порогу.

    Args:
        image_shape (tuple): Форма изображения (height, width, channels) - размер аугментированного изображения.
        gt_objects (list): Список словарей GT объектов {'bbox': [x1, y1, x2, y4], 'class': ..., 'class_id': ...}.
                           Координаты в пикселях аугментированного изображения.
        anchors_by_level (dict): Словарь numpy массивов якорей по уровням FPN (плоский список).
        anchors_info_by_level (dict): Словарь списков словарей с доп инфо по якорям (плоский список).
        config (dict): Словарь конфигурации.

    Returns:
        dict: Словарь с детальной информацией о назначении для каждого якоря,
              сгруппированный по уровням FPN. Формат:
              {
                  'P3': [{'type': ..., 'iou': ..., 'assigned_gt_idx': ..., 'encoded_box': ..., 'class_id': ..., **anchor_info}, ...],
                  ...
              }
              Включает информацию для всех якорей (positive, ignored, negative).
              'type' = 1 (positive), -1 (ignored), 0 (negative).
    """
    img_height, img_width = image_shape[:2]
    num_classes = config.get('num_classes')
    # Нормализованные диапазоны площади GT объектов [0, 1] для назначения на уровни FPN
    fpn_gt_assignment_area_ranges_norm = config.get('fpn_gt_assignment_area_ranges')
    anchor_positive_iou_threshold = config.get('anchor_positive_iou_threshold')
    anchor_ignore_iou_threshold = config.get('anchor_ignore_iou_threshold')
    # fpn_strides = {'P3': 8, 'P4': 16, 'P5': 32} # Не нужны здесь


    if num_classes is None or not fpn_gt_assignment_area_ranges_norm or anchor_positive_iou_threshold is None or anchor_ignore_iou_threshold is None:
         logger.error("Отсутствуют необходимые ключи конфигурации для назначения GT. Невозможно выполнить назначение.")
         # Возвращаем структуру с негативными якорями
         assignment_info_by_level = {}
         for level_name, info_list in anchors_info_by_level.items():
              assignment_info_by_level[level_name] = []
              for anchor_info in info_list:
                  assignment_info_by_level[level_name].append({
                      'type': NEGATIVE_ANCHOR, 'iou': 0.0, 'assigned_gt_idx': -1,
                      'encoded_box': [0.0] * 4, 'class_id': -1, **anchor_info
                  })
         return assignment_info_by_level


    # Инициализируем детальную информацию о назначении для всех якорей
    # Копируем информацию о якорях и добавляем поля назначения
    # Все якоря изначально считаются отрицательными с IoU=0
    assignment_info_by_level = {}
    # Также создадим массив для хранения максимального IoU каждого якоря со *всеми* GT
    # Это нужно для определения игнорируемых якорей на втором проходе
    max_iou_with_any_gt_by_level = {}

    for level_name, info_list in anchors_info_by_level.items():
        assignment_info_by_level[level_name] = []
        for anchor_info in info_list:
             # Инициализируем каждое назначение как отрицательное
             assignment_info_by_level[level_name].append({
                 'type': NEGATIVE_ANCHOR, # 0:negative, 1:positive, -1:ignored
                 'iou': 0.0,              # Максимальный IoU с назначенным GT (для positive) ИЛИ с лучшим GT (для forced positive)
                 'assigned_gt_idx': -1,   # Индекс назначенного GT (-1 если нет)
                 'encoded_box': [0.0] * 4,# Закодированные дельты (нули для негативных)
                 'class_id': -1,          # ID класса назначенного GT (-1 если нет)
                 **anchor_info           # Копируем базовую информацию о якоре (bbox_pixel_raw, level, cell_yx, ...)
             })
        # Инициализируем массив максимальных IoU для этого уровня
        max_iou_with_any_gt_by_level[level_name] = np.zeros(len(info_list), dtype=np.float32)


    # --- Первое назначение: Определяем положительные якоря (включая "гарантированное назначение") ---
    # Проходим по каждому GT объекту
    gt_objects_assigned_to_level = [] # Список GT, которые попали в диапазоны площадей

    for gt_idx, gt_obj in enumerate(gt_objects):
        gt_bbox_pascal = gt_obj['bbox']
        gt_class_id = gt_obj['class_id']

        # Рассчитываем нормализованную площадь GT для назначения на уровень FPN
        xmin, ymin, xmax, ymax = gt_bbox_pascal
        gt_area_pixel = (xmax - xmin) * (ymax - ymin)
        # Пропускаем GT с нулевой площадью (уже должны быть отфильтрованы ранее)
        if gt_area_pixel <= 1e-6:
             continue

        gt_area_norm = gt_area_pixel / (img_width * img_height) # Нормализованная площадь [0, 1]

        # Определяем назначенный уровень FPN по диапазонам площади
        assigned_level_for_gt = None
        # Проходим по диапазонам площадей для каждого уровня FPN (P3, P4, P5 в порядке их шага)
        # Предполагаем, что fpn_gt_assignment_area_ranges_norm отсортирован по уровням (P3, P4, P5)
        level_names_ordered = list(anchors_by_level.keys()) # Берем имена уровней из якорей
        for level_idx, area_range in enumerate(fpn_gt_assignment_area_ranges_norm):
             if level_idx >= len(level_names_ordered):
                 logger.warning(f"Диапазон площадей {area_range} в конфиге выходит за пределы количества уровней FPN ({len(level_names_ordered)}). Проверьте конфиг.")
                 break
             level_name = level_names_ordered[level_idx]
             min_area, max_area = area_range

             # Обработка бесконечности из YAML
             min_area_val = float('-inf') if isinstance(min_area, str) and min_area.lower() == 'inf' else min_area
             max_area_val = float('inf') if isinstance(max_area, str) and max_area.lower() == 'inf' else max_area

             if gt_area_norm >= min_area_val and gt_area_norm < max_area_val:
                 assigned_level_for_gt = level_name
                 break # Нашли соответствующий уровень

        # Если GT не попал ни в один диапазон, он не будет назначен позитивно
        if assigned_level_for_gt is None:
             # logger.debug(f"GT object {gt_idx} (класс: {gt_obj['class']}, рамка: {gt_bbox_pascal}, норм.площадь: {gt_area_norm:.4f}) не попал ни в один диапазон площадей FPN из конфига. Он не будет назначен ни на один якорь как positive.")
             continue # Переходим к следующему GT

        # Добавляем GT в список тех, которые назначены на какой-либо уровень
        gt_objects_assigned_to_level.append({'gt_idx': gt_idx, 'level': assigned_level_for_gt})

        # Получаем якоря и информацию о назначении для этого уровня
        level_anchors = anchors_by_level.get(assigned_level_for_gt) # numpy array [xmin, ymin, xmax, ymax]
        level_assignment_info = assignment_info_by_level.get(assigned_level_for_gt) # list of dicts
        level_max_iou_any_gt = max_iou_with_any_gt_by_level.get(assigned_level_for_gt) # numpy array

        if level_anchors is None or level_assignment_info is None or level_max_iou_any_gt is None:
             logger.warning(f"Данные якорей отсутствуют для назначенного уровня FPN {assigned_level_for_gt} для GT {gt_idx}. Пропускаем назначение для этого GT.")
             continue

        # Рассчитываем IoU между этим GT и всеми якорями на его назначенном уровне FPN
        # Это может быть ресурсоемко, но для Data Loader'а это приемлемо
        ious_with_this_gt = np.array([calculate_iou(gt_bbox_pascal, anchor_bbox) for anchor_bbox in level_anchors])

        # Обновляем максимальное IoU для каждого якоря с любым GT
        # Этот массив нужен будет для определения игнорируемых на втором проходе
        np.maximum(level_max_iou_any_gt, ious_with_this_gt, out=level_max_iou_any_gt)


        # --- Назначаем положительные якоря на основе IoU threshold ---
        # Проходим по якорям на этом уровне FPN
        gt_bbox_cxcywh = convert_pascal_voc_to_cxcywh(gt_bbox_pascal)
        found_positive_for_this_gt_by_threshold = False # Флаг для "гарантированного назначения"

        for anchor_idx, iou in enumerate(ious_with_this_gt):
            # Если этот якорь имеет достаточно высокий IoU с ЭТИМ GT
            if iou >= anchor_positive_iou_threshold:
                # Этот якорь - кандидат на положительный для ЭТОГО GT.
                # Если он еще не был назначен положительным другому GT с бОльшим IoU, назначаем его этому GT.
                # Проверяем, если текущий якорь НЕ назначен положительным ИЛИ назначен положительным другому GT с меньшим IoU.
                current_assign_info = level_assignment_info[anchor_idx]
                is_current_positive = (current_assign_info['type'] == POSITIVE_ANCHOR)
                is_assigned_to_another_gt = (current_assign_info['assigned_gt_idx'] != -1 and current_assign_info['assigned_gt_idx'] != gt_idx)

                if not is_current_positive or (is_assigned_to_another_gt and iou > current_assign_info['iou']):
                    anchor_bbox_pascal = level_anchors[anchor_idx] # Координаты якоря
                    anchor_bbox_cxcywh = convert_pascal_voc_to_cxcywh(anchor_bbox_pascal)

                    current_assign_info['type'] = POSITIVE_ANCHOR
                    current_assign_info['iou'] = iou # Сохраняем IoU с назначенным GT
                    current_assign_info['assigned_gt_idx'] = gt_idx
                    current_assign_info['class_id'] = gt_class_id
                    current_assign_info['encoded_box'] = encode_box(anchor_bbox_cxcywh, gt_bbox_cxcywh)
                    found_positive_for_this_gt_by_threshold = True # Устанавливаем флаг, т.к. нашли positive по порогу


        # --- "Гарантированное назначение" (Best positive anchor) для ЭТОГО GT ---
        # Если после прохода по всем якорям на назначенном уровне, ЭТОТ GT не был назначен
        # ни одному якорю как positive по порогу (iou >= pos_thresh):
        if not found_positive_for_this_gt_by_threshold:
             # Находим якорь с максимальным IoU для ЭТОГО конкретного GT
             if len(ious_with_this_gt) > 0:
                  best_iou_for_this_gt = np.max(ious_with_this_gt)
                  best_iou_anchor_idx = np.argmax(ious_with_this_gt)

                  # Если лучший IoU для ЭТОГО GT >= ignore_threshold, принудительно назначаем этот якорь как positive
                  # НО ТОЛЬКО если этот якорь еще НЕ назначен positive другому GT с бОльшим IoU.
                  # Если он уже positive для другого GT с меньшим IoU, мы его перезаписываем.
                  # Если он уже positive для другого GT с бОльшим IoU, мы его не трогаем (приоритет выше).
                  # Если он NEGATIVE или IGNORED, мы его делаем POSITIVE.
                  if best_iou_for_this_gt >= anchor_ignore_iou_threshold:
                       forced_anchor_assign_info = level_assignment_info[best_iou_anchor_idx]
                       is_assigned_positive_to_another_gt = (forced_anchor_assign_info['type'] == POSITIVE_ANCHOR and forced_anchor_assign_info['assigned_gt_idx'] != gt_idx)

                       if not is_assigned_positive_to_another_gt or (is_assigned_positive_to_another_gt and best_iou_for_this_gt > forced_anchor_assign_info['iou']):
                           # Если он не positive вообще, или positive другому GT, но с меньшим IoU
                           # logger.debug(f"GT {gt_idx} (класс: {gt_obj['class']}) не имеет positive якорей по порогу на уровне {assigned_level_for_gt}. Принудительное назначение якоря {best_iou_anchor_idx} (IoU {best_iou_for_this_gt:.2f}) как positive.")
                           anchor_bbox_pascal = level_anchors[best_iou_anchor_idx] # Координаты якоря
                           anchor_bbox_cxcywh = convert_pascal_voc_to_cxcywh(anchor_bbox_pascal)

                           forced_anchor_assign_info['type'] = POSITIVE_ANCHOR
                           forced_anchor_assign_info['iou'] = best_iou_for_this_gt # Сохраняем фактический лучший IoU
                           forced_anchor_assign_info['assigned_gt_idx'] = gt_idx
                           forced_anchor_assign_info['class_id'] = gt_class_id
                           forced_anchor_assign_info['encoded_box'] = encode_box(anchor_bbox_cxcywh, gt_bbox_cxcywh)

                       # else:
                           # logger.debug(f"GT {gt_idx} (класс: {gt_obj['class']}) не имеет positive якорей по порогу, но лучший якорь ({best_iou_anchor_idx}, IoU {best_iou_for_this_gt:.2f}) уже назначен positive другому GT с бОльшим IoU ({forced_anchor_assign_info['iou']:.2f}). Принудительное назначение не выполнено.")


             # else:
                  # logger.debug(f"Нет якорей на уровне {assigned_level_for_gt} для GT {gt_idx}. Пропускаем принудительное назначение.")


    # --- Второй проход: Определяем игнорируемые якоря ---
    # Проходим по всем якорям на всех уровнях еще раз.
    # Якорь становится игнорируемым, если его тип остался NEGATIVE,
    # но его МАКСИМАЛЬНЫЙ IoU со ВСЕМИ GT (хранится в max_iou_with_any_gt_by_level)
    # >= ignore_threshold.
    # Если якорь стал POSITIVE на первом проходе, он остается POSITIVE.

    total_positive = 0
    total_ignored = 0
    total_negative = 0

    for level_name, assign_info_list in assignment_info_by_level.items():
        level_max_iou_any_gt = max_iou_with_any_gt_by_level.get(level_name) # Получаем соответствующий массив макс IoU

        if level_max_iou_any_gt is None or len(level_max_iou_any_gt) != len(assign_info_list):
             logger.error(f"Ошибка соответствия размеров массивов макс IoU для уровня {level_name}. Пропускаем второй проход для этого уровня.")
             # Просто посчитаем якоря по текущему типу
             total_positive += sum(1 for a in assign_info_list if a['type'] == POSITIVE_ANCHOR)
             total_ignored += sum(1 for a in assign_info_list if a['type'] == IGNORED_ANCHOR)
             total_negative += sum(1 for a in assign_info_list if a['type'] == NEGATIVE_ANCHOR)
             continue


        for anchor_idx, anchor_assign_info in enumerate(assign_info_list):
            if anchor_assign_info['type'] == NEGATIVE_ANCHOR:
                 # Если якорь остался отрицательным, но его максимальный IoU со *всеми* GT
                 # (не только с тем, который пытались назначить positive) >= ignore_threshold,
                 # он должен стать игнорируемым.
                 max_iou_this_anchor = level_max_iou_any_gt[anchor_idx]
                 if max_iou_this_anchor >= anchor_ignore_iou_threshold:
                      anchor_assign_info['type'] = IGNORED_ANCHOR
                      # В информацию о назначении для игнорируемых якорей сохраняем max_iou со *всеми* GT
                      # Это может быть полезно для отладки.
                      # anchor_assign_info['iou'] = max_iou_this_anchor # Уже сохранялось на первом проходе
                      total_ignored += 1
                 else:
                      total_negative += 1
            elif anchor_assign_info['type'] == POSITIVE_ANCHOR:
                 total_positive += 1
            # Если type == IGNORED_ANCHOR (присвоен на первом проходе из-за guaranteed assignment),
            # он остается игнорируемым.


    # Логирование итогов назначения
    logger.debug(f"Финализировано назначение: Положительных: {total_positive}, Игнорируемых: {total_ignored}, Отрицательных: {total_negative}.")

    # Проверка: если есть GT объекты, но нет положительных якорей
    if total_positive == 0 and len(gt_objects) > 0:
         # Фильтрация GT, которые не попали ни в один диапазон, чтобы не считать их отсутствие проблемой
         if len(gt_objects_assigned_to_level) > 0:
              logger.warning(f"Назначено 0 положительных якорей для {len(gt_objects_assigned_to_level)} GT объектов, назначенных на уровни FPN! Проверьте конфиг якорей/IoU/диапазоны площадей или логику.")

    return assignment_info_by_level


# --- Сборка y_true тензоров ---

def format_assignment_info_to_y_true(image_shape, assignment_info_by_level, config):
     """
     Форматирует детальную информацию о назначении якорей в тензоры y_true
     для использования в функции потерь.

     Args:
         image_shape (tuple): Форма изображения (height, width, channels) - размер аугментированного изображения.
         assignment_info_by_level (dict): Выход функции assign_gt_to_anchors.
         config (dict): Словарь конфигурации.

     Returns:
         tuple: Кортеж TF тензоров y_true в порядке:
                (y_true_reg_P3, y_true_cls_P3, y_true_mask_P3,
                 y_true_reg_P4, y_true_cls_P4, y_true_mask_P4,
                 y_true_reg_P5, y_true_cls_P5, y_true_mask_P5)
                для каждого уровня FPN.
                Каждый тензор имеет shape (H, W, num_anchors_per_level, tasks_dim).
                tasks_dim = 4 для регрессии, num_classes для классификации, 1 для маски.
     """
     img_height, img_width = image_shape[:2]
     num_classes = config.get('num_classes')
     fpn_strides = {'P3': 8, 'P4': 16, 'P5': 32}
     num_anchors_per_level_config = config.get('num_anchors_per_level')

     if num_classes is None or not num_anchors_per_level_config:
          logger.error("Отсутствуют необходимые ключи конфигурации для форматирования y_true ('num_classes', 'num_anchors_per_level'). Невозможно сформатировать y_true.")
          # Вернуть заглушку: 9 нулевых тензоров. Размеры карт признаков неизвестны без input_shape.
          # Используем фиктивные размеры 1x1 и num_anchors=1.
          dummy_tensors = []
          for level_name in ['P3', 'P4', 'P5']:
              dummy_tensors.append(tf.zeros((1, 1, 1, 4), dtype=tf.float32)) # Reg
              dummy_tensors.append(tf.zeros((1, 1, 1, num_classes if num_classes is not None else 2), dtype=tf.float32)) # Cls (используем 2 как заглушку)
              dummy_tensors.append(tf.zeros((1, 1, 1), dtype=tf.float32)) # Mask (используем 1 для последней размерности)
          return tuple(dummy_tensors)


     y_true_tensors_list = [] # Список тензоров в порядке reg, cls, mask ... P3, P4, P5

     level_names_ordered = ['P3', 'P4', 'P5'] # Убедимся в правильном порядке
     for level_name in level_names_ordered:
         if level_name not in assignment_info_by_level or level_name not in num_anchors_per_level_config:
             # logger.warning(f"Информация о назначении или конфиг якорей отсутствует для уровня {level_name}. Генерируем нулевые y_true тензоры.")
             # Добавляем нулевые тензоры соответствующей формы
             stride = fpn_strides.get(level_name, 1) # Default stride 1
             h = math.ceil(img_height / stride)
             w = math.ceil(img_width / stride)
             num_anchors = num_anchors_per_level_config.get(level_name, 1) # Default 1
             y_true_tensors_list.append(tf.zeros((h, w, num_anchors, 4), dtype=tf.float32)) # Reg
             y_true_tensors_list.append(tf.zeros((h, w, num_anchors, num_classes), dtype=tf.float32)) # Cls
             y_true_tensors_list.append(tf.zeros((h, w, num_anchors), dtype=tf.float32)) # Mask
             continue


         level_assignment_info_list = assignment_info_by_level[level_name]
         stride = fpn_strides[level_name]
         h = math.ceil(img_height / stride)
         w = math.ceil(img_width / stride)
         num_anchors = num_anchors_per_level_config[level_name] # Общее количество якорей на ячейку для этого уровня

         # Проверка соответствия количества якорей
         expected_total_anchors_on_level = h * w * num_anchors
         if len(level_assignment_info_list) != expected_total_anchors_on_level:
             logger.error(f"Несоответствие количества якорей в assignment_info ({len(level_assignment_info_list)}) и ожидаемого ({expected_total_anchors_on_level}) для уровня {level_name}. Проверьте логику генерации и назначения!")
             # Попытаемся продолжить, но это может привести к ошибкам индексации
             # Заполним только те якоря, которые есть в списке
             pass # Продолжаем с тем, что есть


         # Инициализируем numpy массивы для y_true этого уровня
         y_true_reg_np = np.zeros((h, w, num_anchors, 4), dtype=np.float32)
         y_true_cls_np = np.zeros((h, w, num_anchors, num_classes), dtype=np.float32)
         # Маска с float значениями типа якоря: 1.0 для positive, 0.0 для negative, -1.0 для ignored
         y_true_mask_np = np.full((h, w, num_anchors), float(NEGATIVE_ANCHOR), dtype=np.float32)


         # Заполняем массивы на основе информации о назначении
         # Используем 'cell_yx' и 'anchor_in_cell_idx' из info_list для заполнения
         for anchor_assign_info in level_assignment_info_list:
             r, c = anchor_assign_info.get('cell_yx', (-1, -1))
             anchor_in_cell_idx = anchor_assign_info.get('anchor_in_cell_idx', -1)
             assign_type = anchor_assign_info.get('type', NEGATIVE_ANCHOR)

             # Проверка границ
             if r < 0 or r >= h or c < 0 or c >= w or anchor_in_cell_idx < 0 or anchor_in_cell_idx >= num_anchors:
                  # logger.error(f"Некорректные индексы ({r},{c},{anchor_in_cell_idx}) для якоря на уровне {level_name} с размером карты признаков {h}x{w} и якорей на ячейку {num_anchors}. Пропускаем форматирование для этого якоря.")
                  continue # Пропускаем якорь с некорректными индексами

             # Заполняем маску
             y_true_mask_np[r, c, anchor_in_cell_idx] = float(assign_type)

             if assign_type == POSITIVE_ANCHOR:
                  class_id = anchor_assign_info.get('class_id', -1)
                  encoded_box = anchor_assign_info.get('encoded_box', [0.0] * 4)

                  # Целевые значения регрессии
                  y_true_reg_np[r, c, anchor_in_cell_idx, :] = np.array(encoded_box, dtype=np.float32)

                  # Целевые значения классификации (one-hot)
                  if 0 <= class_id < num_classes:
                       y_true_cls_np[r, c, anchor_in_cell_idx, class_id] = 1.0
                  else:
                       # logger.warning(f"Некорректный class_id ({class_id}) для положительного якоря на уровне {level_name} ({r},{c},{anchor_in_cell_idx}). Ожидается 0-{num_classes-1}. Классификация установлена в ноль.")
                       # y_true_cls_np[r, c, anchor_in_cell_idx, :] уже нули при инициализации.
                       pass # Классификация уже нули для этого якоря


             # Для IGNORED_ANCHOR и NEGATIVE_ANCHOR:
             # y_true_reg_np остается нулями.
             # y_true_cls_np остается нулями (что представляет фон/отсутствие объекта).


         # Добавляем тензоры для этого уровня в список (конвертируем в TF тензоры)
         y_true_tensors_list.append(tf.constant(y_true_reg_np, dtype=tf.float32))
         y_true_tensors_list.append(tf.constant(y_true_cls_np, dtype=tf.float32))
         y_true_tensors_list.append(tf.constant(y_true_mask_np, dtype=tf.float32)) # Маска теперь float32

         # Логирование для проверки форм
         # logger.debug(f"Уровень {level_name}: y_true_reg shape {y_true_reg_np.shape}, y_true_cls shape {y_true_cls_np.shape}, y_true_mask shape {y_true_mask_np.shape}")


     # Возвращаем как кортеж TF тензоров
     # logger.debug("Сформатирована информация о назначении в y_true тензоры.")
     return tuple(y_true_tensors_list)


# --- Главная функция загрузки и подготовки данных ---

# import tensorflow as tf # Уже импортирован
# import numpy as np # Уже импортирован
# import cv2 # Уже импортирован
# import xml.etree.ElementTree as ET # Уже импортирован
# import math # Уже импортирован
# import yaml # Уже импортирован
# from pathlib import Path # Уже импортирован
# import logging # Уже импортирован
# import random # Уже импортирован
# import albumentations as A # Уже импортирован

def load_and_prepare_image_and_targets(image_path_str, annotation_path_str, config_path_str, use_augmentation_flag, debug_mode=False, aug_seed=None):
    """
    Загружает, препроцессит изображение (ресайз+падинг), парсит аннотации,
    применяет аугментацию, генерирует якоря и назначает им GT.

    Args:
        image_path_str (str): Путь к файлу изображения.
        annotation_path_str (str): Путь к файлу аннотации (XML).
        config_path_str (str): Путь к файлу конфигурации.
        use_augmentation_flag (bool): Применять ли аугментацию.
        debug_mode (bool): Если True, возвращает расширенные данные для отладки (numpy/python).
                           Если False (для tf.data), возвращает TF тензоры.
        aug_seed (int, optional): Сид для аугментации. Если None, используется случайный сид NumPy.

    Returns:
        Если debug_mode is False:
            tuple: (normalized_image_tensor, y_true_tensors_tuple_tf)
        Если debug_mode is True:
            tuple: (image_padded_rgb_uint8, # np.ndarray, uint8 RGB после ресайза/падинга (до аугментации)
                    image_np_augmented_uint8, # np.ndarray, uint8 RGB после ресайза/падинга/аугментации
                    y_true_tensors_tuple_np, # Tuple из numpy массивов y_true
                    original_gt_objects, # list of dicts, GT до всех трансформаций (только парсинг)
                    scaled_shifted_gt_objects, # list of dicts, GT после ресайза/падинга (до аугментации)
                    augmented_gt_objects, # list of dicts, GT после аугментации и фильтрации Albumentations
                    assignment_info_by_level, # dict of lists of dicts, информация о назначении якорей
                    anchors_info_by_level_augmented_image # dict of list of dicts, информация о всех якорях
                   )
           В режиме debug_mode=True возвращаются numpy/python объекты.
           При debug_mode=False (для tf.data), возвращаются TF тензоры.
    """
    # Убедимся, что пути - строки
    if isinstance(image_path_str, tf.Tensor):
         image_path_str = image_path_str.numpy().decode('utf-8')
    if isinstance(annotation_path_str, tf.Tensor):
         annotation_path_str = annotation_path_str.numpy().decode('utf-8')
    if isinstance(config_path_str, tf.Tensor):
         config_path_str = config_path_str.numpy().decode('utf-8')

    # 1. Загрузка конфига
    config = load_config(config_path_str)
    if config is None:
        logger.error(f"Не удалось загрузить конфиг из {config_path_str}. Невозможно обработать данные.")
        raise ValueError(f"Не удалось загрузить конфиг: {config_path_str}")

    input_shape = config.get('input_shape')
    class_names = config.get('class_names')

    if input_shape is None or class_names is None:
         logger.error("Отсутствуют ключи 'input_shape' или 'class_names' в конфиге. Невозможно обработать данные.")
         raise ValueError("Отсутствуют ключи 'input_shape' или 'class_names' в конфиге.")

    target_h, target_w = input_shape[:2]

    # 2. Загрузка изображения
    image_bgr = cv2.imread(image_path_str)
    if image_bgr is None:
        logger.error(f"Не удалось загрузить изображение: {image_path_str}")
        raise ValueError(f"Не удалось загрузить изображение: {image_path_str}")

    original_h, original_w = image_bgr.shape[:2]

    # 3. Парсинг аннотаций (по исходным размерам)
    original_gt_objects = parse_pascal_voc_xml(annotation_path_str, original_w, original_h, class_names)
    # logger.debug(f"Парсинг завершен. Найдено {len(original_gt_objects)} объектов в оригинальных аннотациях.")

    # 4. Ресайз и падинг изображения до целевого input_shape (Letterboxing)
    # Сохраняем соотношение сторон. Добавляем падинг серым цветом (128).
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Ресайз изображения
    image_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA) # INTER_AREA для уменьшения

    # Создаем изображение с падингом до целевого размера
    padded_image_bgr = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    # Вычисляем смещение для вставки по центру
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded_image_bgr[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = image_resized

    image_padded_rgb = cv2.cvtColor(padded_image_bgr, cv2.COLOR_BGR2RGB) # Конвертируем в RGB для Albumentations

    # Сохраняем копиюpadded_image_rgb для возврата в debug_mode
    image_padded_rgb_uint8 = image_padded_rgb.copy()


    # 5. Масштабирование и смещение рамок GT под размер и положение padded_image_bgr
    scaled_shifted_gt_objects = []
    for gt_obj in original_gt_objects:
        xmin, ymin, xmax, ymax = gt_obj['bbox'] # Координаты в исходных пикселях
        # Масштабирование
        xmin_scaled = xmin * scale
        ymin_scaled = ymin * scale
        xmax_scaled = xmax * scale
        ymax_scaled = ymax * scale
        # Смещение
        xmin_shifted = xmin_scaled + pad_w
        ymin_shifted = ymin_scaled + pad_h
        xmax_shifted = xmax_scaled + pad_w
        ymax_shifted = ymax_scaled + pad_h


        # Проверка границ после масштабирования и смещения
        xmin_final = max(0.0, xmin_shifted)
        ymin_final = max(0.0, ymin_shifted)
        xmax_final = min(float(target_w - 1), xmax_shifted)
        ymax_final = min(float(target_h - 1), ymax_shifted)

        # Проверка на валидность после всех трансформаций до аугментации
        # Оставим только рамки с положительной площадью. Albumentations bbox_params
        # дополнительно отфильтрует по min_visibility и min_area.
        if xmax_final > xmin_final and ymax_final > ymin_final:
             scaled_shifted_gt_objects.append({
                 'bbox': [xmin_final, ymin_final, xmax_final, ymax_final],
                 'class': gt_obj['class'],
                 'class_id': gt_obj['class_id']
             })
        else:
             logger.debug(f"Пропущен GT объект {gt_obj['class']} после ресайза/падинга ({image_path_str}): [{xmin_final:.2f}, {ymin_final:.2f}, {xmax_final:.2f}, {ymax_final:.2f}]. Исходная рамка: {gt_obj['bbox']}")

    # logger.debug(f"После ресайза/падинга осталось {len(scaled_shifted_gt_objects)} валидных GT.")


    # 6. Аугментация
    image_np_augmented = image_padded_rgb # Начинаем с padded_image_rgb
    augmented_gt_objects = scaled_shifted_gt_objects # Начинаем с масштабированных/смещенных GT

    if use_augmentation_flag:
        try:
            # Передаем размер padded_image_rgb в get_detector_train_augmentations (он == input_shape)
            augs = get_detector_train_augmentations(image_np_augmented.shape[0], image_np_augmented.shape[1])
            if augs:
                 # Устанавливаем сид Albumentations через NumPy, если задан
                 if aug_seed is not None:
                      np.random.seed(aug_seed) # Устанавливаем сид NumPy
                      # logger.debug(f"NumPy seed установлен в {aug_seed} для Albumentations для {image_path_str}")
                 else:
                      # Если сид не задан (для случайной аугментации), используем текущее состояние np.random.
                      # которое было инициализировано в начале скрипта if __name__ == '__main__':
                      pass # Ничего не делаем с сидом NumPy здесь, если aug_seed is None

                 # Подготавливаем данные для Albumentations
                 bboxes_for_alb = [obj['bbox'] for obj in scaled_shifted_gt_objects]
                 labels_for_alb = [obj['class'] for obj in scaled_shifted_gt_objects] # Albumentations требует строковые метки

                 # Применяем аугментацию
                 augmented = augs(image=image_np_augmented, bboxes=bboxes_for_alb, class_labels_for_albumentations=labels_for_alb)

                 # Albumentations возвращает uint8 RGB изображение того же размера, что и входное (input_shape)
                 image_np_augmented = augmented['image']
                 augmented_bboxes_alb = augmented['bboxes'] # Уже отфильтрованы Albumentations по min_area/min_visibility
                 augmented_labels_alb = augmented['class_labels_for_albumentations'] # Соответствуют augmented_bboxes_alb

                 # Конвертируем аугментированные рамки обратно в наш внутренний формат list of dicts
                 augmented_gt_objects = []
                 for bbox_alb, label_alb in zip(augmented_bboxes_alb, augmented_labels_alb):
                      if label_alb in class_names: # Проверка на всякий случай, хотя Albumentations сохраняет метки
                           class_id = class_names.index(label_alb)
                           augmented_gt_objects.append({
                               'bbox': list(bbox_alb), # Конвертируем в list
                               'class': label_alb,
                               'class_id': class_id
                           })
                      # else:
                            # logger.warning(f"Аугментированная рамка с неизвестной меткой '{label_alb}' получена от Albumentations для {image_path_str}. Пропускаем.")

            # else:
                 # logger.warning(f"get_detector_train_augmentations вернула None для {image_path_str}. Аугментация не применена.")

        except Exception as e:
            logger.error(f"Ошибка во время аугментации для {image_path_str}: {e}", exc_info=True) # Логируем traceback
            # Если аугментация не удалась, продолжаем с изображением после ресайза/падинга и scaled_shifted_gt_objects.
            # image_np_augmented уже установлен в image_padded_rgb
            # augmented_gt_objects уже установлен в scaled_shifted_objects
            # В случае ошибки аугментации, состояние np.random уже не важно для этого изображения.
            pass


    # logger.debug(f"После аугментации осталось {len(augmented_gt_objects)} валидных GT.")
    if len(augmented_gt_objects) == 0 and len(original_gt_objects) > 0:
         logger.warning(f"Все {len(original_gt_objects)} оригинальных GT объектов были потеряны после ресайза/падинга/аугментации для {image_path_str}.")


    # 7. Генерация якорей для АУГМЕНТИРОВАННОГО изображения (размер == input_shape)
    # Якоря генерируются на основе размера изображения ПОСЛЕ всех предварительных трансформаций (ресайз/падинг)
    # и аугментации. Этот размер должен быть input_shape из конфига.
    # Проверяем, что размер аугментированного изображения соответствует input_shape
    if image_np_augmented.shape[:2] != (target_h, target_w):
         logger.error(f"Размер аугментированного изображения {image_np_augmented.shape[:2]} не совпадает с целевым input_shape {target_h}x{target_w} для {image_path_str}. Проверьте пайплайн трансформаций! Генерация якорей будет выполнена для фактического размера.")
         # Продолжаем с фактическим размером аугментированного изображения
         # logger.warning(f"Генерация якорей будет выполнена для фактического размера {image_np_augmented.shape[:2]}.")


    anchors_by_level_augmented_image, anchors_info_by_level_augmented_image = generate_all_anchors(image_np_augmented.shape, config)
    # logger.debug(f"Сгенерированы якоря для аугментированного изображения размером {image_np_augmented.shape[:2]}.")


    # 8. Назначение GT якорям
    # Используем аугментированные GT объекты и якоря, сгенерированные для аугментированного изображения.
    assignment_info_by_level = assign_gt_to_anchors(image_np_augmented.shape, augmented_gt_objects,
                                                    anchors_by_level_augmented_image, anchors_info_by_level_augmented_image, config)
    # logger.debug("Завершено назначение GT якорям.")


    # 9. Форматирование назначений в y_true тензоры (NumPy массивы)
    # Эти массивы будут сконвертированы в TF тензоры при debug_mode=False.
    y_true_tensors_tuple_np = format_assignment_info_to_y_true(image_np_augmented.shape, assignment_info_by_level, config)
    # logger.debug(f"Сформатирована информация о назначении в y_true тензоры (NumPy).")


    # 10. Нормализация изображения
    # image_np_augmented - это uint8 RGB [0, 255] после аугментации
    image_np_augmented_float = image_np_augmented.astype(np.float32)

    image_normalization_method = config.get('image_normalization_method', 'none')
    if image_normalization_method == 'imagenet':
        # ImageNet means and stddevs for RGB channels (для диапазона [0, 255])
        mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) # RGB means
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)     # RGB stddevs
        normalized_image_np = (image_np_augmented_float - mean) / std
        # logger.debug("Изображение нормализовано по ImageNet.")
    elif image_normalization_method == 'divide_255':
        normalized_image_np = image_np_augmented_float / 255.0
        # logger.debug("Изображение нормализовано делением на 255.")
    else:
        # Не применяем нормализацию, просто убедимся, что тип float
        normalized_image_np = image_np_augmented_float
        # logger.debug("Нормализация изображения не применена.")

    # logger.debug(f"Обработка изображения завершена для {image_path_str}. Режим отладки: {debug_mode}.")

    # 11. Возврат данных в зависимости от debug_mode
    if debug_mode:
        # Возвращаем numpy/python объекты для отладки
        return (image_padded_rgb_uint8, # np.ndarray, uint8 RGB после ресайза/падинга (до аугментации)
                image_np_augmented, # np.ndarray, uint8 RGB после ресайза/падинга/аугментации (image_np_augmented теперь uint8)
                y_true_tensors_tuple_np, # Tuple из numpy массивов y_true
                original_gt_objects, # list of dicts, GT до всех трансформаций
                scaled_shifted_gt_objects, # list of dicts, GT после ресайза/падинга
                augmented_gt_objects, # list of dicts, GT после аугментации и фильтрации
                assignment_info_by_level, # dict of lists of dicts, информация о назначении якорей
                anchors_info_by_level_augmented_image # dict of list of dicts, информация о всех якорях
               )
    else:
        # Конвертируем y_true numpy массивы в TF тензоры и возвращаем их вместе с нормализованным TF тензором изображения
        y_true_tensors_tuple_tf = tuple(tf.constant(y, dtype=tf.float32) for y in y_true_tensors_tuple_np)
        normalized_image_tensor = tf.constant(normalized_image_np, dtype=tf.float32) # Используем normalized_image_np (float32)
        return (normalized_image_tensor, y_true_tensors_tuple_tf)


# --- TensorFlow wrapper function (для использования с tf.data.Dataset) ---
# Этот блок будет реализован позже.
# from typing import Tuple # Уже импортирован в комментариях

# def tf_wrapper_func(image_path_tensor: tf.Tensor, annotation_path_tensor: tf.Tensor, config_path_tensor: tf.Tensor, use_augmentation_tensor: tf.Tensor): # -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
#     """
#     Wrapper function for tf.data.Dataset.map().
#     Calls load_and_prepare_image_and_targets in TensorFlow graph.
#     """
#     # При использовании tf.py_function нужно явно указывать возвращаемые типы (Tout)
#     # и желательно формы (например, через tf.TensorSpec), но формы могут быть динамическими.
#     # Количество возвращаемых y_true тензоров (3 уровня * 3 тензора на уровень = 9).
#     # Типы y_true тензоров: Reg (float32), Cls (float32), Mask (float32).
#     # Тип изображения: float32 (после нормализации).
#     # Общее количество выходных тензоров = 1 (изображение) + 9 (y_true).
#     # Tout = [tf.float32] + [tf.float32] * 9 # Пример
#
#     # Конфиг и флаг аугментации должны быть переданы как тензоры
#     # Внутри py_function их нужно конвертировать обратно в нативные типы (.numpy().decode('utf-8') и .numpy())
#     # debug_mode всегда False для tf.data пайплайна
#     # aug_seed = None, т.к. случайность контролируется общим сидом Dataset shuffle/map или не контролируется вовсе
#
#     [image_tensor, *y_true_tensors_list] = tf.py_function(
#         func=load_and_prepare_image_and_targets,
#         inp=[image_path_tensor, annotation_path_tensor, config_path_tensor, use_augmentation_tensor, False, None],
#         Tout=[tf.float32] + [tf.float32] * 9 # 1 изображение + 9 y_true тензоров
#     )
#
#     # Важно: Установить формы для выходных тензоров, если они известны.
#     # Это критично для производительности и правильной работы модели.
#     # Форма изображения известна: [target_h, target_w, 3]
#     # image_tensor.set_shape(input_shape) # input_shape нужно получить из конфига
#     # Формы y_true тензоров зависят от stride FPN и num_anchors_per_level.
#     # H_level = ceil(target_h / stride), W_level = ceil(target_w / stride)
#     # Shapes: (H_level, W_level, num_anchors_level, dim)
#     # Нужны значения strides и num_anchors_per_level из конфига.
#     # Это можно сделать, передав эти значения как дополнительные входные тензоры в py_function,
#     # или загрузив конфиг внутри (менее эффективно).
#     # Пример (предполагает, что config и num_anchors_per_level_config доступны):
#     # level_names_ordered = ['P3', 'P4', 'P5']
#     # fpn_strides = {'P3': 8, 'P4': 16, 'P5': 32}
#     # for i, level_name in enumerate(level_names_ordered):
#     #     stride = fpn_strides[level_name]
#     #     h = math.ceil(target_h / stride)
#     #     w = math.ceil(target_w / stride)
#     #     num_anchors = num_anchors_per_level_config[level_name]
#     #     y_true_tensors_list[i*3+0].set_shape((h, w, num_anchors, 4)) # Reg
#     #     y_true_tensors_list[i*3+1].set_shape((h, w, num_anchors, config['num_classes'])) # Cls
#     #     y_true_tensors_list[i*3+2].set_shape((h, w, num_anchors)) # Mask (размерность 1 для маски не нужна)
#
#     # Возвращаем кортеж из изображения и кортежа y_true тензоров
#     # return image_tensor, tuple(y_true_tensors_list)


# --- Тестовый Блок (для отладки логики назначения GT) ---
if __name__ == '__main__':
    print("--- Запуск тестового блока data_loader_v3_standard.py ---")
    logger.setLevel(logging.DEBUG) # Включаем детальное логирование только для тестового блока

    # Указываем пути к конфигу и тестовым данным
    _current_script_dir = Path(__file__).parent.resolve()
    # Предполагаем, что скрипт находится в src/datasets
    _project_root_dir = _current_script_dir.parent.parent
    _config_path = _project_root_dir / "src" / "configs" / "detector_config_v3_standard.yaml"
    _graphs_output_dir = _project_root_dir / "graphs" / "problem_scenarios" # Папка для сохранения графиков проблемных сценариев
    _graphs_output_dir.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее нет


    # Указываем путь к папкам с изображениями и аннотациями для тренировочного набора
    # Берем пути из detector_config_v3_standard.yaml
    _config_data = load_config(str(_config_path))
    if _config_data is None:
        print(f"Ошибка загрузки конфига {_config_path}. Невозможно запустить тест.")
        exit()

    _dataset_base_path = _project_root_dir / _config_data.get('dataset_path', 'data/Detector_Dataset_Ready')
    _train_images_subdir = _config_data.get('train_images_subdir', 'train/images')
    _train_annotations_subdir = _config_data.get('train_annotations_subdir', 'train/Annotations')

    _train_images_dir = _dataset_base_path / _train_images_subdir
    _train_annotations_dir = _dataset_base_path / _train_annotations_subdir


    if not _train_images_dir.exists():
         print(f"Папка с изображениями не найдена: {_train_images_dir}. Невозможно запустить тест.")
         exit()
    if not _train_annotations_dir.exists():
         print(f"Папка с аннотациями не найдена: {_train_annotations_dir}. Невозможно запустить тест.")
         exit()

    # Собираем список пар путей к изображениям и аннотациям
    # Ищем изображения и формируем соответствующий путь к аннотации
    image_paths = sorted(list(_train_images_dir.glob("*.jpg")) + list(_train_images_dir.glob("*.jpeg")) + list(_train_images_dir.glob("*.png")))
    data_paths = []
    for img_path in image_paths:
        anno_path = _train_annotations_dir / (img_path.stem + ".xml")
        if anno_path.exists():
            data_paths.append((str(img_path), str(anno_path))) # Используем str для совместимости

    if not data_paths:
        print(f"Не найдено пар изображение/аннотация (.jpg/.jpeg/.png с соответствующим .xml) в папках: {_train_images_dir}, {_train_annotations_dir}. Невозможно запустить тест.")
        exit()

    print(f"Найдено {len(data_paths)} пар изображение/аннотация.")

    # --- Параметры тестового анализа ---
    # Количество изображений для анализа случайных аугментаций
    num_images_to_test = min(100, len(data_paths)) # Анализируем до 100 изображений для лучшей статистики

    # Порог для определения "проблемного сценария"
    # Если количество положительных якорей <= positive_anchor_problem_threshold при наличии GT объектов
    positive_anchor_problem_threshold = 1 # 0 или 1 положительный якорь - это проблема

    # Максимальное количество проблемных сценариев для логирования/сохранения сидов и ВИЗУАЛИЗАЦИИ
    max_problem_scenarios_to_visualize = 15 # Визуализируем до 15 проблемных случаев

    # Порог IoU для отображения негативных якорей на графике (чтобы не рисовать все)
    negative_anchor_viz_iou_threshold = 0.1


    print(f"\n--- Анализ {num_images_to_test} случайных аугментаций ---")

    # Перемешиваем пути, чтобы брать случайные изображения для анализа
    # Используем фиксированный сид для перемешивания путей, чтобы тест был воспроизводим
    np.random.seed(42) # Сид для перемешивания путей
    np.random.shuffle(data_paths)
    test_data_paths = data_paths[:num_images_to_test]

    problem_scenarios_found = []
    figures_to_show = [] # Список фигур matplotlib для отображения в конце


    # Инициализируем генератор случайных чисел NumPy для генерации СЛУЧАЙНЫХ сидов аугментации для КАЖДОГО изображения
    # Используем более надежный способ инициализации сида numpy
    rng = np.random.default_rng() # Предпочтительный способ в новых версиях numpy

    # НЕ фиксируем глобальный сид plot_utils здесь, чтобы цвета были разные для разных запусков теста,
    # ЕСЛИ ТОЛЬКО вы не захотите воспроизвести точную визуализацию для конкретного сида теста.
    # plot_utils.set_global_seed(SOME_PLOT_SEED)


    for i, (img_path_str, anno_path_str) in enumerate(test_data_paths):
        # Генерируем СЛУЧАЙНЫЙ сид для аугментации для КАЖДОГО изображения
        # Этот сид будет передан в load_and_prepare_image_and_targets и зафиксирует np.random.seed() внутри
        # для Albumentations для этого конкретного изображения.
        # Генерируем сид в безопасном диапазоне для np.random.seed(), который принимает uint32.
        # Максимальное значение для uint32 - 2**32 - 1.
        current_aug_seed = rng.integers(0, 2**32 - 1, dtype=np.int64) # Диапазон [0, 2**32 - 1]
        current_aug_seed = int(current_aug_seed) # Преобразуем в стандартный int

        # logger.info(f"Обработка изображения {i+1}/{num_images_to_test}: {Path(img_path_str).name} с сидом аугментации {current_aug_seed}")
        print(f"Обработка изображения {i+1}/{num_images_to_test}: {Path(img_path_str).name} (сид: {current_aug_seed})") # Более краткий вывод прогресса

        try:
            # Вызываем главную функцию в debug_mode=True
            # Возвращает numpy/python объекты
            (image_padded_rgb_uint8, # uint8 RGB после ресайза/падинга (до аугментации)
             image_np_augmented_uint8, # uint8 RGB после ресайза/падинга/аугментации
             y_true_tuple_np, # Tuple из numpy массивов y_true
             original_gt_objects, # list of dicts до ресайза/аугментации (из XML)
             scaled_shifted_gt_objects, # list of dicts после ресайза/падинга
             augmented_gt_objects, # list of dicts после аугментации и фильтрации
             assignment_info_by_level, # dict of lists of dicts
             anchors_info_by_level_augmented_image # dict of list of dicts (информация о всех якорях)
            ) = load_and_prepare_image_and_targets(
                img_path_str,
                anno_path_str,
                str(_config_path), # Передаем путь к конфигу как строку
                use_augmentation_flag=True, # Включаем аугментацию
                debug_mode=True, # Включаем отладочный режим
                aug_seed=current_aug_seed # Фиксируем сид аугментации для этого вызова
            )

            # --- Анализ результатов назначения ---
            total_pos_anchors = 0
            total_ign_anchors = 0
            total_neg_anchors = 0
            total_anchors = 0
            anchor_counts_by_level = {} # Для логирования по уровням FPN

            for level_name, assign_info_list in assignment_info_by_level.items():
                 level_pos = sum(1 for a in assign_info_list if a['type'] == POSITIVE_ANCHOR)
                 level_ign = sum(1 for a in assign_info_list if a['type'] == IGNORED_ANCHOR)
                 level_neg = sum(1 for a in assign_info_list if a['type'] == NEGATIVE_ANCHOR)
                 level_total = len(assign_info_list)
                 total_pos_anchors += level_pos
                 total_ign_anchors += level_ign
                 total_neg_anchors += level_neg
                 total_anchors += level_total
                 anchor_counts_by_level[level_name] = {'positive': level_pos, 'ignored': level_ign, 'negative': level_neg, 'total': level_total}

            # Проверяем, были ли GT объекты, которые прошли до этапа назначения (после аугментации)
            has_augmented_gt_objects = len(augmented_gt_objects) > 0

            if has_augmented_gt_objects and total_pos_anchors <= positive_anchor_problem_threshold:
                logger.error(f"  !!! ОБНАРУЖЕН ПРОБЛЕМНЫЙ СЦЕНАРИЙ ({Path(img_path_str).name}) !!!")
                logger.error(f"  Сид аугментации: {current_aug_seed}")
                logger.error(f"  Количество аугментированных GT: {len(augmented_gt_objects)}")
                logger.error(f"  ИТОГО назначено положительных якорей: {total_pos_anchors}")
                for level_name, counts in anchor_counts_by_level.items():
                    logger.error(f"    {level_name}: Положительных: {counts['positive']}, Игнорируемых: {counts['ignored']}, Отрицательных: {counts['negative']}")

                # Сохраняем информацию о проблемном сценарии
                scenario_info = {
                    'image_path': img_path_str,
                    'aug_seed': current_aug_seed,
                    'total_positive_anchors': total_pos_anchors,
                    'augmented_gt_count': len(augmented_gt_objects),
                    'anchor_counts_by_level': anchor_counts_by_level,
                    'error': None # Нет ошибки, это проблемный сценарий назначения
                }
                problem_scenarios_found.append(scenario_info)

                # --- Визуализация проблемного сценария (Шаг 4) ---
                if len(figures_to_show) < max_problem_scenarios_to_visualize:
                    print(f"  Визуализация проблемного сценария (график {len(figures_to_show) + 1}/{max_problem_scenarios_to_visualize})...")
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1 строка, 3 колонки
                    fig.suptitle(f"Проблемный сценарий: {Path(img_path_str).name}, Сид: {current_aug_seed}, Pos: {total_pos_anchors}", fontsize=12)

                    # График 1: Изображение после ресайза/падинга + GT (Scaled/Shifted GT)
                    plot_utils.plot_augmented_gt(axes[0], image_padded_rgb_uint8, scaled_shifted_gt_objects) # Используем plot_augmented_gt, т.к. оно ожидает uint8 RGB
                    axes[0].set_title("Padded/Resized Image + Scaled/Shifted GT")

                    # График 2: Аугментированное изображение + GT (Augmented GT)
                    plot_utils.plot_augmented_gt(axes[1], image_np_augmented_uint8, augmented_gt_objects)
                    axes[1].set_title("Augmented Image + Augmented GT")


                    # График 3: Аугментированное изображение + Назначенные якоря
                    anchors_to_plot = []
                    # Собираем положительные, игнорируемые якоря
                    for level_name, assign_info_list in assignment_info_by_level.items():
                         for anchor_info in assign_info_list:
                              # Включаем положительные и игнорируемые
                              if anchor_info['type'] == POSITIVE_ANCHOR or anchor_info['type'] == IGNORED_ANCHOR:
                                   # Добавляем только необходимые поля для plot_utils.plot_specific_anchors_on_image
                                   anchors_to_plot.append({
                                       'bbox': anchor_info['bbox_pixel_raw'], # Координаты якоря
                                       'level': anchor_info['level'], # Уровень FPN
                                       'iou': anchor_info['iou'], # IoU с назначенным/лучшим GT
                                       'type': 'positive' if anchor_info['type'] == POSITIVE_ANCHOR else 'ignored' # Тип для отрисовки
                                   })
                              # Опционально: добавляем отрицательные якоря с IoU выше порога
                              elif anchor_info['type'] == NEGATIVE_ANCHOR and anchor_info['iou'] >= negative_anchor_viz_iou_threshold:
                                    anchors_to_plot.append({
                                        'bbox': anchor_info['bbox_pixel_raw'],
                                        'level': anchor_info['level'],
                                        'iou': anchor_info['iou'], # Макс IoU со *всеми* GT
                                        'type': 'negative'
                                    })


                    plot_utils.plot_specific_anchors_on_image(axes[2], image_np_augmented_uint8, anchors_to_plot, title="Augmented + Assigned Anchors")


                    # Настройка и сохранение графика
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Корректируем расположение
                    save_path = _graphs_output_dir / f"problem_seed_{current_aug_seed}_{Path(img_path_str).stem}.png"
                    plot_utils.save_plot(fig, str(save_path)) # plot_utils ожидает строку для пути

                    # Добавляем фигуру в список для опционального отображения в конце
                    # figures_to_show.append(fig) # Можно закомментировать, если не хотите показывать окна

                    # Закрываем фигуру, чтобы освободить память (оставляем только сохраненные файлы)
                    plt.close(fig)
                    print(f"  График проблемного сценария сохранен: {save_path}")


        except Exception as e:
            logger.error(f"  Ошибка при обработке изображения {Path(img_path_str).name} с сидом {current_aug_seed}: {e}", exc_info=True)
            # Добавляем информацию об ошибке как проблемный сценарий
            # Сохраняем информацию об ошибке
            scenario_info = {
                 'image_path': img_path_str,
                 'aug_seed': current_aug_seed,
                 'error': str(e),
                 'total_positive_anchors': 0 # Считаем ошибку как 0 positive якорей
            }
            problem_scenarios_found.append(scenario_info)


    print(f"\n--- Анализ завершен ---")
    print(f"Проверено изображений: {num_images_to_test}")
    print(f"Обнаружено проблемных сценариев (с положительными якорями <= {positive_anchor_problem_threshold}): {len(problem_scenarios_found)}")
    print(f"Визуализировано проблемных сценариев: {min(len(problem_scenarios_found), max_problem_scenarios_to_visualize)}")

    if problem_scenarios_found:
        print(f"\nИнформация о всех обнаруженных проблемных сценариях (используйте эти сиды для детальной отладки):")
        for i, scenario in enumerate(problem_scenarios_found):
            print(f"  {i+1}. Изображение: {Path(scenario['image_path']).name}, Сид аугментации: {scenario['aug_seed']}, Позитивных якорей: {scenario.get('total_positive_anchors', 'Ошибка')}, Аугм. GT: {scenario.get('augmented_gt_count', 'N/A')}. Ошибка: {scenario.get('error', 'Нет')}")
            if 'anchor_counts_by_level' in scenario and scenario['anchor_counts_by_level']:
                 print("    Детали по уровням:")
                 for level_name, counts in scenario['anchor_counts_by_level'].items():
                       print(f"      {level_name}: Pos: {counts['positive']}, Ign: {counts['ignored']}, Neg: {counts['negative']}, Total: {counts['total']}")

        print(f"\nГрафики проблемных сценариев сохранены в папке: {_graphs_output_dir}")

    # Опционально показать фигуры в окнах (может быть много окон!)
    # if figures_to_show:
    #     print(f"\nОтображение {len(figures_to_show)} окон с графиками. Закройте их, чтобы завершить скрипт.")
    #     plot_utils.show_plot()


    print("\n--- Тестовый блок data_loader_v3_standard.py завершен ---")