import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import yaml
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

# --- Загрузка Конфигурации (для путей и списка классов) ---
_current_project_root = Path(__file__).parent.resolve()  # Корень проекта

_base_config_path = _current_project_root / 'src' / 'configs' / 'base_config.yaml'
_detector_config_path = _current_project_root / 'src' / 'configs' / 'detector_config.yaml'

BASE_CONFIG = {}
DETECTOR_CONFIG = {}
CONFIG_LOAD_SUCCESS_ANCHORS = True

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS_ANCHORS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_ANCHORS = False;
    print(f"ОШИБКА (calculate_anchors.py): base_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_ANCHORS = False;
    print(f"ОШИБКА (calculate_anchors.py): YAML в base_config.yaml.")

try:
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_ANCHORS = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_ANCHORS = False;
    print(f"ОШИБКА (calculate_anchors.py): detector_config.yaml не найден.")
except yaml.YAMLError:
    CONFIG_LOAD_SUCCESS_ANCHORS = False;
    print(f"ОШИБКА (calculate_anchors.py): YAML в detector_config.yaml.")

if not CONFIG_LOAD_SUCCESS_ANCHORS:
    print("ОШИБКА: Не удалось загрузить конфиги. Задайте пути и параметры вручную в скрипте.")
    # Задаем дефолты, чтобы скрипт хотя бы попытался запуститься с ручным вводом
    # Эти пути нужно будет ОБЯЗАТЕЛЬНО проверить и изменить при ручном запуске, если конфиги не загрузились
    ANNOTATIONS_DIR_FOR_KMEANS_DEFAULT = str(_current_project_root / "data/Detector_Dataset_Ready/train/Annotations")
    IMAGES_DIR_FOR_KMEANS_DEFAULT = str(_current_project_root / "data/Detector_Dataset_Ready/train/JPEGImages")
    CLASSES_LIST_FOR_KMEANS_DEFAULT = ['pit', 'crack']  # Убедись, что это твои классы
else:
    # Пути к данным (используем данные из обучающей выборки Detector_Dataset_Ready)
    _detector_dataset_ready_path_rel = "data/Detector_Dataset_Ready"
    _detector_dataset_ready_abs = (_current_project_root / _detector_dataset_ready_path_rel).resolve()
    _images_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
    _annotations_subdir_name_cfg = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')

    ANNOTATIONS_DIR_FOR_KMEANS_DEFAULT = str(_detector_dataset_ready_abs / "train" / _annotations_subdir_name_cfg)
    IMAGES_DIR_FOR_KMEANS_DEFAULT = str(_detector_dataset_ready_abs / "train" / _images_subdir_name_cfg)
    CLASSES_LIST_FOR_KMEANS_DEFAULT = DETECTOR_CONFIG.get('classes', ['pit', 'crack'])

# --- Параметры для K-Means ---
# Эти параметры можно будет переопределить при вызове функции или оставить по умолчанию
NUM_ANCHORS_TO_CALCULATE_DEFAULT = DETECTOR_CONFIG.get('num_anchors_per_location', 6)  # Сколько якорей мы хотим найти
KMEANS_RANDOM_STATE = 42  # Для воспроизводимости результатов K-Means


def parse_xml_for_kmeans(xml_file_path, classes_list, image_dir_path=None):
    """
    Парсит XML файл и извлекает нормализованные ширины и высоты bounding box'ов.
    Если image_dir_path указан, пытается прочитать размеры изображения из файла,
    если они отсутствуют или некорректны в XML.
    """
    boxes_wh_normalized = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        img_width_xml, img_height_xml = None, None
        size_node = root.find('size')
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None and height_node is not None and width_node.text and height_node.text:
                try:
                    img_width_xml = int(width_node.text)
                    img_height_xml = int(height_node.text)
                    if img_width_xml <= 0 or img_height_xml <= 0:
                        img_width_xml, img_height_xml = None, None  # Сбрасываем, если некорректные
                except ValueError:
                    img_width_xml, img_height_xml = None, None

        # Если размеры не найдены/некорректны в XML, и указана папка с изображениями
        if (img_width_xml is None or img_height_xml is None) and image_dir_path:
            filename_node = root.find('filename')
            if filename_node is not None and filename_node.text:
                image_filename = filename_node.text
                # Пытаемся найти изображение с разными расширениями
                found_img_path = None
                for ext_candidate in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate_path = Path(image_dir_path) / (Path(image_filename).stem + ext_candidate)
                    if candidate_path.exists():
                        found_img_path = candidate_path
                        break

                if found_img_path:
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(found_img_path) as img:
                            img_width_xml, img_height_xml = img.size
                        if img_width_xml <= 0 or img_height_xml <= 0:
                            # print(f"  Предупреждение: Нулевые размеры из файла изображения {found_img_path} для XML {os.path.basename(xml_file_path)}")
                            return []  # Не можем нормализовать
                    except Exception as e_pil:
                        print(
                            f"  Предупреждение: Ошибка чтения размеров из файла изображения {found_img_path} для XML {os.path.basename(xml_file_path)}: {e_pil}")
                        return []  # Не можем нормализовать
                else:
                    # print(f"  Предупреждение: Файл изображения {image_filename} не найден в {image_dir_path} для XML {os.path.basename(xml_file_path)} для получения размеров.")
                    return []
            else:  # Если и имени файла нет в XML
                # print(f"  Предупреждение: Тег <filename> или <size> не найден/некорректен в {os.path.basename(xml_file_path)}, и нет пути к изображениям. Пропуск файла.")
                return []

        if img_width_xml is None or img_height_xml is None or img_width_xml == 0 or img_height_xml == 0:
            # print(f"  Предупреждение: Не удалось определить размеры изображения для {os.path.basename(xml_file_path)}. Пропуск файла.")
            return []

        for obj_node in root.findall('object'):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None: continue
            class_name = class_name_node.text
            if classes_list and class_name not in classes_list:  # Пропускаем, если класс не в списке
                continue

            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None: continue
            try:
                xmin = float(bndbox_node.find('xmin').text)
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text)
                ymax = float(bndbox_node.find('ymax').text)
            except (ValueError, AttributeError, TypeError):
                continue

            if xmin >= xmax or ymin >= ymax: continue  # Невалидный бокс

            # Клиппинг по размерам изображения, чтобы избежать W/H > 1.0
            xmin = max(0, min(xmin, img_width_xml))
            ymin = max(0, min(ymin, img_height_xml))
            xmax = max(0, min(xmax, img_width_xml))
            ymax = max(0, min(ymax, img_height_xml))
            if xmin >= xmax or ymin >= ymax: continue  # Проверка после клиппинга

            box_width_pixels = xmax - xmin
            box_height_pixels = ymax - ymin

            if box_width_pixels > 0 and box_height_pixels > 0:
                width_normalized = box_width_pixels / img_width_xml
                height_normalized = box_height_pixels / img_height_xml
                boxes_wh_normalized.append([width_normalized, height_normalized])

    except ET.ParseError:
        print(f"  Предупреждение: Ошибка парсинга XML: {os.path.basename(xml_file_path)}")
    except Exception as e:
        print(f"  Предупреждение: Неожиданная ошибка при обработке {os.path.basename(xml_file_path)}: {e}")
    return boxes_wh_normalized


def calculate_kmeans_anchors(annotations_dir, images_dir, num_anchors, classes_list, random_state=None):
    """
    Вычисляет якоря с помощью K-Means на основе размеров bounding box'ов.
    """
    all_boxes_wh_normalized = []
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))

    if not xml_files:
        print(f"ОШИБКА: XML файлы не найдены в {annotations_dir}")
        return None

    print(f"Найдено {len(xml_files)} XML файлов. Обработка...")
    processed_files_count = 0
    for xml_file in xml_files:
        boxes_wh = parse_xml_for_kmeans(xml_file, classes_list, images_dir)
        if boxes_wh:  # Если функция вернула не пустой список
            all_boxes_wh_normalized.extend(boxes_wh)
            processed_files_count += 1

    print(f"Обработано {processed_files_count} XML файлов с валидными объектами и размерами.")

    if not all_boxes_wh_normalized:
        print("ОШИБКА: Не найдено ни одного bounding box'а для K-Means анализа.")
        return None

    all_boxes_np = np.array(all_boxes_wh_normalized)
    print(f"Всего найдено {all_boxes_np.shape[0]} bounding box'ов для K-Means.")
    if all_boxes_np.shape[0] < num_anchors:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Количество найденных bounding box'ов ({all_boxes_np.shape[0]}) меньше, "
              f"чем запрашиваемое количество якорей ({num_anchors}). "
              f"K-Means может работать некорректно или вернуть меньше якорей.")
        # В sklearn KMeans упадет, если n_samples < n_clusters.
        # Можно вернуть уникальные боксы, если их меньше чем num_anchors,
        # или просто прервать, чтобы пользователь исправил num_anchors или добавил данных.
        if all_boxes_np.shape[0] == 0: return None
        print("K-Means будет запущен с n_clusters = количеству уникальных размеров боксов, если их меньше num_anchors")
        # Убедимся, что num_anchors не больше, чем количество уникальных точек
        # Это не совсем корректно, так как k-means ищет центроиды, а не просто уникальные.
        # Лучше, если n_samples >= n_clusters.
        # Мы можем выдать предупреждение и продолжить, sklearn сам обработает, если сможет.
        # Или мы можем сделать num_anchors = min(num_anchors, all_boxes_np.shape[0])
        # если all_boxes_np.shape[0] > 0.
        if all_boxes_np.shape[0] < num_anchors:
            print(f"  Уменьшаем num_anchors до {all_boxes_np.shape[0]} для K-Means.")
            num_anchors = all_boxes_np.shape[0]
            if num_anchors == 0: return None  # Если после этого 0, то выходим.

    print(f"\nЗапуск K-Means для поиска {num_anchors} якорей...")
    kmeans = KMeans(n_clusters=num_anchors, random_state=random_state,
                    n_init='auto')  # n_init='auto' для подавления warning
    kmeans.fit(all_boxes_np)
    anchors = kmeans.cluster_centers_  # Это наши якоря [width_norm, height_norm]

    # Сортируем якоря по площади (от меньшего к большему) для консистентности
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    print("\nНайденные якоря (ширина_норм, высота_норм):")
    for anchor in anchors:
        print(f"  - [{anchor[0]:.4f}, {anchor[1]:.4f}]")

    # Визуализация
    plt.figure(figsize=(10, 7))
    plt.scatter(all_boxes_np[:, 0], all_boxes_np[:, 1], alpha=0.3, label='Ground Truth Boxes (W_norm, H_norm)')
    plt.scatter(anchors[:, 0], anchors[:, 1], color='red', marker='x', s=100, label=f'{num_anchors} K-Means Anchors')
    plt.xlabel("Нормализованная Ширина (W_norm)")
    plt.ylabel("Нормализованная Высота (H_norm)")
    plt.title(f"Распределение Размеров Bounding Box'ов и {num_anchors} Якорей")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Сохранение графика
    plot_save_path = _current_project_root / "graphs/anchor_clusters_visualization.png"
    try:
        plt.savefig(plot_save_path)
        print(f"\nГрафик сохранен в: {plot_save_path}")
    except Exception as e_plot:
        print(f"Ошибка сохранения графика: {e_plot}")
    plt.show()

    return anchors


if __name__ == '__main__':
    print("--- Запуск calculate_anchors.py ---")

    # Используем пути из загруженных конфигов (или дефолты, если конфиги не загрузились)
    annotations_directory = ANNOTATIONS_DIR_FOR_KMEANS_DEFAULT
    images_directory = IMAGES_DIR_FOR_KMEANS_DEFAULT
    num_clusters = NUM_ANCHORS_TO_CALCULATE_DEFAULT
    target_classes = CLASSES_LIST_FOR_KMEANS_DEFAULT

    print(f"Директория аннотаций для K-Means: {annotations_directory}")
    print(f"Директория изображений для K-Means (для размеров): {images_directory}")
    print(f"Желаемое количество якорей: {num_clusters}")
    print(f"Классы, учитываемые для K-Means: {target_classes}")
    print(f"KMeans random_state: {KMEANS_RANDOM_STATE}\n")

    if not Path(annotations_directory).is_dir():
        print(f"ОШИБКА: Директория аннотаций '{annotations_directory}' не найдена.")
        print(
            "Убедитесь, что датасет (например, из data/Detector_Dataset_Ready/train/Annotations) существует и путь корректен.")
    elif not Path(images_directory).is_dir():
        print(f"ОШИБКА: Директория изображений '{images_directory}' не найдена.")
        print("Это необходимо для получения размеров изображений, если они отсутствуют в XML.")
    else:
        calculated_anchors = calculate_kmeans_anchors(
            annotations_directory,
            images_directory,  # Передаем путь к изображениям
            num_clusters,
            target_classes,  # Учитываем только объекты этих классов
            random_state=KMEANS_RANDOM_STATE
        )
        if calculated_anchors is not None:
            print("\n--- K-Means для якорей успешно завершен ---")
        else:
            print("\n--- K-Means для якорей завершен с ошибками или без результатов ---")