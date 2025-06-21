import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Загрузка Конфигураций ---
_project_root_anchor_calc = Path(__file__).resolve().parent
_src_path_anchor_calc = _project_root_anchor_calc / 'src'
_base_config_path_ac = _src_path_anchor_calc / 'configs' / 'base_config.yaml'
_detector_config_path_ac = _src_path_anchor_calc / 'configs' / 'detector_config_single_level_debug.yaml'  # Используем отладочный конфиг для классов

BASE_CONFIG_AC = {}
DETECTOR_CONFIG_AC = {}  # Будет содержать detector_config_single_level_debug.yaml

try:
    with open(_base_config_path_ac, 'r', encoding='utf-8') as f:
        BASE_CONFIG_AC = yaml.safe_load(f)
    with open(_detector_config_path_ac, 'r', encoding='utf-8') as f:  # Загружаем отладочный конфиг
        DETECTOR_CONFIG_AC = yaml.safe_load(f)
except Exception as e:
    print(f"ОШИБКА: Не удалось загрузить конфигурационные файлы: {e}")
    print("Убедитесь, что base_config.yaml и detector_config_single_level_debug.yaml существуют и корректны.")
    exit()

IMAGES_SUBDIR_NAME_AC = BASE_CONFIG_AC.get('dataset', {}).get('images_dir', 'images')
# Сначала получаем имя подпапки аннотаций из конфига
ANNOTATIONS_SUBDIR_NAME_FROM_CONFIG_AC = DETECTOR_CONFIG_AC.get('fpn_detector_params', {}).get('dataset_internal_paths_for_yolo_conversion_or_similar', {}).get('annotations_subdir_name',
                                        BASE_CONFIG_AC.get('dataset', {}).get('annotations_dir', 'Annotations'))
# ^^^ Используем более надежное получение, если структура конфига сложная,
# или просто: ANNOTATIONS_SUBDIR_NAME_FROM_CONFIG_AC = BASE_CONFIG_AC.get('dataset', {}).get('annotations_dir', 'Annotations')
# если 'annotations_dir' лежит в BASE_CONFIG -> dataset

# Путь к РАЗДЕЛЕННОМУ датасету для детектора (берем только TRAIN выборку)
_detector_dataset_ready_path_rel_ac = "../data/Detector_Dataset_Ready"  # Это должно быть согласовано с create_data_splits.py
DETECTOR_DATASET_READY_ABS_AC = (_project_root_anchor_calc / _detector_dataset_ready_path_rel_ac).resolve()

# Теперь строим полный путь, используя полученное имя подпапки
TRAIN_ANNOT_DIR_AC = DETECTOR_DATASET_READY_ABS_AC / "train" / ANNOTATIONS_SUBDIR_NAME_FROM_CONFIG_AC

# Классы из отладочного конфига детектора
_fpn_params_ac = DETECTOR_CONFIG_AC.get('fpn_detector_params', {})
CLASSES_LIST_AC = _fpn_params_ac.get('classes', ['pit', 'crack'])

print(f"--- Расчет якорей для ОДНОУРОВНЕВОЙ модели ---")
print(f"Используемые классы: {CLASSES_LIST_AC}")
print(f"Поиск аннотаций в: {TRAIN_ANNOT_DIR_AC}") # Теперь TRAIN_ANNOT_DIR_AC будет правильно сформирован
# Классы из отладочного конфига детектора
_fpn_params_ac = DETECTOR_CONFIG_AC.get('fpn_detector_params',
                                        {})  # detector_config_single_level_debug.yaml имеет эту структуру
CLASSES_LIST_AC = _fpn_params_ac.get('classes', ['pit', 'crack'])

print(f"--- Расчет якорей для ОДНОУРОВНЕВОЙ модели ---")
print(f"Используемые классы: {CLASSES_LIST_AC}")
print(f"Поиск аннотаций в: {TRAIN_ANNOT_DIR_AC}")


def parse_xml_for_bbox_dimensions(xml_file_path):
    """Извлекает размеры (ширина, высота) всех объектов и размеры изображения из XML."""
    bboxes_dims = []
    img_w, img_h = 0, 0
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size_node = root.find('size')
        if size_node is not None:
            img_w = int(size_node.find('width').text)
            img_h = int(size_node.find('height').text)
        if img_w == 0 or img_h == 0:  # Если в XML нет размеров, пропускаем
            # print(f"  Предупреждение: Нулевые размеры изображения в {os.path.basename(xml_file_path)}, пропускаем.")
            return [], None, None

        for obj_node in root.findall('object'):
            class_name = obj_node.find('name').text
            if class_name not in CLASSES_LIST_AC:  # Учитываем только наши классы
                continue
            bndbox_node = obj_node.find('bndbox')
            xmin = float(bndbox_node.find('xmin').text)
            ymin = float(bndbox_node.find('ymin').text)
            xmax = float(bndbox_node.find('xmax').text)
            ymax = float(bndbox_node.find('ymax').text)
            if xmin >= xmax or ymin >= ymax: continue  # Пропускаем некорректные рамки

            # Нормализованные ширина и высота
            box_w_norm = (xmax - xmin) / img_w
            box_h_norm = (ymax - ymin) / img_h
            bboxes_dims.append([box_w_norm, box_h_norm])
        return bboxes_dims, img_w, img_h
    except Exception as e:
        # print(f"  Ошибка парсинга {os.path.basename(xml_file_path)}: {e}")
        return [], None, None


def calculate_iou_for_kmeans(box_wh, cluster_wh):
    """
    Расчет IoU между одним box_wh [w,h] и одним cluster_wh [w,h].
    Используется в K-Means, где сравниваются только формы (площади пересечения).
    """
    inter_w = min(box_wh[0], cluster_wh[0])
    inter_h = min(box_wh[1], cluster_wh[1])
    intersection = inter_w * inter_h
    union = box_wh[0] * box_wh[1] + cluster_wh[0] * cluster_wh[1] - intersection
    return intersection / (union + 1e-6)  # Добавляем эпсилон для избежания деления на ноль


def kmeans_anchors(all_normalized_boxes_wh, num_clusters):
    """
    Выполняет K-Means кластеризацию для нахождения центроидов якорей.
    all_normalized_boxes_wh: numpy array формы [N, 2] с нормализованными (ширина, высота).
    num_clusters: количество якорей, которое мы хотим получить.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')  # n_init='auto' для подавления UserWarning
    kmeans.fit(all_normalized_boxes_wh)
    anchors = kmeans.cluster_centers_  # Это и есть наши [width, height] якоря

    # Сортируем якоря по площади (от меньшего к большему) для консистентности
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = anchors[sorted_indices]

    return sorted_anchors


def calculate_avg_iou(all_normalized_boxes_wh, anchors_wh):
    """
    Рассчитывает средний IoU между всеми GT боксами и набором якорей.
    Каждый GT бокс сопоставляется с якорем, дающим максимальный IoU.
    """
    total_iou = 0
    num_boxes = len(all_normalized_boxes_wh)
    if num_boxes == 0:
        return 0

    for box_wh in all_normalized_boxes_wh:
        max_iou_for_box = 0
        for anchor_wh in anchors_wh:
            iou = calculate_iou_for_kmeans(box_wh, anchor_wh)
            if iou > max_iou_for_box:
                max_iou_for_box = iou
        total_iou += max_iou_for_box

    return total_iou / num_boxes


def main_anchor_calculation():
    if not TRAIN_ANNOT_DIR_AC.is_dir():
        print(f"ОШИБКА: Директория с аннотациями для обучения не найдена: {TRAIN_ANNOT_DIR_AC}")
        print(
            "Убедитесь, что скрипт 'create_data_splits.py' был запущен и создал data/Detector_Dataset_Ready/train/Annotations/")
        return

    xml_files = list(TRAIN_ANNOT_DIR_AC.glob("*.xml"))
    if not xml_files:
        print(f"В директории {TRAIN_ANNOT_DIR_AC} не найдено XML файлов.")
        return

    print(f"Найдено {len(xml_files)} XML файлов для анализа...")

    all_gt_boxes_normalized_wh = []
    parsed_files_count = 0
    error_files_count = 0

    for xml_file in xml_files:
        bboxes_dims, _, _ = parse_xml_for_bbox_dimensions(str(xml_file))
        if bboxes_dims is not None:  # Успешный парсинг (даже если список bboxes_dims пуст)
            all_gt_boxes_normalized_wh.extend(bboxes_dims)
            parsed_files_count += 1
        else:  # Ошибка парсинга файла
            error_files_count += 1
        if parsed_files_count % 500 == 0:
            print(f"  Обработано {parsed_files_count} файлов...")

    print(f"Всего обработано XML: {parsed_files_count}, из них с ошибками парсинга: {error_files_count}")
    if not all_gt_boxes_normalized_wh:
        print("Не найдено ни одного bounding box'а в аннотациях. Невозможно рассчитать якоря.")
        return

    all_gt_boxes_np = np.array(all_gt_boxes_normalized_wh)
    print(f"Всего извлечено {all_gt_boxes_np.shape[0]} bounding box'ов для K-Means.")

    # --- K-Means для разного количества якорей ---
    num_anchors_to_test = range(1, 16)  # Будем тестировать от 1 до 15 якорей
    avg_ious = []
    all_calculated_anchors = {}

    print("\n--- Запуск K-Means для разного количества якорей ---")
    for k_anchors in num_anchors_to_test:
        print(f"  Рассчитываем для {k_anchors} якорей...")
        current_anchors_wh = kmeans_anchors(all_gt_boxes_np, k_anchors)
        avg_iou_for_k = calculate_avg_iou(all_gt_boxes_np, current_anchors_wh)
        avg_ious.append(avg_iou_for_k)
        all_calculated_anchors[k_anchors] = current_anchors_wh
        print(f"    Avg IoU: {avg_iou_for_k:.4f}")
        # print(f"    Якоря [W_norm, H_norm]:\n{current_anchors_wh}")

    # --- Построение графика "локтя" ---
    plt.figure(figsize=(10, 6))
    plt.plot(list(num_anchors_to_test), avg_ious, marker='o')
    plt.title('Метод "Локтя" для Выбора Количества Якорей (Одноуровневая Модель)')
    plt.xlabel('Количество Якорей (Кластеров)')
    plt.ylabel('Средний IoU')
    plt.xticks(list(num_anchors_to_test))
    plt.grid(True)
    elbow_plot_path = _project_root_anchor_calc / "graphs" / "single_level_anchors_elbow_plot.png"
    elbow_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(elbow_plot_path))
    print(f"\nГрафик 'локтя' сохранен в: {elbow_plot_path}")
    # plt.show() # Раскомментируй, если хочешь увидеть график сразу

    # --- Вывод результатов ---
    print("\n--- Результаты K-Means (Avg IoU и Предложенные Якоря) ---")
    for k_anchors, iou_val in zip(num_anchors_to_test, avg_ious):
        print(f"\nКоличество якорей: {k_anchors}, Средний IoU: {iou_val:.4f}")
        print("  Предложенные якоря [ширина_норм, высота_норм] (отсортированы по площади):")
        anchors_for_k = all_calculated_anchors[k_anchors]
        for anchor in anchors_for_k:
            print(f"    - [{anchor[0]:.4f}, {anchor[1]:.4f}]")

    print("\n--- Как выбрать количество якорей ---")
    print("1. Посмотри на график 'локтя'. Выбери точку, после которой прирост Avg IoU становится незначительным.")
    print("2. Для одноуровневой модели обычно выбирают от 3 до 9 якорей.")
    print("3. Учитывай сложность модели: больше якорей -> больше параметров в предсказывающей голове.")
    print("4. Если у тебя 7 якорей для P4_debug в конфиге, сравни Avg IoU для k=7 с другими значениями.")
    print("\nПосле выбора оптимального количества якорей, скопируй их [ширина, высота] значения")
    print("в 'detector_config_single_level_debug.yaml' в секцию:")
    print("fpn_detector_params -> detector_fpn_anchor_configs -> P4_debug -> anchors_wh_normalized")
    print("И обнови 'num_anchors_this_level' для P4_debug.")


if __name__ == '__main__':
    # Убедись, что create_data_splits.py был запущен, и данные лежат в
    # data/Detector_Dataset_Ready/train/Annotations/
    # Также убедись, что пути в начале скрипта и в конфигах верны.
    main_anchor_calculation()