# calculate_anchors.py
import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import yaml
import os
from pathlib import Path
import random
import sys

# --- Загрузка Нового Конфигурационного Файла ---
# Определяем базовый путь к скрипту и пытаемся найти конфиг
_current_script_root = Path(__file__).resolve().parent
# Пробуем найти конфиг в src/configs или в корневой папке configs
_config_path = _current_script_root / 'src' / 'configs' / 'detector_config_v3_standard.yaml'
if not _config_path.exists():
    _config_path = _current_script_root / 'configs' / 'detector_config_v3_standard.yaml'
    if not _config_path.exists():
        print(f"ОШИБКА: Конфигурационный файл 'detector_config_v3_standard.yaml' не найден.")
        print(f"Ожидалось по путям: {_current_script_root / 'src' / 'configs' / 'detector_config_v3_standard.yaml'} или {_current_script_root / 'configs' / 'detector_config_v3_standard.yaml'}")
        sys.exit("Не удалось найти файл конфигурации. Выход.")

CONFIG = {}
CONFIG_LOAD_SUCCESS = True
try:
    with open(_config_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    if not isinstance(CONFIG, dict):
        CONFIG = {}
        CONFIG_LOAD_SUCCESS = False
        print(f"ОШИБКА: Конфигурационный файл загружен, но его содержимое не является словарем: {_config_path}")
except Exception as e:
    CONFIG_LOAD_SUCCESS = False
    print(f"ОШИБКА при загрузке или парсинге конфигурационного файла {_config_path}: {e}")

if not CONFIG_LOAD_SUCCESS:
    sys.exit("Критическая ошибка загрузки конфигурации. Выход.")

# --- Глобальные Параметры из Нового Конфига ---
# Проверяем наличие необходимых ключей в конфиге
REQUIRED_KEYS = [
    'dataset_path', 'train_images_subdir', 'train_annotations_subdir',
    'class_names', 'input_shape', 'fpn_gt_assignment_area_ranges',
    'num_anchors_per_level', 'anchor_calc_k_range', 'anchor_analysis_output_dir'
]
for key in REQUIRED_KEYS:
    if key not in CONFIG:
        sys.exit(f"ОШИБКА: Отсутствует обязательный ключ '{key}' в конфигурационном файле. Выход.")

DATASET_BASE_PATH = Path(CONFIG['dataset_path'])
TRAIN_IMAGES_SUBDIR = CONFIG['train_images_subdir']
TRAIN_ANNOTATIONS_SUBDIR = CONFIG['train_annotations_subdir']
CLASSES_LIST = CONFIG['class_names']
INPUT_HEIGHT = CONFIG['input_shape'][0]
INPUT_WIDTH = CONFIG['input_shape'][1]

FPN_GT_ASSIGNMENT_AREA_RANGES = CONFIG['fpn_gt_assignment_area_ranges']
# Преобразуем диапазоны площадей в удобный для использования формат
# Например: [[0, 1024], [1024, 9216], [9216, float('inf')]] -> [[0, 1024], [1024, 9216], [9216, 1e10]]
# Убедимся, что количество диапазонов соответствует количеству уровней FPN P3, P4, P5
if len(FPN_GT_ASSIGNMENT_AREA_RANGES) != 3:
    sys.exit("ОШИБКА: 'fpn_gt_assignment_area_ranges' в конфиге должен содержать 3 диапазона для P3, P4, P5. Выход.")

# Заменяем float('inf') или очень большие числа на более приемлемое большое число, если нужно
# Убеждаемся, что пороги последовательны
parsed_area_ranges = []
last_upper_bound = 0
for i, range_pair in enumerate(FPN_GT_ASSIGNMENT_AREA_RANGES):
    if len(range_pair) != 2:
         sys.exit(f"ОШИБКА: Диапазон {i} в 'fpn_gt_assignment_area_ranges' должен содержать 2 значения [min, max]. Выход.")
    lower, upper = range_pair[0], range_pair[1]
    if lower < last_upper_bound:
         print(f"ПРЕДУПРЕЖДЕНИЕ: Нижняя граница диапазона {i} ({lower}) меньше верхней границы предыдущего диапазона ({last_upper_bound}). Проверьте 'fpn_gt_assignment_area_ranges'.")
    if upper == float('inf') or upper > 1e9: # Обрабатываем float('inf') или очень большие числа
         upper = 1e10 # Используем большое число для практических целей
    if lower >= upper:
         print(f"ПРЕДУПРЕЖДЕНИЕ: Нижняя граница диапазона {i} ({lower}) >= верхней границе ({upper}). Проверьте 'fpn_gt_assignment_area_ranges'.")
    parsed_area_ranges.append([lower, upper])
    last_upper_bound = upper

FPN_GT_ASSIGNMENT_AREA_RANGES = parsed_area_ranges

NUM_FINAL_ANCHORS_PER_LEVEL = CONFIG['num_anchors_per_level_K_means']
if len(NUM_FINAL_ANCHORS_PER_LEVEL) != 3 or 'P3' not in NUM_FINAL_ANCHORS_PER_LEVEL or 'P4' not in NUM_FINAL_ANCHORS_PER_LEVEL or 'P5' not in NUM_FINAL_ANCHORS_PER_LEVEL:
    sys.exit("ОШИБКА: 'num_anchors_per_level' в конфиге должен содержать ключи P3, P4, P5. Выход.")

K_RANGE_FOR_ANALYSIS = list(CONFIG.get('anchor_calc_k_range', range(1, 8))) # Дефолтный диапазон, если нет в конфиге

OUTPUT_GRAPHS_DIR = Path(CONFIG['anchor_analysis_output_dir'])
OUTPUT_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_FINAL_GRAPH_FILENAME = "fpn_anchors_distribution_final.png"


# --- Вспомогательные функции (без изменений, т.к. логика универсальна) ---

def parse_annotations_for_wh(annotations_dir_arg, images_dir_for_size_fallback_arg, classes_to_consider_arg=None):
    """Парсит XML аннотации и возвращает список нормализованных пар (ширина, высота) bbox."""
    all_boxes_wh_norm = [];
    print(f"Сканирование аннотаций в: {annotations_dir_arg}")
    xml_files = glob.glob(os.path.join(annotations_dir_arg, "*.xml"))
    if not xml_files: print(f"  ПРЕДУПРЕЖДЕНИЕ: XML файлы не найдены в {annotations_dir_arg}"); return np.array(
        all_boxes_wh_norm)
    num_processed = 0
    for xml_file in xml_files:
        num_processed += 1
        if num_processed % 100 == 0:
            print(f"  Обработано {num_processed} XML файлов...")
        try:
            tree = ET.parse(xml_file);
            root = tree.getroot()
            img_w_xml, img_h_xml = None, None
            size_node = root.find('size')
            if size_node is not None:
                w_n, h_n = size_node.find('width'), size_node.find('height')
                if w_n is not None and h_n is not None and w_n.text and h_n.text:
                    try:
                        img_w_xml, img_h_xml = int(w_n.text), int(h_n.text)
                    except ValueError:
                        pass
                    if img_w_xml is not None and (img_w_xml <= 0 or img_h_xml <= 0): img_w_xml, img_h_xml = None, None
            if img_w_xml is None or img_h_xml is None:
                img_fn_node = root.find('filename')
                if img_fn_node is not None and img_fn_node.text:
                    img_path_cand = Path(images_dir_for_size_fallback_arg) / img_fn_node.text
                    if not img_path_cand.exists():
                        base_fn, _ = os.path.splitext(img_fn_node.text)
                        # Пробуем разные расширения, если оригинальное не найдено
                        for ext_try_fn in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                            if (Path(images_dir_for_size_fallback_arg) / (base_fn + ext_try_fn)).exists():
                                img_path_cand = Path(images_dir_for_size_fallback_arg) / (base_fn + ext_try_fn);
                                break
                    if img_path_cand.exists():
                        try:
                            from PIL import Image;img_pil = Image.open(
                                img_path_cand);img_w_xml, img_h_xml = img_pil.size;img_pil.close()
                        except Exception as img_e:
                            #print(f"    Не удалось прочитать размеры из изображения {img_path_cand} ({img_e}). Пропуск XML.")
                            continue # Пропускаем этот XML, если не можем получить размер изображения
                    else:
                        #print(f"    Файл изображения для {xml_file} не найден для размеров. Пропуск XML.");
                        continue # Пропускаем этот XML
                else:
                    #print(f"    Тег <filename> или <size> не найден или пуст в {xml_file}. Пропуск XML.");
                    continue # Пропускаем этот XML
            if img_w_xml is None or img_h_xml is None:
                 print(f"    Не удалось получить размеры изображения для {xml_file}. Пропуск XML."); continue

            for obj_node in root.findall('object'):
                cls_nm_node = obj_node.find('name')
                if cls_nm_node is None or cls_nm_node.text is None: continue
                # Учитываем только классы, указанные в конфиге
                if classes_to_consider_arg and cls_nm_node.text not in classes_to_consider_arg: continue

                bndb_node = obj_node.find('bndbox')
                if bndb_node is None: continue
                try:
                    # Учитываем, что в VOC формат может быть 1-based indexing, но обычно парсинг дает 0-based
                    # Убедимся, что координаты не меньше 0 и не больше размера изображения
                    xmin_str = bndb_node.find('xmin').text
                    ymin_str = bndb_node.find('ymin').text
                    xmax_str = bndb_node.find('xmax').text
                    ymax_str = bndb_node.find('ymax').text

                    xmin = float(xmin_str)
                    ymin = float(ymin_str)
                    xmax = float(xmax_str)
                    ymax = float(ymax_str)

                    # Простая проверка на адекватность координат после парсинга
                    if xmin < 0 or ymin < 0 or xmax > img_w_xml or ymax > img_h_xml or xmin >= xmax or ymin >= ymax:
                        #print(f"    Неадекватные координаты bbox в {xml_file}: [{xmin},{ymin},{xmax},{ymax}] при размере {img_w_xml}x{img_h_xml}. Пропуск bbox.")
                        continue # Пропускаем этот bbox

                except Exception as coord_e:
                    #print(f"    Ошибка парсинга координат bbox в {xml_file} ({coord_e}). Пропуск bbox.")
                    continue # Пропускаем этот bbox

                box_w = xmax - xmin
                box_h = ymax - ymin

                # Нормализуем ширину и высоту к размерам изображения
                all_boxes_wh_norm.append([box_w / float(img_w_xml), box_h / float(img_h_xml)])

        except Exception as parse_e:
            print(f"  Ошибка обработки XML файла {xml_file} ({parse_e}). Пропуск.")
    print(f"Завершено сканирование. Найдено {len(all_boxes_wh_norm)} BBox.")
    return np.array(all_boxes_wh_norm)


def iou_for_kmeans(boxes_arg, clusters_arg):
    """Вычисляет IoU между каждым боксом и каждым кластером (якорем) только по WxH."""
    n = boxes_arg.shape[0];
    k = clusters_arg.shape[0]
    if n == 0 or k == 0: return np.zeros((n,k))

    box_area_calc = boxes_arg[:, 0] * boxes_arg[:, 1];
    cluster_area_calc = clusters_arg[:, 0] * clusters_arg[:, 1]

    # Расширяем массивы для broadcast
    box_w_reshaped = np.repeat(boxes_arg[:, 0], k).reshape(n, k)
    box_h_reshaped = np.repeat(boxes_arg[:, 1], k).reshape(n, k)
    cluster_w_reshaped = np.tile(clusters_arg[:, 0], n).reshape(n, k)
    cluster_h_reshaped = np.tile(clusters_arg[:, 1], n).reshape(n, k)

    # Пересечение (Minimum) по каждой оси
    intersection_w_calc = np.minimum(box_w_reshaped, cluster_w_reshaped)
    intersection_h_calc = np.minimum(box_h_reshaped, cluster_h_reshaped)
    intersection_area_calc = intersection_w_calc * intersection_h_calc

    # Объединение
    box_area_reshaped_calc = np.repeat(box_area_calc, k).reshape(n, k);
    cluster_area_reshaped_calc = np.tile(cluster_area_calc, n).reshape(n, k)
    union_area_calc = box_area_reshaped_calc + cluster_area_reshaped_calc - intersection_area_calc

    # IoU
    # Избегаем деления на ноль, если и пересечение и объединение равны нулю (случай нулевых боксов/кластеров)
    iou_calc = np.zeros_like(intersection_area_calc)
    valid_indices = union_area_calc > 1e-6 # Избегаем деления на ноль
    iou_calc[valid_indices] = intersection_area_calc[valid_indices] / union_area_calc[valid_indices]

    return iou_calc


def avg_iou_calc(boxes_arg, clusters_arg):
    """Вычисляет средний максимальный IoU для каждого бокса относительно всех кластеров."""
    if clusters_arg.shape[0] == 0 or boxes_arg.shape[0] == 0: return 0.0
    # Для каждого бокса находим максимальный IoU среди всех кластеров, затем усредняем по всем боксам
    return np.mean(np.max(iou_for_kmeans(boxes_arg, clusters_arg), axis=1))


def run_kmeans_analysis_for_group(boxes_wh_group_arg, k_range_arg, group_name_arg):
    """Выполняет K-Means для группы боксов для каждого K в диапазоне."""
    print(f"\n--- Анализ якорей для '{group_name_arg}' (найдено {boxes_wh_group_arg.shape[0]} рамок) ---")
    if boxes_wh_group_arg.shape[0] == 0: print("  Нет рамок для этой группы.");return {}, {}, {}, []
    wcss_res, avg_iou_vals_res, anchors_for_k_res = {}, {}, {};
    valid_k_range_res = []
    # Сортируем k_range_arg для более логичного вывода графиков
    sorted_k_range_arg = sorted([k for k in k_range_arg if isinstance(k, int) and k > 0])

    for k_val_arg in sorted_k_range_arg:
        if boxes_wh_group_arg.shape[0] < k_val_arg:
            # print(f"  K={k_val_arg}: Пропуск (мало рамок для K-Means)");
            continue # Пропускаем K, если данных меньше, чем K кластеров
        valid_k_range_res.append(k_val_arg)
        # Используем n_init='auto' или число (например, 10) для подавления предупреждения в новых версиях sklearn
        kmeans = KMeans(n_clusters=k_val_arg, random_state=42, n_init=10);
        kmeans.fit(boxes_wh_group_arg)
        wcss_res[k_val_arg] = kmeans.inertia_
        # Сортируем кластеры (якоря) по их площади для единообразия
        curr_anchors = kmeans.cluster_centers_;
        curr_anchors = curr_anchors[np.argsort(curr_anchors[:, 0] * curr_anchors[:, 1])]
        anchors_for_k_res[k_val_arg] = curr_anchors
        avg_iou_vals_res[k_val_arg] = avg_iou_calc(boxes_wh_group_arg, curr_anchors)
        print(f"  K={k_val_arg}: WCSS={wcss_res[k_val_arg]:.2f}, AvgIoU={avg_iou_vals_res[k_val_arg]:.4f}")

    if not valid_k_range_res:
         print(f"  Недостаточно рамок для выполнения K-Means для группы '{group_name_arg}' с любым K из диапазона {sorted_k_range_arg}.")

    return wcss_res, avg_iou_vals_res, anchors_for_k_res, valid_k_range_res


def plot_elbow_and_iou(k_range_plot_arg, wcss_dict_arg, avg_iou_dict_arg, group_name_plot_arg, save_dir_arg):
    """Строит графики WCSS и среднего IoU."""
    # Фильтруем данные, оставляя только те K, для которых есть результаты
    actual_k_range_plot = sorted([k for k in k_range_plot_arg if k in wcss_dict_arg and k in avg_iou_dict_arg])
    if not actual_k_range_plot: print(f"Нет валидных K для построения графика для группы '{group_name_plot_arg}'."); return

    wcss_list_plot = [wcss_dict_arg[k_item] for k_item in actual_k_range_plot]
    avg_iou_list_plot = [avg_iou_dict_arg[k_item] for k_item in actual_k_range_plot]

    fig, ax1 = plt.subplots(figsize=(10, 6));
    color1 = 'tab:red'
    ax1.set_xlabel('Количество якорей (K)');
    ax1.set_ylabel('WCSS (Inertia)', color=color1)
    ax1.plot(actual_k_range_plot, wcss_list_plot, color=color1, marker='o', label='WCSS');
    ax1.tick_params(axis='y', labelcolor=color1);
    ax1.grid(True, linestyle=':')
    ax2 = ax1.twinx();
    color2 = 'tab:blue'
    ax2.set_ylabel('Средний IoU', color=color2);
    ax2.plot(actual_k_range_plot, avg_iou_list_plot, color=color2, marker='x', label='Avg IoU');
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'Метод "Локтя" и Средний IoU для якорей группы {group_name_plot_arg}');
    fig.tight_layout()
    lines1_leg, labels1_leg = ax1.get_legend_handles_labels();
    lines2_leg, labels2_leg = ax2.get_legend_handles_labels()
    ax2.legend(lines1_leg + lines2_leg, labels1_leg + labels2_leg, loc='center right')
    save_path = save_dir_arg / f"elbow_iou_anchors_{group_name_plot_arg}.png"
    plt.savefig(save_path);
    print(f"График для '{group_name_plot_arg}' сохранен: {save_path}")
    plt.close()


def plot_final_anchors(all_boxes_wh_norm_plot_arg, final_anchors_dict_plot_arg, save_path_plot_arg, input_shape_plot_arg):
    """Строит scatter plot всех bbox и выбранных финальных якорей."""
    plt.figure(figsize=(12, 9));
    plt.scatter(all_boxes_wh_norm_plot_arg[:, 0], all_boxes_wh_norm_plot_arg[:, 1], alpha=0.3, s=5, label='Ground Truth Boxes (W_norm, H_norm)')

    colors_plot = {'P3': 'red', 'P4': 'green', 'P5': 'purple'};
    markers_plot = {'P3': 'x', 'P4': 's', 'P5': '^'}
    level_strides = {'P3': 8, 'P4': 16, 'P5': 32} # Предполагаемые страйды

    for level_name, anchors in final_anchors_dict_plot_arg.items():
        if anchors is not None and anchors.size > 0:
            plt.scatter(anchors[:, 0], anchors[:, 1],
                        color=colors_plot.get(level_name, 'black'),
                        marker=markers_plot.get(level_name, 'o'),
                        s=150, edgecolor='white', linewidth=1.5,
                        label=f'{anchors.shape[0]} K-Means Anchors {level_name} (stride {level_strides.get(level_name, "?")})')

    plt.xlabel('Нормализованная Ширина (W / Image_W)');
    plt.ylabel('Нормализованная Высота (H / Image_H)')
    plt.title(f'Распределение Размеров BBox и Финальных Якорей для FPN (Image Size: {input_shape_plot_arg[1]}x{input_shape_plot_arg[0]})');
    plt.xlim(0, 1.05);
    plt.ylim(0, 1.05);
    plt.grid(True, linestyle='--');
    plt.legend()
    plt.savefig(save_path_plot_arg);
    print(f"Финальный график якорей сохранен: {save_path_plot_arg}");
    plt.close()


def main():
    """Основная точка входа для расчета якорей."""
    print("--- Запуск скрипта calculate_anchors.py ---")

    # --- Определяем пути и параметры из конфига ---
    detector_dataset_ready_abs = (_current_script_root / DATASET_BASE_PATH).resolve()

    annotations_dir_train = str(detector_dataset_ready_abs / TRAIN_ANNOTATIONS_SUBDIR)
    images_dir_train = str(detector_dataset_ready_abs / TRAIN_IMAGES_SUBDIR)

    classes_list = CLASSES_LIST
    input_shape = CONFIG['input_shape']
    area_ranges_normalized = FPN_GT_ASSIGNMENT_AREA_RANGES # [[min_norm_area_P3, max_norm_area_P3], ...]

    num_final_anchors_p3 = NUM_FINAL_ANCHORS_PER_LEVEL['P3']
    num_final_anchors_p4 = NUM_FINAL_ANCHORS_PER_LEVEL['P4']
    num_final_anchors_p5 = NUM_FINAL_ANCHORS_PER_LEVEL['P5']

    k_range = K_RANGE_FOR_ANALYSIS
    output_graph_dir = OUTPUT_GRAPHS_DIR
    final_plot_save_path = output_graph_dir / DEFAULT_FINAL_GRAPH_FILENAME

    print(f"\nИспользуются данные из ОБУЧАЮЩЕЙ ВЫБОРКИ ({DATASET_BASE_PATH} / {TRAIN_ANNOTATIONS_SUBDIR}) для расчета якорей.")
    print(f"Целевой входной размер изображения для модели: {input_shape[1]}x{input_shape[0]}.")
    print(f"Классы для рассмотрения: {classes_list}")
    print(f"Диапазон K для анализа (для каждой группы FPN): {k_range}")
    print(f"Предполагаемое количество финальных якорей для каждой группы (из конфига): P3={num_final_anchors_p3}, P4={num_final_anchors_p4}, P5={num_final_anchors_p5}")
    print(f"Диапазоны нормализованных площадей для разделения на группы FPN (из конфига): {area_ranges_normalized}")

    if not Path(annotations_dir_train).is_dir():
        print(f"\nОШИБКА: Директория аннотаций не найдена: {annotations_dir_train}")
        sys.exit("Не найдены данные для расчета якорей. Выход.")
    if not Path(images_dir_train).is_dir():
         print(f"\nПРЕДУПРЕЖДЕНИЕ: Директория изображений не найдена: {images_dir_train}. Размеры будут читаться только из XML.")
         # Продолжаем, но предупреждаем пользователя

    # --- Парсинг всех нормализованных WxH bbox ---
    all_gt_boxes_wh_norm = parse_annotations_for_wh(annotations_dir_train, images_dir_train, classes_list)
    if all_gt_boxes_wh_norm.shape[0] == 0:
        print("Не найдено BBox для анализа. Проверьте датасет и классы в конфиге.")
        sys.exit("Нет BBox для анализа якорей. Выход.")

    print(f"Найдено {all_gt_boxes_wh_norm.shape[0]} нормализованных BBox для анализа.")

    # --- Разделение BBox на группы по нормализованной площади ---
    areas_norm = all_gt_boxes_wh_norm[:, 0] * all_gt_boxes_wh_norm[:, 1]

    # Построение гистограммы площадей
    plt.figure(figsize=(10, 6));
    plt.hist(areas_norm, bins=50, edgecolor='black');
    plt.title(f'Гистограмма Нормализованных Площадей BBox (Total: {all_gt_boxes_wh_norm.shape[0]})');
    plt.xlabel('Нормализованная Площадь (Area / Image_Area)');
    plt.ylabel('Количество')
    # Добавляем линии порогов из конфига
    colors = ['r', 'g'] # Цвета для порогов между P3/P4 и P4/P5
    labels = ['P3/P4 Threshold', 'P4/P5 Threshold']
    threshold_values = [area_ranges_normalized[0][1], area_ranges_normalized[1][1]]
    for i, thresh in enumerate(threshold_values):
        if i < len(colors): # Убедимся, что есть цвет для порога
             plt.axvline(thresh, color=colors[i], linestyle='dashed', linewidth=1, label=f'{labels[i]}={thresh:.4f}')
    plt.legend();
    plt.grid(True, linestyle='--');
    hist_save_path = output_graph_dir / "gt_box_areas_hist_thresholds.png"
    plt.savefig(hist_save_path);
    plt.close()
    print(f"Гистограмма нормализованных площадей сохранена: {hist_save_path}")


    boxes_by_level = {'P3': [], 'P4': [], 'P5': []}
    level_names = ['P3', 'P4', 'P5']

    for i_b, wh_n_b in enumerate(all_gt_boxes_wh_norm):
        area_cb = areas_norm[i_b]
        assigned_level = None
        for i, (min_area, max_area) in enumerate(area_ranges_normalized):
            if min_area <= area_cb < max_area:
                assigned_level = level_names[i]
                break
        if assigned_level:
            boxes_by_level[assigned_level].append(wh_n_b)
        # else: BBox не попал ни в один диапазон (крайне маловероятно при правильных диапазонах)

    boxes_p3_np = np.array(boxes_by_level['P3'])
    boxes_p4_np = np.array(boxes_by_level['P4'])
    boxes_p5_np = np.array(boxes_by_level['P5'])

    print(f"\nРазделение GT рамок по нормализованным порогам площадей из конфига:")
    print(f"  Группа P3 (площадь < {area_ranges_normalized[0][1]:.4f}): {boxes_p3_np.shape[0]} рамок")
    print(f"  Группа P4 ({area_ranges_normalized[0][1]:.4f} <= площадь < {area_ranges_normalized[1][1]:.4f}): {boxes_p4_np.shape[0]} рамок")
    print(f"  Группа P5 (площадь >= {area_ranges_normalized[1][1]:.4f}): {boxes_p5_np.shape[0]} рамок")


    # --- Анализ K-Means для каждой группы ---
    wcss_results, iou_results, anchors_results_dict, valid_k_ranges = {}, {}, {}, {}

    wcss_results['P3'], iou_results['P3'], anchors_results_dict['P3'], valid_k_ranges['P3'] = run_kmeans_analysis_for_group(boxes_p3_np, k_range, "P3")
    wcss_results['P4'], iou_results['P4'], anchors_results_dict['P4'], valid_k_ranges['P4'] = run_kmeans_analysis_for_group(boxes_p4_np, k_range, "P4")
    wcss_results['P5'], iou_results['P5'], anchors_results_dict['P5'], valid_k_ranges['P5'] = run_kmeans_analysis_for_group(boxes_p5_np, k_range, "P5")

    # Построение графиков для каждой группы
    if 'P3' in valid_k_ranges and wcss_results['P3']: plot_elbow_and_iou(valid_k_ranges['P3'], wcss_results['P3'], iou_results['P3'], "P3", output_graph_dir)
    if 'P4' in valid_k_ranges and wcss_results['P4']: plot_elbow_and_iou(valid_k_ranges['P4'], wcss_results['P4'], iou_results['P4'], "P4", output_graph_dir)
    if 'P5' in valid_k_ranges and wcss_results['P5']: plot_elbow_and_iou(valid_k_ranges['P5'], wcss_results['P5'], iou_results['P5'], "P5", output_graph_dir)

    # --- Вывод предложенных якорей для конфига ---
    print("\n\n--- Предложенные якоря для detector_config_v3_standard.yaml ---")
    print(f"# Целевой размер изображения: {input_shape[1]}x{input_shape[0]}")
    print("# Выберите подходящее количество якорей K для каждого уровня FPN на основе графиков 'Локтя' и Avg IoU.")
    print("# Затем скопируйте соответствующие нормализованные (W_norm, H_norm) якоря в конфиг в 'anchor_scales' (как базовые размеры).")
    print("# Убедитесь, что 'num_anchors_per_level' соответствует произведению количества масштабов (scales) и соотношений сторон (ratios), которые вы выберете.")
    print("# Например, если вы выбираете 3 якоря с нормированными WxH [w1, h1], [w2, h2], [w3, h3], и хотите использовать их как базовые размеры (scales), и задать ratios [0.5, 1.0, 2.0], то итоговое количество якорей на уровне будет 3 * 3 = 9.")
    print("# Более простой вариант для начала: используйте выбранные K якорей как базовые scales, и оставьте ratios [1.0], тогда num_anchors_per_level = K.")

    # Уровни FPN и соответствующие предполагаемые страйды
    level_info = {'P3': {'stride': 8}, 'P4': {'stride': 16}, 'P5': {'stride': 32}}


    for level_name in ['P3', 'P4', 'P5']:
        print(f"\n# --- Якоря для {level_name} (предполагаемый stride {level_info[level_name]['stride']}) ---")
        level_anchors_results = anchors_results_dict.get(level_name, {})
        level_iou_results = iou_results.get(level_name, {})
        level_valid_ks = valid_k_ranges.get(level_name, [])

        if not level_valid_ks:
            print(f"# Нет данных для {level_name}. Невозможно рассчитать якоря.")
            continue

        # Сортируем K для вывода
        sorted_ks = sorted(level_valid_ks)

        for k_val in sorted_ks:
            anchors_norm = level_anchors_results.get(k_val)
            avg_iou = level_iou_results.get(k_val, 0.0)

            if anchors_norm is not None and anchors_norm.size > 0:
                print(f"  # Для K = {k_val} якорей (AvgIoU: {avg_iou:.4f}):")
                print(f"  # num_anchors_per_level: {k_val} (если использовать только эти K якорей без дополнительных ratios)")
                print(f"  # anchor_scales:")
                # Выводим нормализованные WxH и примерные пиксельные WxH для целевого input_shape
                for i, anchor_norm in enumerate(anchors_norm):
                     anchor_w_px = anchor_norm[0] * input_shape[1] # input_shape: [H, W, C]
                     anchor_h_px = anchor_norm[1] * input_shape[0]
                     print(f"  #   - [{anchor_norm[0]:.6f}, {anchor_norm[1]:.6f}] # Примерно {anchor_w_px:.2f}x{anchor_h_px:.2f} px")
                print(f"  # anchor_ratios: [0.5, 1.0, 2.0] # Типичные соотношения сторон (или выберите другие/оставьте [1.0])")
                print(f"  # Если использовать K={k_val} якорей как scales И ratios [0.5, 1.0, 2.0], то num_anchors_per_level = {k_val} * 3 = {k_val*3}")
            else:
                print(f"  # Нет рассчитанных якорей для K = {k_val}")


    # --- Построение финального графика с выбранным количеством якорей из конфига ---
    final_anchors_p3 = anchors_results_dict['P3'].get(num_final_anchors_p3, np.array([]))
    final_anchors_p4 = anchors_results_dict['P4'].get(num_final_anchors_p4, np.array([]))
    final_anchors_p5 = anchors_results_dict['P5'].get(num_final_anchors_p5, np.array([]))

    final_anchors_for_plot = {
        'P3': final_anchors_p3,
        'P4': final_anchors_p4,
        'P5': final_anchors_p5
    }

    plot_final_anchors(all_gt_boxes_wh_norm, final_anchors_for_plot, final_plot_save_path, input_shape)

    print("\n--- Расчет и Анализ Якорей завершен ---")
    print(f"Графики WCSS/AvgIoU и финального распределения сохранены в: {output_graph_dir}")
    print("Пожалуйста, проанализируйте графики и выберите окончательное количество и размеры якорей для каждого уровня FPN, затем обновите 'detector_config_v3_standard.yaml'.")


if __name__ == "__main__":
    main()