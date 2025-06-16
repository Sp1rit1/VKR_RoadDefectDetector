# calculate_anchors.py
import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import yaml
import os
from pathlib import Path
import random  # Добавлен импорт random

# --- Загрузка Конфигураций ---
_current_script_root = Path(__file__).resolve().parent
_src_path_calc = _current_script_root / 'src'
if not _src_path_calc.exists():
    _src_path_calc = _current_script_root
    _base_config_path_calc = _current_script_root / 'configs' / 'base_config.yaml'
    _detector_config_path_calc = _current_script_root / 'configs' / 'detector_config.yaml'
    if not _base_config_path_calc.exists():
        _base_config_path_calc = _current_script_root / 'src' / 'configs' / 'base_config.yaml'
        _detector_config_path_calc = _current_script_root / 'src' / 'configs' / 'detector_config.yaml'
else:
    _base_config_path_calc = _src_path_calc / 'configs' / 'base_config.yaml'
    _detector_config_path_calc = _src_path_calc / 'configs' / 'detector_config.yaml'

BASE_CONFIG_CALC = {}
DETECTOR_CONFIG_CALC = {}
CONFIG_LOAD_SUCCESS_CALC = True
try:
    with open(_base_config_path_calc, 'r', encoding='utf-8') as f:
        BASE_CONFIG_CALC = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG_CALC, dict): BASE_CONFIG_CALC = {}; CONFIG_LOAD_SUCCESS_CALC = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_CALC = False; print(f"ERROR loading base_config: {_base_config_path_calc}")
except Exception as e:
    CONFIG_LOAD_SUCCESS_CALC = False; print(f"ERROR parsing base_config: {e}")

try:
    with open(_detector_config_path_calc, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG_CALC = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG_CALC, dict): DETECTOR_CONFIG_CALC = {}; CONFIG_LOAD_SUCCESS_CALC = False
except FileNotFoundError:
    CONFIG_LOAD_SUCCESS_CALC = False; print(f"ERROR loading detector_config: {_detector_config_path_calc}")
except Exception as e:
    CONFIG_LOAD_SUCCESS_CALC = False; print(f"ERROR parsing detector_config: {e}")

if not CONFIG_LOAD_SUCCESS_CALC:
    print("ОШИБКА: Не удалось загрузить конфиги. Используются дефолты для calculate_anchors.py.")
    BASE_CONFIG_CALC.setdefault('dataset', {'images_dir': 'JPEGImages', 'annotations_dir': 'Annotations'})
    DETECTOR_CONFIG_CALC.setdefault('input_shape', [416, 416, 3])
    DETECTOR_CONFIG_CALC.setdefault('fpn_anchor_configs', {
        'P3': {'num_anchors_this_level': 3, 'stride': 8},
        'P4': {'num_anchors_this_level': 3, 'stride': 16},
        'P5': {'num_anchors_this_level': 3, 'stride': 32}
    })
    DETECTOR_CONFIG_CALC.setdefault('classes', ['pit', 'crack'])
    DETECTOR_CONFIG_CALC.setdefault('anchor_calc_params', {'area_thresh_p3_end': 0.01, 'area_thresh_p4_end': 0.09})

# --- Глобальные Параметры из Конфигов (для удобства доступа в функциях) ---
IMAGES_SUBFOLDER_NAME_GLOBAL_CFG = BASE_CONFIG_CALC.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBFOLDER_NAME_GLOBAL_CFG = BASE_CONFIG_CALC.get('dataset', {}).get('annotations_dir', 'Annotations')
CLASSES_LIST_CALC = DETECTOR_CONFIG_CALC.get('classes', ['pit', 'crack'])

FPN_ANCHOR_CFGS_CALC = DETECTOR_CONFIG_CALC.get('fpn_anchor_configs', {})
NUM_ANCHORS_P3_CFG = FPN_ANCHOR_CFGS_CALC.get('P3', {}).get('num_anchors_this_level', 3)
NUM_ANCHORS_P4_CFG = FPN_ANCHOR_CFGS_CALC.get('P4', {}).get('num_anchors_this_level', 3)
NUM_ANCHORS_P5_CFG = FPN_ANCHOR_CFGS_CALC.get('P5', {}).get('num_anchors_this_level', 3)
STRIDE_P3_CFG = FPN_ANCHOR_CFGS_CALC.get('P3', {}).get('stride', 8)
STRIDE_P4_CFG = FPN_ANCHOR_CFGS_CALC.get('P4', {}).get('stride', 16)
STRIDE_P5_CFG = FPN_ANCHOR_CFGS_CALC.get('P5', {}).get('stride', 32)

INPUT_HEIGHT_CALC = DETECTOR_CONFIG_CALC.get('input_shape', [416, 416, 3])[0]
INPUT_WIDTH_CALC = DETECTOR_CONFIG_CALC.get('input_shape', [416, 416, 3])[1]

AREA_THRESH_P3_END_PARAM_CFG = DETECTOR_CONFIG_CALC.get('anchor_calc_params', {}).get('area_thresh_p3_end', 0.01)
AREA_THRESH_P4_END_PARAM_CFG = DETECTOR_CONFIG_CALC.get('anchor_calc_params', {}).get('area_thresh_p4_end', 0.09)

OUTPUT_GRAPHS_DIR_PARAM_CFG = _current_script_root / "graphs" / "anchor_analysis"
OUTPUT_GRAPHS_DIR_PARAM_CFG.mkdir(parents=True, exist_ok=True)
DEFAULT_FINAL_GRAPH_FILENAME = "fpn_anchors_distribution_final.png"
DEFAULT_K_RANGE = range(1, 8)


def parse_annotations_for_wh(annotations_dir_arg, images_dir_for_size_fallback_arg, classes_to_consider_arg=None):
    all_boxes_wh_norm = [];
    print(f"Сканирование аннотаций в: {annotations_dir_arg}")
    xml_files = glob.glob(os.path.join(annotations_dir_arg, "*.xml"))
    if not xml_files: print(f"  ПРЕДУПРЕЖДЕНИЕ: XML файлы не найдены в {annotations_dir_arg}"); return np.array(
        all_boxes_wh_norm)
    for xml_file in xml_files:
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
                        for ext_try_fn in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                            if (Path(images_dir_for_size_fallback_arg) / (base_fn + ext_try_fn)).exists():
                                img_path_cand = Path(images_dir_for_size_fallback_arg) / (base_fn + ext_try_fn);
                                break
                    if img_path_cand.exists():
                        try:
                            from PIL import Image;img_pil = Image.open(
                                img_path_cand);img_w_xml, img_h_xml = img_pil.size;img_pil.close()
                        except:
                            print(f"    Не удалось прочитать размеры из {img_path_cand}"); continue
                    else:
                        print(f"    Файл для {xml_file} не найден для размеров. Пропуск."); continue
                else:
                    print(f"    Тег <filename> или <size> не найден в {xml_file}. Пропуск."); continue
            for obj_node in root.findall('object'):
                cls_nm_node = obj_node.find('name')
                if cls_nm_node is None or cls_nm_node.text is None: continue
                if classes_to_consider_arg and cls_nm_node.text not in classes_to_consider_arg: continue
                bndb_node = obj_node.find('bndbox')
                if bndb_node is None: continue
                try:
                    xmin, ymin, xmax, ymax = (float(bndb_node.find(t).text) for t in ['xmin', 'ymin', 'xmax', 'ymax'])
                except:
                    continue
                if xmin >= xmax or ymin >= ymax: continue
                all_boxes_wh_norm.append([(xmax - xmin) / float(img_w_xml), (ymax - ymin) / float(img_h_xml)])
        except:
            print(f"  Ошибка обработки {xml_file}")
    return np.array(all_boxes_wh_norm)


def iou_for_kmeans(boxes_arg, clusters_arg):
    n = boxes_arg.shape[0];
    k = clusters_arg.shape[0]
    box_area_calc = boxes_arg[:, 0] * boxes_arg[:, 1];
    cluster_area_calc = clusters_arg[:, 0] * clusters_arg[:, 1]
    box_area_reshaped_calc = np.repeat(box_area_calc, k).reshape(n, k);
    cluster_area_reshaped_calc = np.tile(cluster_area_calc, n).reshape(n, k)
    intersection_w_calc = np.minimum(np.repeat(boxes_arg[:, 0], k).reshape(n, k),
                                     np.tile(clusters_arg[:, 0], n).reshape(n, k))
    intersection_h_calc = np.minimum(np.repeat(boxes_arg[:, 1], k).reshape(n, k),
                                     np.tile(clusters_arg[:, 1], n).reshape(n, k))
    intersection_area_calc = intersection_w_calc * intersection_h_calc
    union_area_calc = box_area_reshaped_calc + cluster_area_reshaped_calc - intersection_area_calc
    return intersection_area_calc / (union_area_calc + 1e-6)


def avg_iou_calc(boxes_arg, clusters_arg):
    if clusters_arg.shape[0] == 0 or boxes_arg.shape[0] == 0: return 0.0
    return np.mean(np.max(iou_for_kmeans(boxes_arg, clusters_arg), axis=1))


def run_kmeans_analysis_for_group(boxes_wh_group_arg, k_range_arg, group_name_arg):
    print(f"\nАнализ якорей для '{group_name_arg}' (найдено {boxes_wh_group_arg.shape[0]} рамок):")
    if boxes_wh_group_arg.shape[0] == 0: print("  Нет рамок.");return {}, {}, {}, []
    wcss_res, avg_iou_vals_res, anchors_for_k_res = {}, {}, {};
    valid_k_range_res = []
    for k_val_arg in k_range_arg:
        if boxes_wh_group_arg.shape[0] < k_val_arg: print(f"  K={k_val_arg}: Пропуск (мало рамок)"); continue
        valid_k_range_res.append(k_val_arg)
        kmeans = KMeans(n_clusters=k_val_arg, random_state=42, n_init='auto');
        kmeans.fit(boxes_wh_group_arg)
        wcss_res[k_val_arg] = kmeans.inertia_
        curr_anchors = kmeans.cluster_centers_;
        curr_anchors = curr_anchors[np.argsort(curr_anchors[:, 0] * curr_anchors[:, 1])]
        anchors_for_k_res[k_val_arg] = curr_anchors
        avg_iou_vals_res[k_val_arg] = avg_iou_calc(boxes_wh_group_arg, curr_anchors)
        print(f"  K={k_val_arg}: WCSS={wcss_res[k_val_arg]:.2f}, AvgIoU={avg_iou_vals_res[k_val_arg]:.4f}")
    return wcss_res, avg_iou_vals_res, anchors_for_k_res, valid_k_range_res


def plot_elbow_and_iou(k_range_plot_arg, wcss_dict_arg, avg_iou_dict_arg, group_name_plot_arg, save_dir_arg):
    if not k_range_plot_arg or not wcss_dict_arg or not avg_iou_dict_arg: print(
        f"Нет данных для графика {group_name_plot_arg}"); return
    wcss_list_plot = [wcss_dict_arg[k_item] for k_item in k_range_plot_arg if k_item in wcss_dict_arg]
    avg_iou_list_plot = [avg_iou_dict_arg[k_item] for k_item in k_range_plot_arg if k_item in avg_iou_dict_arg]
    actual_k_range_plot = [k_item for k_item in k_range_plot_arg if k_item in wcss_dict_arg]
    if not actual_k_range_plot: print(f"Нет валидных K для графика {group_name_plot_arg}"); return
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
    plt.savefig(save_dir_arg / f"elbow_iou_anchors_{group_name_plot_arg}.png");
    print(f"График для {group_name_plot_arg} сохранен.")
    plt.close()


def plot_final_anchors(all_boxes_wh_norm_plot_arg, final_anchors_p3_plot, final_anchors_p4_plot, final_anchors_p5_plot,
                       save_path_plot_arg):
    plt.figure(figsize=(12, 9));
    plt.scatter(all_boxes_wh_norm_plot_arg[:, 0], all_boxes_wh_norm_plot_arg[:, 1], alpha=0.3,
                label='Ground Truth Boxes (W_norm, H_norm)')
    colors_plot = {'P3': 'red', 'P4': 'green', 'P5': 'purple'};
    markers_plot = {'P3': 'x', 'P4': 's', 'P5': '^'}

    # Используем глобальные _CFG переменные для легенды, так как они отражают текущие настройки из конфига
    num_p3_plot = NUM_ANCHORS_P3_CFG;
    num_p4_plot = NUM_ANCHORS_P4_CFG;
    num_p5_plot = NUM_ANCHORS_P5_CFG;
    stride_p3_plot = STRIDE_P3_CFG;
    stride_p4_plot = STRIDE_P4_CFG;
    stride_p5_plot = STRIDE_P5_CFG;

    if final_anchors_p3_plot.size > 0: plt.scatter(final_anchors_p3_plot[:, 0], final_anchors_p3_plot[:, 1],
                                                   color=colors_plot['P3'], marker=markers_plot['P3'], s=100,
                                                   label=f'{final_anchors_p3_plot.shape[0]} K-Means Anchors P3 (stride {stride_p3_plot})')  # Используем shape[0] для фактического числа
    if final_anchors_p4_plot.size > 0: plt.scatter(final_anchors_p4_plot[:, 0], final_anchors_p4_plot[:, 1],
                                                   color=colors_plot['P4'], marker=markers_plot['P4'], s=100,
                                                   label=f'{final_anchors_p4_plot.shape[0]} K-Means Anchors P4 (stride {stride_p4_plot})')
    if final_anchors_p5_plot.size > 0: plt.scatter(final_anchors_p5_plot[:, 0], final_anchors_p5_plot[:, 1],
                                                   color=colors_plot['P5'], marker=markers_plot['P5'], s=100,
                                                   label=f'{final_anchors_p5_plot.shape[0]} K-Means Anchors P5 (stride {stride_p5_plot})')

    plt.xlabel('Нормализованная Ширина (W_norm)');
    plt.ylabel('Нормализованная Высота (H_norm)')
    plt.title(f'Распределение Размеров BBox и Финальных Якорей для FPN');
    plt.xlim(0, 1.05);
    plt.ylim(0, 1.05);
    plt.grid(True, linestyle='--');
    plt.legend()
    plt.savefig(save_path_plot_arg);
    print(f"Финальный график якорей сохранен: {save_path_plot_arg}");
    plt.close()


def main_calculate_anchors_entry_point(annotations_dir_arg, images_dir_arg, classes_list_arg,
                                       area_thresh_p3_arg, area_thresh_p4_arg,
                                       k_range_list_arg,  # Один общий диапазон K для всех уровней
                                       num_final_anchors_p3_arg, num_final_anchors_p4_arg, num_final_anchors_p5_arg,
                                       output_graph_dir_arg, final_plot_save_path_arg):
    print("--- Расчет и Анализ Якорей для FPN ---")
    all_gt_boxes_wh_norm_calc = parse_annotations_for_wh(annotations_dir_arg, images_dir_arg, classes_list_arg)
    if all_gt_boxes_wh_norm_calc.shape[0] == 0: print("Не найдено BBox."); return
    print(f"Найдено {all_gt_boxes_wh_norm_calc.shape[0]} BBox для анализа.")

    areas_norm_calc = all_gt_boxes_wh_norm_calc[:, 0] * all_gt_boxes_wh_norm_calc[:, 1]
    plt.figure(figsize=(10, 6));
    plt.hist(areas_norm_calc, bins=50, edgecolor='black');
    plt.title('Гистограмма Норм. Площадей BBox');
    plt.xlabel('Норм. Площадь');
    plt.ylabel('Количество')
    plt.axvline(area_thresh_p3_arg, color='r', linestyle='dashed', linewidth=1,
                label=f'P3_end={area_thresh_p3_arg:.3f}')
    plt.axvline(area_thresh_p4_arg, color='g', linestyle='dashed', linewidth=1,
                label=f'P4_end={area_thresh_p4_arg:.3f}')
    plt.legend();
    plt.grid(True, linestyle='--');
    plt.savefig(output_graph_dir_arg / "gt_box_areas_hist_thresholds.png");
    plt.close()
    print(f"Гистограмма площадей сохранена.")

    boxes_p3_g_list_calc, boxes_p4_g_list_calc, boxes_p5_g_list_calc = [], [], []
    for i_b_calc, wh_n_b_calc in enumerate(all_gt_boxes_wh_norm_calc):
        area_cb_calc = areas_norm_calc[i_b_calc]
        if area_cb_calc < area_thresh_p3_arg:
            boxes_p3_g_list_calc.append(wh_n_b_calc)
        elif area_cb_calc < area_thresh_p4_arg:
            boxes_p4_g_list_calc.append(wh_n_b_calc)
        else:
            boxes_p5_g_list_calc.append(wh_n_b_calc)
    boxes_p3_g_np_calc, boxes_p4_g_np_calc, boxes_p5_g_np_calc = np.array(boxes_p3_g_list_calc), np.array(
        boxes_p4_g_list_calc), np.array(boxes_p5_g_list_calc)

    print(f"\nРазделение GT рамок по вычисленным порогам:")
    print(f"  Группа P3 (<{area_thresh_p3_arg:.3f}): {boxes_p3_g_np_calc.shape[0]} рамок")
    print(f"  Группа P4 ({area_thresh_p3_arg:.3f}-<{area_thresh_p4_arg:.3f}): {boxes_p4_g_np_calc.shape[0]} рамок")
    print(f"  Группа P5 (>={area_thresh_p4_arg:.3f}): {boxes_p5_g_np_calc.shape[0]} рамок")

    wcss_p3, iou_p3, anchors_p3_k_dict, valid_k_p3 = run_kmeans_analysis_for_group(boxes_p3_g_np_calc, k_range_list_arg,
                                                                                   "P3")
    wcss_p4, iou_p4, anchors_p4_k_dict, valid_k_p4 = run_kmeans_analysis_for_group(boxes_p4_g_np_calc, k_range_list_arg,
                                                                                   "P4")
    wcss_p5, iou_p5, anchors_p5_k_dict, valid_k_p5 = run_kmeans_analysis_for_group(boxes_p5_g_np_calc, k_range_list_arg,
                                                                                   "P5")

    if wcss_p3: plot_elbow_and_iou(valid_k_p3, wcss_p3, iou_p3, "P3", output_graph_dir_arg)
    if wcss_p4: plot_elbow_and_iou(valid_k_p4, wcss_p4, iou_p4, "P4", output_graph_dir_arg)
    if wcss_p5: plot_elbow_and_iou(valid_k_p5, wcss_p5, iou_p5, "P5", output_graph_dir_arg)

    print("\n--- Предлагаемые якоря (W_norm, H_norm) на основе анализа ---")
    for group_name_calc, k_results_dict_calc, iou_results_dict_calc_print, k_vals_list_calc in zip(
            ["P3", "P4", "P5"],
            [anchors_p3_k_dict, anchors_p4_k_dict, anchors_p5_k_dict],
            [iou_p3, iou_p4, iou_p5],
            [valid_k_p3, valid_k_p4, valid_k_p5]):
        print(f"  {group_name_calc}:")
        if not k_results_dict_calc: print("    # Нет якорей."); continue
        for k_val_res_loop_calc in k_vals_list_calc:
            anchors_res_list_loop_calc = k_results_dict_calc.get(k_val_res_loop_calc)
            avg_iou_val_loop_calc = iou_results_dict_calc_print.get(k_val_res_loop_calc, 0.0)
            print(f"    # Для K = {k_val_res_loop_calc} (AvgIoU: {avg_iou_val_loop_calc:.4f}):")
            print(f"    # num_anchors_this_level: {k_val_res_loop_calc}")
            print(f"    # anchors_wh_normalized:")
            if anchors_res_list_loop_calc is not None and anchors_res_list_loop_calc.size > 0:
                for a_res_loop_calc in anchors_res_list_loop_calc: print(
                    f"    #   - [{a_res_loop_calc[0]:.4f}, {a_res_loop_calc[1]:.4f}]")
            else:
                print("    #   # Нет якорей")
            current_stride_calc = globals().get(f'STRIDE_{group_name_calc}_CFG', 'N/A')
            print(f"    # stride: {current_stride_calc}\n")

    final_anchors_p3_plot = anchors_p3_k_dict.get(num_final_anchors_p3_arg, np.array([]))
    final_anchors_p4_plot = anchors_p4_k_dict.get(num_final_anchors_p4_arg, np.array([]))
    final_anchors_p5_plot = anchors_p5_k_dict.get(num_final_anchors_p5_arg, np.array([]))
    plot_final_anchors(all_gt_boxes_wh_norm_calc, final_anchors_p3_plot, final_anchors_p4_plot, final_anchors_p5_plot,
                       final_plot_save_path_arg)


if __name__ == "__main__":
    print("--- Запуск скрипта calculate_anchors.py ---")
    if not CONFIG_LOAD_SUCCESS_CALC:
        print("ОШИБКА: Конфигурационные файлы не были загружены. Выход.");
        exit()

    # --- Определяем пути и параметры для функции main_calculate_anchors_entry_point ---
    _detector_dataset_ready_path_rel_main_block = DETECTOR_CONFIG_CALC.get('prepared_detector_dataset_path',
                                                                           "data/Detector_Dataset_Ready")
    _detector_dataset_ready_abs_main_block = (
                _current_script_root / _detector_dataset_ready_path_rel_main_block).resolve()

    annotations_dir_train_call = str(
        _detector_dataset_ready_abs_main_block / "train" / ANNOTATIONS_SUBFOLDER_NAME_GLOBAL_CFG)
    images_dir_train_call = str(_detector_dataset_ready_abs_main_block / "train" / IMAGES_SUBFOLDER_NAME_GLOBAL_CFG)

    classes_list_call = CLASSES_LIST_CALC
    num_final_anchors_p3_call = NUM_ANCHORS_P3_CFG
    num_final_anchors_p4_call = NUM_ANCHORS_P4_CFG
    num_final_anchors_p5_call = NUM_ANCHORS_P5_CFG

    area_thresh_p3_call = AREA_THRESH_P3_END_PARAM_CFG
    area_thresh_p4_call = AREA_THRESH_P4_END_PARAM_CFG

    k_range_call = list(DEFAULT_K_RANGE)
    output_graph_dir_call = OUTPUT_GRAPHS_DIR_PARAM_CFG
    final_plot_save_path_call = output_graph_dir_call / DEFAULT_FINAL_GRAPH_FILENAME

    print(f"\nИспользуются данные из ОБУЧАЮЩЕЙ ВЫБОРКИ для расчета якорей:")
    print(f"  Директория аннотаций (XML): {annotations_dir_train_call}")
    print(f"  Директория изображений (для получения размеров): {images_dir_train_call}")
    print(f"Диапазон K для анализа (для каждого уровня FPN): {k_range_call}")
    print(
        f"Пороги площадей для разделения на группы (из конфига): P3_end={area_thresh_p3_call:.4f}, P4_end={area_thresh_p4_call:.4f}")
    print(
        f"Количество якорей для финального графика (из конфига): P3={num_final_anchors_p3_call}, P4={num_final_anchors_p4_call}, P5={num_final_anchors_p5_call}")

    if Path(annotations_dir_train_call).is_dir() and Path(images_dir_train_call).is_dir():
        main_calculate_anchors_entry_point(
            annotations_dir_train_call,
            images_dir_train_call,
            classes_list_call,
            area_thresh_p3_call,
            area_thresh_p4_call,
            k_range_call,  # <--- ИЗМЕНЕНИЕ: Передаем k_range_call ОДИН РАЗ
            num_final_anchors_p3_call,
            num_final_anchors_p4_call,
            num_final_anchors_p5_call,
            output_graph_dir_call,
            final_plot_save_path_call
        )
    else:
        print("\nОШИБКА: Директории для обучающих данных не найдены.")
        print(f"  Ожидалась Annotations: {annotations_dir_train_call}")
        print(f"  Ожидалась JPEGImages: {images_dir_train_call}")

    print("\n--- Скрипт calculate_anchors.py завершил работу ---")