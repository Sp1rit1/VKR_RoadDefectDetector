# RoadDefectDetector/analyze_fpn_assignments.py
import xml.etree.ElementTree as ET
import numpy as np
import os
import glob
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# --- Загрузка Конфигураций ---
_project_root_analyzer = Path(__file__).resolve().parent
_src_path_analyzer = _project_root_analyzer / 'src'
import sys

if str(_src_path_analyzer) not in sys.path:
    sys.path.insert(0, str(_src_path_analyzer))

try:
    from datasets.detector_data_loader import calculate_iou_numpy  # Нам нужна функция IoU
except ImportError:
    print("ОШИБКА: Не удалось импортировать calculate_iou_numpy из detector_data_loader.")
    print("Убедитесь, что detector_data_loader.py находится в src/datasets/ и не содержит ошибок импорта.")


    # Заглушка, если импорт не удался, но скрипт упадет при вызове
    def calculate_iou_numpy(box_wh, anchors_wh):
        return np.zeros(anchors_wh.shape[0])


    exit()


def load_config_analyzer(config_path_obj, config_name_for_log, default_on_error=None):
    if default_on_error is None: default_on_error = {}
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: {config_path_obj.name} пуст или неверный формат. Дефолты для '{config_name_for_log}'.")
            return default_on_error
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден: {config_path_obj}. Выход.");
        exit()
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML в {config_path_obj.name}: {e}. Выход.");
        exit()


_base_config_path_analyzer = _src_path_analyzer / 'configs' / 'base_config.yaml'
_detector_config_path_analyzer = _src_path_analyzer / 'configs' / 'detector_config.yaml'

print("--- Загрузка конфигурационных файлов для анализа FPN назначений ---")
BASE_CONFIG_ANALYZER = load_config_analyzer(_base_config_path_analyzer, "Base Config")
DETECTOR_CONFIG_ANALYZER = load_config_analyzer(_detector_config_path_analyzer, "Detector Config")

# --- Параметры из Конфигов ---
# Пути к обучающей выборке детектора
_detector_dataset_ready_path_rel_analyzer = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_ANALYZER = (_project_root_analyzer / _detector_dataset_ready_path_rel_analyzer).resolve()
TRAIN_ANNOT_DIR_ANALYZER = str(
    DETECTOR_DATASET_READY_ABS_ANALYZER / "train" / BASE_CONFIG_ANALYZER.get('dataset', {}).get('annotations_dir',
                                                                                                'Annotations'))

# Параметры FPN
_fpn_params_analyzer = DETECTOR_CONFIG_ANALYZER.get('fpn_detector_params', {})
FPN_LEVELS_ANALYZER = _fpn_params_analyzer.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
FPN_ANCHOR_CONFIGS_ANALYZER = _fpn_params_analyzer.get('detector_fpn_anchor_configs', {})
P3_END_AREA_THRESHOLD_ANALYZER = float(_fpn_params_analyzer.get('p3_end_area_threshold', 0.01))
P4_END_AREA_THRESHOLD_ANALYZER = float(_fpn_params_analyzer.get('p4_end_area_threshold', 0.09))
CLASSES_LIST_ANALYZER = DETECTOR_CONFIG_ANALYZER.get('classes', ['pit', 'crack'])


def analyze_gt_assignments():
    print(f"\nАнализ назначений GT объектов на уровни FPN из: {TRAIN_ANNOT_DIR_ANALYZER}")
    if not Path(TRAIN_ANNOT_DIR_ANALYZER).is_dir():
        print(f"ОШИБКА: Директория аннотаций для обучения не найдена: {TRAIN_ANNOT_DIR_ANALYZER}")
        return

    xml_files = glob.glob(os.path.join(TRAIN_ANNOT_DIR_ANALYZER, "*.xml"))
    if not xml_files:
        print(f"XML файлы не найдены в {TRAIN_ANNOT_DIR_ANALYZER}")
        return
    print(f"Найдено {len(xml_files)} XML файлов для анализа.")

    assignments_count = {level: 0 for level in FPN_LEVELS_ANALYZER}
    lost_gt_count = 0
    total_gt_objects = 0
    max_ious_per_level = {level: [] for level in FPN_LEVELS_ANALYZER}
    gt_areas_assigned = {level: [] for level in FPN_LEVELS_ANALYZER}
    gt_areas_lost = []

    for xml_file_path in xml_files:
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            size_node = root.find('size')
            if size_node is None: continue  # Пропускаем, если нет размеров
            img_width = int(size_node.find('width').text)
            img_height = int(size_node.find('height').text)
            if img_width == 0 or img_height == 0: continue

            for obj_node in root.findall('object'):
                total_gt_objects += 1
                class_name = obj_node.find('name').text
                if class_name not in CLASSES_LIST_ANALYZER:
                    # print(f"  Пропущен неизвестный класс {class_name} в {os.path.basename(xml_file_path)}")
                    continue  # Пропускаем объекты неизвестных классов

                bndbox_node = obj_node.find('bndbox')
                xmin = float(bndbox_node.find('xmin').text)
                ymin = float(bndbox_node.find('ymin').text)
                xmax = float(bndbox_node.find('xmax').text)
                ymax = float(bndbox_node.find('ymax').text)

                if xmin >= xmax or ymin >= ymax: continue

                # Нормализованные ширина и высота GT объекта
                gt_w_norm = (xmax - xmin) / img_width
                gt_h_norm = (ymax - ymin) / img_height
                gt_area_norm = gt_w_norm * gt_h_norm

                assigned_level = None
                if gt_area_norm <= P3_END_AREA_THRESHOLD_ANALYZER:
                    assigned_level = 'P3'
                elif P3_END_AREA_THRESHOLD_ANALYZER < gt_area_norm <= P4_END_AREA_THRESHOLD_ANALYZER:
                    assigned_level = 'P4'
                elif gt_area_norm > P4_END_AREA_THRESHOLD_ANALYZER:
                    assigned_level = 'P5'

                if assigned_level and assigned_level in FPN_LEVELS_ANALYZER:
                    assignments_count[assigned_level] += 1
                    gt_areas_assigned[assigned_level].append(gt_area_norm)

                    level_anchors_wh = np.array(FPN_ANCHOR_CONFIGS_ANALYZER[assigned_level]['anchors_wh_normalized'],
                                                dtype=np.float32)
                    gt_box_shape_wh_for_iou = [gt_w_norm, gt_h_norm]
                    ious_with_anchors = calculate_iou_numpy(gt_box_shape_wh_for_iou, level_anchors_wh)
                    if ious_with_anchors.size > 0:
                        max_ious_per_level[assigned_level].append(np.max(ious_with_anchors))
                else:
                    lost_gt_count += 1
                    gt_areas_lost.append(gt_area_norm)

        except Exception as e:
            print(f"Ошибка при обработке файла {xml_file_path}: {e}")
            continue

    print("\n--- Статистика Назначения GT Объектов на Уровни FPN ---")
    print(f"Всего обработано GT объектов (известных классов): {total_gt_objects}")
    for level, count in assignments_count.items():
        print(
            f"  Назначено на уровень {level}: {count} объектов ({count / total_gt_objects * 100 if total_gt_objects else 0:.1f}%)")
    print(
        f"  'Потеряно' GT объектов (не назначено ни одному уровню): {lost_gt_count} ({lost_gt_count / total_gt_objects * 100 if total_gt_objects else 0:.1f}%)")

    print("\n--- Статистика Максимального IoU между GT и Якорями на Назначенном Уровне ---")
    for level, ious_list in max_ious_per_level.items():
        if ious_list:
            print(f"  Уровень {level}:")
            print(f"    Количество сопоставленных GT: {len(ious_list)}")
            print(f"    Средний Max IoU: {np.mean(ious_list):.3f}")
            print(f"    Медианный Max IoU: {np.median(ious_list):.3f}")
            print(f"    Мин Max IoU: {np.min(ious_list):.3f}")
            print(f"    Макс Max IoU: {np.max(ious_list):.3f}")
            plt.figure(figsize=(8, 4))
            plt.hist(ious_list, bins=20, range=(0, 1))
            plt.title(f"Гистограмма Max IoU для уровня {level}")
            plt.xlabel("Max IoU с якорями уровня")
            plt.ylabel("Количество GT объектов")
            plt.grid(True)
            plot_save_path_iou = _project_root_analyzer / "graphs" / f"max_iou_hist_{level}.png"
            plt.savefig(plot_save_path_iou)
            print(f"    Гистограмма сохранена: {plot_save_path_iou}")
            plt.close()
        else:
            print(f"  Уровень {level}: Нет сопоставленных GT объектов для анализа IoU.")

    if gt_areas_lost:
        plt.figure(figsize=(8, 4))
        plt.hist(gt_areas_lost, bins=30, range=(
        0, max(0.2, max(gt_areas_lost) if gt_areas_lost else 0.2)))  # Ограничим диапазон для наглядности
        plt.title(f"Гистограмма площадей 'потерянных' GT объектов ({len(gt_areas_lost)} шт.)")
        plt.xlabel("Нормализованная площадь GT объекта")
        plt.ylabel("Количество")
        plt.grid(True)
        plot_save_path_lost = _project_root_analyzer / "graphs" / "lost_gt_areas_hist.png"
        plt.savefig(plot_save_path_lost)
        print(f"\nГистограмма площадей потерянных GT объектов сохранена: {plot_save_path_lost}")
        plt.close()

    print("\n--- Анализ завершен ---")


if __name__ == "__main__":
    analyze_gt_assignments()