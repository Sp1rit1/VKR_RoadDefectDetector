# RoadDefectDetector/analyze_fpn_assignments.py
import xml.etree.ElementTree as ET
import numpy as np
import os
import glob
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import random  # Добавим для --max_files_per_split

# --- Загрузка Конфигураций ---
_project_root_analyzer = Path(__file__).resolve().parent
_src_path_analyzer = _project_root_analyzer / 'src'
import sys

if str(_src_path_analyzer) not in sys.path:
    sys.path.insert(0, str(_src_path_analyzer))

# --- Импорты из твоего detector_data_loader ---
try:
    from datasets.detector_data_loader import (
        calculate_iou_numpy,
        parse_xml_annotation, # <<<--- ДОБАВЛЕН ИМПОРТ ЭТОЙ ФУНКЦИИ
        # Также импортируем глобальные переменные, которые могут понадобиться,
        # если parse_xml_annotation или другие функции на них полагаются
        # или если мы хотим использовать те же CLASSES_LIST, что и в data_loader.
        CLASSES_LIST_GLOBAL_FOR_DETECTOR as DDL_CLASSES_LIST # Переименуем, чтобы не было конфликта с CLASSES_LIST_ANALYZER
    )
    print("INFO: Функции calculate_iou_numpy и parse_xml_annotation успешно импортированы из detector_data_loader.")
    # Если CLASSES_LIST_ANALYZER должен быть точно таким же, как в data_loader, можно сделать так:
    # CLASSES_LIST_ANALYZER = DDL_CLASSES_LIST
except ImportError as e_imp:
    print(f"ОШИБКА: Не удалось импортировать компоненты из detector_data_loader: {e_imp}")
    print("Убедитесь, что detector_data_loader.py находится в src/datasets/ и не содержит ошибок импорта.")
    # Заглушки, если импорт не удался
    def calculate_iou_numpy(box_wh, anchors_wh):
        print("ПРЕДУПРЕЖДЕНИЕ: Используется ЗАГЛУШКА для calculate_iou_numpy!")
        return np.zeros(anchors_wh.shape[0])
    def parse_xml_annotation(xml_file_path, classes_list_arg): # Добавим classes_list_arg для заглушки
        print("ПРЕДУПРЕЖДЕНИЕ: Используется ЗАГЛУШКА для parse_xml_annotation!")
        return None, None, None, None
    DDL_CLASSES_LIST = ['pit', 'crack'] # Дефолт для заглушки
    # CLASSES_LIST_ANALYZER = DDL_CLASSES_LIST # Обновляем и здесь
    # exit() # Раскомментируй, если без этих функций скрипт не имеет смысла
except Exception as e_gen_imp:
    print(f"ОБЩАЯ ОШИБКА при импорте из detector_data_loader: {e_gen_imp}")
    # ... (те же заглушки)
    def calculate_iou_numpy(box_wh, anchors_wh): return np.zeros(anchors_wh.shape[0])
    def parse_xml_annotation(xml_file_path, classes_list_arg): return None, None, None, None
    DDL_CLASSES_LIST = ['pit', 'crack']



def load_config_analyzer_strict(config_path_obj, config_name_for_log):
    """Загружает конфиг, выходит из программы при критической ошибке."""
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            print(f"ОШИБКА: {config_path_obj.name} пуст или имеет неверный формат. Выход.")
            exit()
        print(f"INFO: Конфиг '{config_name_for_log}' успешно загружен из {config_path_obj.name}.")
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден по пути: {config_path_obj}. Выход.")
        exit()
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML в {config_path_obj.name}: {e}. Выход.")
        exit()


_base_config_path_analyzer = _src_path_analyzer / 'configs' / 'base_config.yaml'
_detector_config_path_analyzer = _src_path_analyzer / 'configs' / 'detector_config.yaml'

print("\n--- Загрузка конфигурационных файлов для анализа FPN назначений ---")
BASE_CONFIG_ANALYZER = load_config_analyzer_strict(_base_config_path_analyzer, "Base Config")
DETECTOR_CONFIG_ANALYZER = load_config_analyzer_strict(_detector_config_path_analyzer, "Detector Config")

# --- Параметры из Конфигов ---
# Пути к данным (будут выбираться train или validation ниже)
_detector_dataset_ready_path_rel_analyzer = "data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS_ANALYZER = (_project_root_analyzer / _detector_dataset_ready_path_rel_analyzer).resolve()
IMAGES_SUBDIR_CFG_ANALYZER = BASE_CONFIG_ANALYZER.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_CFG_ANALYZER = BASE_CONFIG_ANALYZER.get('dataset', {}).get('annotations_dir', 'Annotations')

# Параметры FPN
_fpn_params_analyzer = DETECTOR_CONFIG_ANALYZER.get('fpn_detector_params', {})
FPN_LEVEL_NAMES_ANALYZER = _fpn_params_analyzer.get('detector_fpn_levels', ['P3', 'P4', 'P5'])
FPN_ANCHOR_CONFIGS_ANALYZER = _fpn_params_analyzer.get('detector_fpn_anchor_configs', {})
CLASSES_LIST_ANALYZER = _fpn_params_analyzer.get('classes', ['pit', 'crack'])

# <<<--- ИСПОЛЬЗУЕМ fpn_gt_assignment_scale_ranges ИЗ КОНФИГА ---<<<
FPN_GT_ASSIGNMENT_SCALE_RANGES_ANALYZER = _fpn_params_analyzer.get('fpn_gt_assignment_scale_ranges', {
    'P3': [0, 64],
    'P4': [64, 128],
    'P5': [128, 100000]  # Большая верхняя граница
})
print("\nИспользуемые диапазоны масштабов для назначения GT на уровни FPN (пиксели оригинального изображения):")
for level_name_range, ranges in FPN_GT_ASSIGNMENT_SCALE_RANGES_ANALYZER.items():
    if level_name_range in FPN_LEVEL_NAMES_ANALYZER:  # Убедимся, что уровень есть в списке
        print(f"  {level_name_range}: от {ranges[0]} до {ranges[1]} px (по sqrt(площади) или большей стороне GT)")


# --- Конец Ключевого Изменения ---


def analyze_gt_assignments(split_name_for_log, annotations_dir_path_str, images_dir_path_str,
                           max_files_to_analyze=None):
    print(f"\n\n<<<<< АНАЛИЗ ДЛЯ ВЫБОРКИ: {split_name_for_log.upper()} >>>>>")
    print(f"  Директория аннотаций: {annotations_dir_path_str}")
    print(f"  Директория изображений: {images_dir_path_str}")

    annotations_dir = Path(annotations_dir_path_str)
    images_dir = Path(images_dir_path_str)

    if not annotations_dir.is_dir():
        print(f"ОШИБКА: Директория аннотаций не найдена: {annotations_dir}")
        return
    if not images_dir.is_dir():
        print(f"ОШИБКА: Директория изображений не найдена: {images_dir}")
        return

    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    if not xml_files:
        print(f"XML файлы не найдены в {annotations_dir}")
        return

    if max_files_to_analyze is not None and 0 < max_files_to_analyze < len(xml_files):
        print(f"Анализируется {max_files_to_analyze} случайных XML файлов из {len(xml_files)}...")
        xml_files = random.sample(xml_files, max_files_to_analyze)
    else:
        print(f"Анализируется {len(xml_files)} XML файлов...")

    assignments_count = {level: 0 for level in FPN_LEVEL_NAMES_ANALYZER}
    lost_gt_count = 0
    total_gt_objects_processed = 0
    max_ious_per_level = {level: [] for level in FPN_LEVEL_NAMES_ANALYZER}
    # gt_areas_assigned_debug = {level: [] for level in FPN_LEVEL_NAMES_ANALYZER} # Для отладки площадей

    for xml_file_path_obj in xml_files:
        xml_file_path_str = str(xml_file_path_obj)
        base_filename = xml_file_path_obj.stem

        image_path_for_size_str = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path_candidate = images_dir / (base_filename + ext)
            if img_path_candidate.exists():
                image_path_for_size_str = str(img_path_candidate)
                break
        if image_path_for_size_str is None: continue

        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path_for_size_str)
            original_w_px, original_h_px = pil_img.size
            if original_w_px <= 0 or original_h_px <= 0: continue
        except Exception:
            continue

        # Используем parse_xml_annotation для чтения XML
        # Предполагается, что CLASSES_LIST_ANALYZER используется внутри parse_xml_annotation, если она передается
        # или parse_xml_annotation берет ее из своего глобального контекста (импортированную из detector_data_loader)
        gt_objects_from_xml, _, _, _ = parse_xml_annotation(xml_file_path_str, CLASSES_LIST_ANALYZER)
        if gt_objects_from_xml is None: continue

        for gt_obj_data in gt_objects_from_xml:
            total_gt_objects_processed += 1
            xmin, ymin, xmax, ymax = gt_obj_data['xmin'], gt_obj_data['ymin'], gt_obj_data['xmax'], gt_obj_data['ymax']
            gt_w_px = xmax - xmin
            gt_h_px = ymax - ymin
            if gt_w_px <= 0 or gt_h_px <= 0: continue

            # Используем ту же метрику масштаба, что и в detector_data_loader.py
            # Обычно это sqrt(площади) или большая сторона.
            # Давай предположим, что data_loader использует object_scale_px = np.sqrt(gt_w_px * gt_h_px)
            object_scale_px = np.sqrt(gt_w_px * gt_h_px)
            # Если data_loader использует max(gt_w_px, gt_h_px), то и здесь нужно использовать max.
            # object_scale_px = max(gt_w_px, gt_h_px) # Альтернативная метрика масштаба

            assigned_level = None
            # <<<--- ИЗМЕНЕННАЯ ЛОГИКА НАЗНАЧЕНИЯ НА УРОВНИ FPN ---<<<
            for level_name_assign_loop in FPN_LEVEL_NAMES_ANALYZER:  # Должен быть ['P3', 'P4', 'P5']
                ranges = FPN_GT_ASSIGNMENT_SCALE_RANGES_ANALYZER.get(level_name_assign_loop)
                if ranges and ranges[0] <= object_scale_px < ranges[1]:
                    assigned_level = level_name_assign_loop
                    break

            if assigned_level is None and FPN_LEVEL_NAMES_ANALYZER:  # Если не попал ни в один диапазон
                # Проверяем, не больше ли он или равен началу диапазона последнего уровня (P5)
                # или если он просто очень большой
                last_level_name = FPN_LEVEL_NAMES_ANALYZER[-1]
                last_level_ranges = FPN_GT_ASSIGNMENT_SCALE_RANGES_ANALYZER.get(last_level_name)
                if last_level_ranges and object_scale_px >= last_level_ranges[0]:
                    assigned_level = last_level_name
            # --- Конец Измененной Логики ---

            if assigned_level and assigned_level in FPN_LEVEL_NAMES_ANALYZER:
                assignments_count[assigned_level] += 1
                # gt_areas_assigned_debug[assigned_level].append(object_scale_px)

                level_anchor_config = FPN_ANCHOR_CONFIGS_ANALYZER.get(assigned_level)
                if level_anchor_config and 'anchors_wh_normalized' in level_anchor_config:
                    level_anchors_wh = np.array(level_anchor_config['anchors_wh_normalized'], dtype=np.float32)

                    if level_anchors_wh.shape[0] > 0:  # Если есть якоря для этого уровня
                        gt_w_norm_iou = gt_w_px / original_w_px
                        gt_h_norm_iou = gt_h_px / original_h_px

                        ious_with_anchors = calculate_iou_numpy([gt_w_norm_iou, gt_h_norm_iou], level_anchors_wh)
                        if ious_with_anchors.size > 0:
                            max_ious_per_level[assigned_level].append(np.max(ious_with_anchors))
                else:
                    print(
                        f"ПРЕДУПРЕЖДЕНИЕ: Конфигурация якорей не найдена для уровня {assigned_level} в FPN_ANCHOR_CONFIGS_ANALYZER.")
            else:
                lost_gt_count += 1
                # gt_areas_lost.append(object_scale_px)
                # print(f"  GT '{gt_obj_data['class_name']}' (scale: {object_scale_px:.1f}px) не назначен. XML: {os.path.basename(xml_file_path_str)}")

    # --- Вывод итоговой статистики ---
    print("\n--- Итоговая Статистика Назначения GT Объектов на Уровни FPN (по диапазонам из конфига) ---")
    if total_gt_objects_processed == 0:
        print("Не найдено ни одного GT объекта для анализа в этой выборке.")
        return

    print(f"Всего обработано GT объектов (известных классов): {total_gt_objects_processed}")
    for level_name_stat, count_stat_val in assignments_count.items():
        percent_stat_val = count_stat_val / total_gt_objects_processed * 100 if total_gt_objects_processed > 0 else 0
        print(f"  Назначено на уровень {level_name_stat}: {count_stat_val} объектов ({percent_stat_val:.1f}%)")

    lost_percent_val = lost_gt_count / total_gt_objects_processed * 100 if total_gt_objects_processed > 0 else 0
    print(
        f"  'Потеряно' GT объектов (не назначено ни одному уровню по диапазонам): {lost_gt_count} ({lost_percent_val:.1f}%)")

    print("\n--- Статистика Максимального IoU между GT и Якорями на Назначенном Уровне ---")
    for level_name_iou, ious_list_iou in max_ious_per_level.items():
        if ious_list_iou:
            print(f"  Уровень {level_name_iou}:")
            print(f"    Количество GT, для которых рассчитан IoU: {len(ious_list_iou)}")
            iou_arr_stat_final_iou = np.array(ious_list_iou)
            print(f"    Средний Max IoU: {np.mean(iou_arr_stat_final_iou):.3f}")
            print(f"    Медианный Max IoU: {np.median(iou_arr_stat_final_iou):.3f}")
            print(f"    Мин Max IoU: {np.min(iou_arr_stat_final_iou):.3f}")
            print(f"    Макс Max IoU: {np.max(iou_arr_stat_final_iou):.3f}")

            plt_imported_success_iou = False
            try:
                # from matplotlib import pyplot as plt # Уже импортирован глобально
                plt_imported_success_iou = True
            except ImportError:
                print("    ПРЕДУПРЕЖДЕНИЕ: Matplotlib не найден. Гистограмма не будет построена.")

            if plt_imported_success_iou:
                plt.figure(figsize=(8, 4))
                plt.hist(iou_arr_stat_final_iou, bins=20, range=(0, 1), edgecolor='black')
                plt.title(f"Гистограмма Max IoU (GT с якорями) для уровня {level_name_iou} ({split_name_for_log})")
                plt.xlabel("Max IoU")
                plt.ylabel("Количество назначенных GT")
                plt.grid(True)
                graphs_dir_debug_final_iou = _project_root_analyzer / "graphs" / "fpn_gt_assignment_analysis_v2"  # Новое имя папки
                graphs_dir_debug_final_iou.mkdir(parents=True, exist_ok=True)
                plot_save_path_debug_final_iou = graphs_dir_debug_final_iou / f"max_iou_hist_assigned_{level_name_iou}_{split_name_for_log}.png"
                try:
                    plt.savefig(plot_save_path_debug_final_iou)
                    print(f"    Гистограмма сохранена: {plot_save_path_debug_final_iou}")
                except Exception as e_plt_save_final_iou:
                    print(f"    Не удалось сохранить гистограмму: {e_plt_save_final_iou}")
                plt.close()
        else:
            print(
                f"  Уровень {level_name_iou}: Нет GT объектов, назначенных на этот уровень, или для них не рассчитан IoU (или нет якорей на уровне).")

    print(f"\n--- Анализ для выборки {split_name_for_log.upper()} завершен ---")


# --- Блок if __name__ == "__main__": ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Анализ назначения GT объектов на уровни FPN, используя fpn_gt_assignment_scale_ranges из конфига.")
    parser.add_argument(
        "--dataset_split", type=str, default="all", choices=["train", "validation", "all"],
        help="Какую выборку анализировать: 'train', 'validation' или 'all' для обеих (дефолт: 'all')."
    )
    parser.add_argument(
        "--max_files_per_split", type=int, default=None,
        help="Максимальное количество XML файлов для анализа ИЗ КАЖДОЙ ВЫБОРКИ (для быстрого теста)."
    )
    args = parser.parse_args()

    splits_to_analyze_main = []
    if args.dataset_split == "all":
        splits_to_analyze_main = ["train", "validation"]
    elif args.dataset_split in ["train", "validation"]:
        splits_to_analyze_main = [args.dataset_split]
    else:
        print(
            f"Некорректное значение для --dataset_split: {args.dataset_split}. Используйте 'train', 'validation' или 'all'.")
        exit()

    _detector_dataset_ready_path_rel_main = "data/Detector_Dataset_Ready"  # Относительный путь от корня проекта
    _detector_dataset_ready_root_abs_main_script = (
                _project_root_analyzer / _detector_dataset_ready_path_rel_main).resolve()

    if not _detector_dataset_ready_root_abs_main_script.is_dir():
        print(
            f"ОШИБКА: Корневая директория разделенного датасета не найдена: {_detector_dataset_ready_root_abs_main_script}")
        print(
            f"  Ожидался путь: '{_detector_dataset_ready_path_rel_main}' относительно корня проекта '{_project_root_analyzer}'")
        print("Пожалуйста, запустите сначала 'create_data_splits.py' и убедитесь, что папка создана.")
        exit()

    for current_split_name_main_script in splits_to_analyze_main:
        _target_split_dir_abs_main_script = (
                    _detector_dataset_ready_root_abs_main_script / current_split_name_main_script).resolve()

        if not _target_split_dir_abs_main_script.is_dir():
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: Директория для выборки '{current_split_name_main_script}' не найдена: {_target_split_dir_abs_main_script}. Пропускаем эту выборку.")
            continue

        target_annotations_dir_main_script = str(_target_split_dir_abs_main_script / ANNOTATIONS_SUBDIR_CFG_ANALYZER)
        target_images_dir_main_script = str(_target_split_dir_abs_main_script / IMAGES_SUBDIR_CFG_ANALYZER)

        analyze_gt_assignments(
            current_split_name_main_script,
            target_annotations_dir_main_script,
            target_images_dir_main_script,
            args.max_files_per_split
        )