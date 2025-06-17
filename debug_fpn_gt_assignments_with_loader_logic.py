# debug_fpn_gt_assignments_with_loader_logic.py
import sys
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import glob
import argparse

from matplotlib import pyplot as plt

# --- Добавляем src в sys.path, чтобы импортировать из detector_data_loader ---
_project_root_debug = Path(__file__).resolve().parent
_src_path_debug = _project_root_debug / 'src'
if str(_src_path_debug) not in sys.path:
    sys.path.insert(0, str(_src_path_debug))

# --- Импорты из твоего detector_data_loader ---
# Мы импортируем НЕПОСРЕДСТВЕННО переменные и функции, которые он использует,
# чтобы гарантировать идентичность логики.
try:
    from datasets.detector_data_loader import (
        parse_xml_annotation,  # Для чтения XML
        calculate_iou_numpy,  # Для расчета IoU с якорями
        # Глобальные переменные, инициализированные в detector_data_loader из конфигов:
        FPN_LEVELS_CONFIG_GLOBAL,
        FPN_SCALE_RANGES_CFG_FROM_YAML,
        CLASSES_LIST_GLOBAL_FOR_DETECTOR,
        NUM_CLASSES_DETECTOR,
        FPN_LEVEL_NAMES_ORDERED,
        BASE_CONFIG as DDL_BASE_CONFIG,  # Переименуем, чтобы не конфликтовать
        _images_subdir_name_cfg as DDL_IMAGES_SUBDIR,
        _annotations_subdir_name_cfg as DDL_ANNOTATIONS_SUBDIR
    )

    print("INFO: Компоненты из detector_data_loader.py успешно импортированы.")
except ImportError as e:
    print(f"ОШИБКА: Не удалось импортировать компоненты из detector_data_loader.py: {e}")
    print("Убедитесь, что detector_data_loader.py находится в src/datasets/ и не содержит ошибок на уровне импорта.")
    exit()
except Exception as e_general:
    print(f"ОШИБКА при импорте из detector_data_loader.py: {e_general}")
    exit()


# --- Основная функция анализа ---
def analyze_assignments_with_loader_logic(annotations_dir_path_str, images_dir_path_str, max_files_to_analyze=None):
    """
    Анализирует назначение GT объектов на уровни FPN, используя ту же логику и
    конфигурации, что и detector_data_loader.py.
    """
    annotations_dir = Path(annotations_dir_path_str)
    images_dir = Path(images_dir_path_str)

    if not annotations_dir.is_dir():
        print(f"ОШИБКА: Директория аннотаций не найдена: {annotations_dir}")
        return
    if not images_dir.is_dir():
        print(f"ОШИБКА: Директория изображений не найдена: {images_dir}")
        return

    xml_files_list = sorted(list(annotations_dir.glob("*.xml")))
    if not xml_files_list:
        print(f"XML файлы не найдены в {annotations_dir}")
        return

    if max_files_to_analyze is not None and max_files_to_analyze < len(xml_files_list):
        print(f"Анализируется {max_files_to_analyze} случайных файлов из {len(xml_files_list)}...")
        import random
        xml_files_list = random.sample(xml_files_list, max_files_to_analyze)
    else:
        print(f"Анализируется {len(xml_files_list)} XML файлов...")

    assignments_summary = {level: {"count": 0, "iou_values": []} for level in FPN_LEVEL_NAMES_ORDERED}
    total_gt_processed = 0
    gt_not_assigned_count = 0

    for i_xml, xml_file_path_obj in enumerate(xml_files_list):
        xml_file_path_str = str(xml_file_path_obj)
        base_filename = xml_file_path_obj.stem

        # Ищем соответствующее изображение (с разными расширениями)
        image_path_str = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path_candidate = images_dir / (base_filename + ext)
            if img_path_candidate.exists():
                image_path_str = str(img_path_candidate)
                break

        if image_path_str is None:
            # print(f"  Предупреждение: Изображение для {xml_file_path_obj.name} не найдено. Пропуск.")
            continue

        # Получаем ОРИГИНАЛЬНЫЕ размеры изображения из файла, а не из XML (надежнее)
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path_str)
            original_w_px, original_h_px = pil_img.size
            if original_w_px <= 0 or original_h_px <= 0: continue
        except Exception:
            # print(f"  Предупреждение: Не удалось прочитать размеры для {image_path_obj.name}. Пропуск.")
            continue

        gt_objects_from_xml, xml_w, xml_h, _ = parse_xml_annotation(xml_file_path_str, CLASSES_LIST_GLOBAL_FOR_DETECTOR)

        if gt_objects_from_xml is None:  # Ошибка парсинга
            continue

        # print(f"\nФайл: {xml_file_path_obj.name} (Размеры: {original_w_px}x{original_h_px})")

        for gt_obj_data in gt_objects_from_xml:
            total_gt_processed += 1
            xmin, ymin, xmax, ymax = gt_obj_data['xmin'], gt_obj_data['ymin'], gt_obj_data['xmax'], gt_obj_data['ymax']
            gt_w_px_val = xmax - xmin
            gt_h_px_val = ymax - ymin

            if gt_w_px_val <= 0 or gt_h_px_val <= 0: continue

            object_scale_px_val = np.sqrt(gt_w_px_val * gt_h_px_val)
            assigned_level = None

            # Логика назначения (точно такая же, как в detector_data_loader.py)
            for level_name_loop in FPN_LEVEL_NAMES_ORDERED:  # ['P3', 'P4', 'P5']
                min_scale, max_scale = FPN_SCALE_RANGES_CFG_FROM_YAML[level_name_loop]
                if min_scale <= object_scale_px_val < max_scale:
                    assigned_level = level_name_loop
                    break
            if assigned_level is None:
                if object_scale_px_val >= FPN_SCALE_RANGES_CFG_FROM_YAML.get('P5', [0, 0])[0]:
                    assigned_level = 'P5'

            if assigned_level:
                assignments_summary[assigned_level]["count"] += 1

                # Расчет Max IoU с якорями этого уровня
                gt_w_norm_val = gt_w_px_val / original_w_px
                gt_h_norm_val = gt_h_px_val / original_h_px
                level_anchors_wh_val = FPN_LEVELS_CONFIG_GLOBAL[assigned_level]['anchors_wh_normalized']

                if level_anchors_wh_val.shape[0] > 0:
                    ious_val = calculate_iou_numpy([gt_w_norm_val, gt_h_norm_val], level_anchors_wh_val)
                    assignments_summary[assigned_level]["iou_values"].append(np.max(ious_val))

                # print(f"  GT: {gt_obj_data['class_name']}, Scale: {object_scale_px_val:.1f}px -> Назначен на {assigned_level}")
            else:
                gt_not_assigned_count += 1
                # print(f"  GT: {gt_obj_data['class_name']}, Scale: {object_scale_px_val:.1f}px -> НЕ НАЗНАЧЕН")

    # --- Вывод итоговой статистики ---
    print("\n--- Итоговая Статистика Назначения GT Объектов на Уровни FPN (по логике data_loader) ---")
    if total_gt_processed == 0:
        print("Не найдено ни одного GT объекта для анализа.")
        return

    print(f"Всего обработано GT объектов (известных классов): {total_gt_processed}")
    for level_name_stat, data_stat in assignments_summary.items():
        count_stat = data_stat["count"]
        percent_stat = count_stat / total_gt_processed * 100 if total_gt_processed > 0 else 0
        print(f"  Назначено на уровень {level_name_stat}: {count_stat} объектов ({percent_stat:.1f}%)")
        if data_stat["iou_values"]:
            iou_arr_stat = np.array(data_stat["iou_values"])
            print(f"    Средний Max IoU с якорями уровня: {np.mean(iou_arr_stat):.3f}")
            print(f"    Медианный Max IoU: {np.median(iou_arr_stat):.3f}")
            print(f"    Мин/Макс Max IoU: {np.min(iou_arr_stat):.3f} / {np.max(iou_arr_stat):.3f}")

            # Построение гистограммы
            plt.figure(figsize=(8, 4))
            plt.hist(iou_arr_stat, bins=20, range=(0, 1), edgecolor='black')
            plt.title(f"Гистограмма Max IoU (data_loader logic) для уровня {level_name_stat}")
            plt.xlabel("Max IoU GT с якорями назначенного уровня")
            plt.ylabel("Количество GT объектов")
            plt.grid(True)
            graphs_dir_debug = _project_root_debug / "graphs" / "debug_assignments"
            graphs_dir_debug.mkdir(parents=True, exist_ok=True)
            plot_save_path_debug = graphs_dir_debug / f"max_iou_hist_loader_logic_{level_name_stat}.png"
            try:
                plt.savefig(plot_save_path_debug)
                print(f"    Гистограмма сохранена: {plot_save_path_debug}")
            except Exception as e_plt_save:
                print(f"    Не удалось сохранить гистограмму: {e_plt_save}")
            plt.close()  # Закрываем фигуру, чтобы не накапливались

    print(
        f"  'Потеряно' GT объектов (не назначено ни одному уровню): {gt_not_assigned_count} ({gt_not_assigned_count / total_gt_processed * 100 if total_gt_processed > 0 else 0:.1f}%)")
    print("\n--- Анализ завершен ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Анализ назначения GT объектов на уровни FPN согласно логике detector_data_loader.py.")
    parser.add_argument(
        "--dataset_split", type=str, default="train", choices=["train", "validation"],
        help="Какую выборку анализировать: 'train' или 'validation' (дефолт: 'train')."
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Максимальное количество XML файлов для анализа (для быстрого теста)."
    )
    args = parser.parse_args()

    # Определяем пути к выбранной выборке
    _target_split_dir_abs = (_project_root_debug / "data" / "Detector_Dataset_Ready" / args.dataset_split).resolve()
    target_annotations_dir = str(_target_split_dir_abs / DDL_ANNOTATIONS_SUBDIR)
    target_images_dir = str(_target_split_dir_abs / DDL_IMAGES_SUBDIR)

    analyze_assignments_with_loader_logic(target_annotations_dir, target_images_dir, args.max_files)