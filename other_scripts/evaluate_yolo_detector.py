# other_scripts/evaluate_yolo_detector.py

from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import cv2  # Для загрузки изображений и получения их размеров (если нужно)
import numpy as np
from tqdm import tqdm  # Для индикатора прогресса
import xml.etree.ElementTree as ET  # Для парсинга ваших XML аннотаций

# --- Определяем корень проекта и путь к скрипту ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# --- Функции расчета метрик (копируем из вашего evaluate_detector_v3_standard.py) ---
def calculate_iou(box1, box2):
    """Вычисляет IoU между двумя боксами. Формат: [xmin, ymin, xmax, ymax]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        intersection_area = 0.0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / (union_area + 1e-7)


def calculate_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return precision, recall, f1


# --- Функция для парсинга ваших XML аннотаций ---
def parse_voc_xml_for_yolo_eval(xml_path: Path, class_name_to_id_map: dict):
    """Парсит XML и возвращает список GT объектов для оценки."""
    gt_objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_size_node = root.find('size')
        img_width = int(image_size_node.find('width').text)
        img_height = int(image_size_node.find('height').text)

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_name_to_id_map.get(class_name)
            if class_id is None:
                print(f"Предупреждение: Неизвестный класс '{class_name}' в {xml_path}, пропускаем.")
                continue

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Нормализуем координаты
            norm_xmin = xmin / img_width
            norm_ymin = ymin / img_height
            norm_xmax = xmax / img_width
            norm_ymax = ymax / img_height

            gt_objects.append({
                'bbox': [norm_xmin, norm_ymin, norm_xmax, norm_ymax],  # Нормализованные [xmin, ymin, xmax, ymax]
                'class_id': class_id,
                'used': False  # Для сопоставления
            })
    except Exception as e:
        print(f"Ошибка парсинга XML {xml_path}: {e}")
    return gt_objects


def run_yolo_evaluation_with_custom_metrics():
    print("--- Начало оценки YOLO модели с кастомными метриками ---")

    # --- Пути к модели и данным ---
    model_path_relative_to_project = 'training_runs_yolov8/run1_pit_crack2/weights/best.pt'
    model_path = PROJECT_ROOT / model_path_relative_to_project

    # Загружаем ваш основной конфиг, чтобы получить пути к датасету и имена классов
    main_config_path_relative = 'src/configs/detector_config_v3_standard.yaml'
    main_config_path = PROJECT_ROOT / main_config_path_relative
    try:
        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        class_names = main_config['class_names']
        class_name_to_id_map = {name: i for i, name in enumerate(class_names)}
        num_classes = main_config['num_classes']
        # Параметры для NMS и IoU из predict_config (или вашего конфига YOLO)
        # Используем eval_conf_threshold и eval_iou_threshold из вашего основного конфига
        # для консистентности, или определите их отдельно для YOLO.
        yolo_conf_threshold = main_config.get('eval_conf_threshold', 0.25)  # Порог уверенности для предсказаний YOLO
        iou_threshold_for_metrics = main_config.get('eval_iou_threshold',
                                                    0.1)  # Используем тот же порог IoU, что и для вашей модели
        print(f"Используется порог IoU для метрик: {iou_threshold_for_metrics}")
        print(f"Используется порог уверенности для предсказаний YOLO: {yolo_conf_threshold}")

    except Exception as e:
        print(f"Ошибка загрузки основного конфига {main_config_path}: {e}")
        return

    # --- Получаем список валидационных изображений и аннотаций ---
    dataset_path_root = PROJECT_ROOT / main_config['dataset_path']
    val_images_dir = dataset_path_root / main_config['val_images_subdir']
    val_annotations_dir = dataset_path_root / main_config['val_annotations_subdir']

    val_image_paths = sorted(list(val_images_dir.glob('*.jpg')))  # Или другие ваши расширения
    if not val_image_paths:
        print(f"Валидационные изображения не найдены в {val_images_dir}")
        return
    print(f"Найдено {len(val_image_paths)} валидационных изображений.")

    # --- Загрузка YOLO модели ---
    try:
        model = YOLO(str(model_path))
        print(f"Модель YOLO {model_path} успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели YOLO: {e}")
        return

    all_predictions_yolo = []
    all_ground_truths_yolo = []
    image_id_counter = 0

    print("\nНачало инференса YOLO и сбора предсказаний/GT...")
    for img_path in tqdm(val_image_paths, desc="Обработка YOLO"):
        # 1. Получаем предсказания YOLO
        try:
            # Запускаем predict. conf - это порог для детекций, которые вернутся.
            # iou - это порог NMS внутри YOLO.
            results = model.predict(source=str(img_path), conf=yolo_conf_threshold, iou=0.5,
                                    verbose=False)  # verbose=False чтобы уменьшить вывод
        except Exception as e_pred:
            print(f"Ошибка предсказания YOLO для {img_path.name}: {e_pred}")
            continue

        if results and results[0].boxes:
            boxes_xyxyn = results[0].boxes.xyxyn.cpu().numpy()  # Нормализованные [xmin, ymin, xmax, ymax]
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids_pred = results[0].boxes.cls.cpu().numpy().astype(int)

            for box_norm, score, cls_id in zip(boxes_xyxyn, scores, class_ids_pred):
                all_predictions_yolo.append({
                    'image_id': image_id_counter,
                    'bbox': list(box_norm),  # [xmin, ymin, xmax, ymax]
                    'score': float(score),
                    'class_id': int(cls_id)
                })

        # 2. Загружаем GT для этого изображения
        annot_path = val_annotations_dir / (img_path.stem + ".xml")
        if annot_path.exists():
            gt_objects_for_img = parse_voc_xml_for_yolo_eval(annot_path, class_name_to_id_map)
            for gt_obj in gt_objects_for_img:
                gt_obj['image_id'] = image_id_counter  # Добавляем image_id
                all_ground_truths_yolo.append(gt_obj)
        else:
            print(f"Предупреждение: Аннотация не найдена для {img_path.name}")

        image_id_counter += 1

    print("Сбор предсказаний и GT завершен.")

    # 3. Расчет метрик (используем ваши функции)
    print("\n" + "=" * 30 + " Результаты Оценки YOLO " + "=" * 30)
    print(f"Порог IoU для TP/FP: {iou_threshold_for_metrics}")

    total_tp_yolo, total_fp_yolo, total_fn_yolo = 0, 0, 0

    for class_id_eval in range(num_classes):
        class_name_eval = class_names[class_id_eval]
        preds_for_class = [p for p in all_predictions_yolo if p['class_id'] == class_id_eval]
        gts_for_class = [g for g in all_ground_truths_yolo if g['class_id'] == class_id_eval]

        preds_for_class.sort(key=lambda x: x['score'], reverse=True)

        tp_class, fp_class = 0, 0
        for gt in gts_for_class: gt['used'] = False  # Сбрасываем флаг

        for pred in preds_for_class:
            best_iou = 0.0
            best_gt_match_idx = -1
            gts_on_image_for_pred = [(idx, gt) for idx, gt in enumerate(gts_for_class) if
                                     gt['image_id'] == pred['image_id'] and not gt['used']]

            for gt_original_idx, gt_on_img in gts_on_image_for_pred:
                iou = calculate_iou(pred['bbox'], gt_on_img['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_match_idx = gt_original_idx

            if best_iou >= iou_threshold_for_metrics and best_gt_match_idx != -1:
                tp_class += 1
                gts_for_class[best_gt_match_idx]['used'] = True
            else:
                fp_class += 1

        fn_class = len(gts_for_class) - tp_class

        precision, recall, f1 = calculate_precision_recall_f1(tp_class, fp_class, fn_class)
        total_tp_yolo += tp_class
        total_fp_yolo += fp_class
        total_fn_yolo += fn_class

        print(f"\n--- Класс: {class_name_eval} (ID: {class_id_eval}) ---")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  (TP: {tp_class}, FP: {fp_class}, FN: {fn_class}, Total GT: {len(gts_for_class)})")

    micro_precision, micro_recall, micro_f1 = calculate_precision_recall_f1(total_tp_yolo, total_fp_yolo, total_fn_yolo)
    print("\n" + "=" * 30 + " Итоговые Метрики YOLO (Micro-Averaged) " + "=" * 30)
    print(f"Micro-Precision: {micro_precision:.4f}")
    print(f"Micro-Recall:    {micro_recall:.4f}")
    print(f"Micro-F1-score:  {micro_f1:.4f}")
    print(
        f"(Total TP: {total_tp_yolo}, Total FP: {total_fp_yolo}, Total FN: {total_fn_yolo}, Total GT Objects: {len(all_ground_truths_yolo)})")
    print("=" * 78)


# --- ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == '__main__':
    run_yolo_evaluation_with_custom_metrics()