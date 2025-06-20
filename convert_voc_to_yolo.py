# convert_voc_to_yolo.py
import xml.etree.ElementTree as ET
import os
import glob
from PIL import Image


# Функции voc_to_yolo_bbox и convert_single_xml_to_yolo остаются ТАКИМИ ЖЕ,
# как в предыдущем моем ответе. Я их здесь не буду повторять для краткости,
# но они должны быть в твоем файле.
# ... (Вставь сюда код функций voc_to_yolo_bbox и convert_single_xml_to_yolo) ...

def voc_to_yolo_bbox(box_coords_voc, image_width, image_height):
    xmin, ymin, xmax, ymax = box_coords_voc
    xmin = max(0, float(xmin));
    ymin = max(0, float(ymin))
    xmax = min(float(image_width), float(xmax));
    ymax = min(float(image_height), float(ymax))
    if xmin >= xmax or ymin >= ymax: return None
    dw = 1.0 / image_width;
    dh = 1.0 / image_height
    x_center = (xmin + xmax) / 2.0;
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin;
    height = ymax - ymin
    return x_center * dw, y_center * dh, width * dw, height * dh


def convert_single_xml_to_yolo(xml_file_path, image_file_path, classes_list, output_txt_path):
    try:
        tree = ET.parse(xml_file_path);
        root = tree.getroot()
        size_node = root.find('size');
        img_width = None;
        img_height = None
        if size_node is not None:
            try:
                img_width = int(size_node.find('width').text);
                img_height = int(size_node.find('height').text)
            except:
                pass
        if not img_width or not img_height or img_width <= 0 or img_height <= 0:
            if not os.path.exists(image_file_path): return False
            try:
                with Image.open(image_file_path) as img:
                    img_width, img_height = img.size
                if img_width <= 0 or img_height <= 0: return False
            except:
                return False
        yolo_lines = []
        for obj_node in root.findall('object'):
            class_name_node = obj_node.find('name')
            if class_name_node is None or class_name_node.text is None: continue
            class_name = class_name_node.text
            if class_name not in classes_list: continue
            class_id = classes_list.index(class_name)
            bndbox_node = obj_node.find('bndbox')
            if bndbox_node is None: continue
            try:
                xmin, ymin, xmax, ymax = (float(bndbox_node.find(t).text) for t in ['xmin', 'ymin', 'xmax', 'ymax'])
            except:
                continue
            yolo_coords = voc_to_yolo_bbox([xmin, ymin, xmax, ymax], img_width, img_height)
            if yolo_coords: yolo_lines.append(
                f"{class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}")
        with open(output_txt_path, 'w') as f:
            if yolo_lines: f.write("\n".join(yolo_lines))
        return True
    except Exception:
        return False


def process_dataset_split_for_yolo(base_split_dir_path, classes_config_list):
    """
    Конвертирует XML в YOLO TXT для указанного разделения (например, train или validation).
    Создает папку 'labels' внутри base_split_dir_path.
    base_split_dir_path: Путь к папке split'а (например, "data/Detector_Dataset_Ready/train/")
                         Ожидается, что внутри есть "Annotations/" и "JPEGImages/".
    """
    annotations_input_dir = os.path.join(base_split_dir_path, "Annotations")
    images_input_dir = os.path.join(base_split_dir_path, "JPEGImages")
    labels_output_dir = os.path.join(base_split_dir_path, "labels")  # <<<--- НОВАЯ ПАПКА ЗДЕСЬ

    if not os.path.isdir(annotations_input_dir):
        print(f"ОШИБКА: Директория аннотаций не найдена: {annotations_input_dir}")
        return
    if not os.path.isdir(images_input_dir):
        print(f"ОШИБКА: Директория изображений не найдена: {images_input_dir}")
        return

    os.makedirs(labels_output_dir, exist_ok=True)
    print(f"Конвертация XML из '{annotations_input_dir}' в YOLO TXT в '{labels_output_dir}'...")

    xml_files = glob.glob(os.path.join(annotations_input_dir, "*.xml"))
    if not xml_files:
        print("XML файлы не найдены в указанной директории аннотаций.")
        return

    conversion_summary = {'success': 0, 'failed_or_skipped': 0, 'total': len(xml_files)}

    for xml_path in xml_files:
        base_filename = os.path.splitext(os.path.basename(xml_path))[0]

        image_path_found = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_image_path = os.path.join(images_input_dir, base_filename + ext)
            if os.path.exists(potential_image_path):
                image_path_found = potential_image_path
                break

        if not image_path_found:
            print(
                f"WARNING: Изображение для аннотации '{os.path.basename(xml_path)}' не найдено в '{images_input_dir}'. Пропускаем.")
            conversion_summary['failed_or_skipped'] += 1
            continue

        output_txt_file_path = os.path.join(labels_output_dir, base_filename + ".txt")

        if convert_single_xml_to_yolo(xml_path, image_path_found, classes_config_list, output_txt_file_path):
            conversion_summary['success'] += 1
        else:
            conversion_summary['failed_or_skipped'] += 1

    print("\n--- Сводка Конвертации для этого split'а ---")
    print(f"Всего XML файлов для обработки: {conversion_summary['total']}")
    print(f"Успешно конвертировано в TXT: {conversion_summary['success']}")
    print(f"Пропущено или с ошибками: {conversion_summary['failed_or_skipped']}")
    print(f"YOLO TXT файлы сохранены в: {os.path.abspath(labels_output_dir)}")


if __name__ == "__main__":
    # --- НАСТРОЙКИ ---
    # Базовый путь к папке, где лежат train и validation (т.е., к Detector_Dataset_Ready)
    # Путь должен быть либо абсолютным, либо относительным от места запуска скрипта (корня проекта).
    DETECTOR_DATASET_READY_ROOT = "data/Detector_Dataset_Ready"

    # Список классов в том порядке, как ты хочешь, чтобы им были присвоены ID (0, 1, ...)
    CLASSES_CONFIG = ["pit", "crack"]  # pit будет 0, crack будет 1

    # --- КОНВЕРТАЦИЯ ДЛЯ TRAIN ---
    train_split_path = os.path.join(os.getcwd(), DETECTOR_DATASET_READY_ROOT, "train")
    print(f"\n--- Конвертация ОБУЧАЮЩЕЙ выборки: {train_split_path} ---")
    if os.path.isdir(train_split_path):
        process_dataset_split_for_yolo(train_split_path, CLASSES_CONFIG)
    else:
        print(f"ОШИБКА: Директория обучающей выборки не найдена: {train_split_path}")

    # --- КОНВЕРТАЦИЯ ДЛЯ VALIDATION ---
    val_split_path = os.path.join(os.getcwd(), DETECTOR_DATASET_READY_ROOT, "validation")
    print(f"\n--- Конвертация ВАЛИДАЦИОННОЙ выборки: {val_split_path} ---")
    if os.path.isdir(val_split_path):
        process_dataset_split_for_yolo(val_split_path, CLASSES_CONFIG)
    else:
        print(f"ОШИБКА: Директория валидационной выборки не найдена: {val_split_path}")

    # После этого скрипта тебе НЕ нужно будет отдельно копировать изображения для YOLO,
    # так как YOLO будет читать изображения из существующих папок JPEGImages/train и JPEGImages/validation,
    # а метки - из новых labels/train и labels/validation.
    # Тебе нужно будет только правильно настроить `data.yaml` для YOLO.
    print("\nКонвертация завершена. Проверьте созданные папки 'labels' внутри 'train' и 'validation'.")