# RoadDefectDetector/prepare_classifier_dataset.py
import os
import shutil
import glob
import random
import yaml

# --- Загрузка Конфигураций ---
# Скрипт находится в корневой папке проекта RoadDefectDetector/
# Пути к конфигам указываются относительно этой корневой папки.
_current_project_root = os.path.dirname(os.path.abspath(__file__))

_base_config_path = os.path.join(_current_project_root, 'src', 'configs', 'base_config.yaml')
_classifier_config_path = os.path.join(_current_project_root, 'src', 'configs', 'classifier_config.yaml')

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:  # Добавил encoding
        BASE_CONFIG = yaml.safe_load(f)
    with open(_classifier_config_path, 'r', encoding='utf-8') as f:  # Добавил encoding
        CLASSIFIER_CONFIG = yaml.safe_load(f)
except FileNotFoundError as e:
    print(f"ОШИБКА: Не найден один из файлов конфигурации.")
    print(f"Ожидаемый путь к base_config.yaml: {_base_config_path}")
    print(f"Ожидаемый путь к classifier_config.yaml: {_classifier_config_path}")
    print(f"Детали ошибки: {e}")
    exit()
except yaml.YAMLError as e:
    print(f"ОШИБКА: Не удалось прочитать YAML файл конфигурации.")
    print(f"Детали ошибки: {e}")
    exit()

# --- Параметры из Конфигов ---
# Путь к исходному "мастер" датасету от друга (может быть абсолютным или относительным от корня проекта)
MASTER_DATASET_PATH_FROM_CONFIG = BASE_CONFIG.get('master_dataset_path', '')
if not MASTER_DATASET_PATH_FROM_CONFIG:
    print("ОШИБКА: 'master_dataset_path' не указан в 'base_config.yaml'.")
    exit()

# Если путь в конфиге относительный, делаем его абсолютным от корня проекта
if not os.path.isabs(MASTER_DATASET_PATH_FROM_CONFIG):
    MASTER_DATASET_PATH = os.path.join(_current_project_root, MASTER_DATASET_PATH_FROM_CONFIG)
else:
    MASTER_DATASET_PATH = MASTER_DATASET_PATH_FROM_CONFIG

# Путь к целевому датасету для классификатора (относительно корня проекта)
CLASSIFIER_TARGET_PATH_FROM_CONFIG = CLASSIFIER_CONFIG.get('prepared_dataset_path', 'data/Classifier_Dataset')
CLASSIFIER_TARGET_PATH_ABS = os.path.join(_current_project_root, CLASSIFIER_TARGET_PATH_FROM_CONFIG)

TRAIN_RATIO = CLASSIFIER_CONFIG.get('train_ratio', 0.8)

# Имена классов для папок классификатора
CLASS_ROAD = "road"
CLASS_NOT_ROAD = "not_road"

# Имена подпапок в Master_Dataset_From_Friend (как их предоставит друг)
# Эти подпапки должны лежать внутри MASTER_DATASET_PATH
# и каждая из них должна содержать подпапку JPEGImages
SOURCE_DEFECTIVE_ROAD_IMG_PARENT_SUBDIR = "Defective_Road_Images"
SOURCE_NORMAL_ROAD_IMG_PARENT_SUBDIR = "Normal_Road_Images"
SOURCE_NOT_ROAD_IMG_PARENT_SUBDIR = "Not_Road_Images"
JPEGIMAGES_SUBFOLDER_NAME = "JPEGImages"  # Общее имя подпапки с картинками


def collect_image_paths_from_source():
    """Собирает абсолютные пути к изображениям из исходных директорий."""
    road_image_paths = []
    not_road_image_paths = []
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.PNG']

    # Собираем пути к изображениям дорог (с дефектами и без)
    road_source_parent_subdirs = [SOURCE_DEFECTIVE_ROAD_IMG_PARENT_SUBDIR, SOURCE_NORMAL_ROAD_IMG_PARENT_SUBDIR]
    for parent_subdir_name in road_source_parent_subdirs:
        # Путь к папке JPEGImages внутри родительской подпапки (Defective_Road_Images, Normal_Road_Images)
        images_folder_path = os.path.join(MASTER_DATASET_PATH, parent_subdir_name, JPEGIMAGES_SUBFOLDER_NAME)
        if os.path.isdir(images_folder_path):
            for ext_pattern in valid_extensions:
                road_image_paths.extend(glob.glob(os.path.join(images_folder_path, ext_pattern)))
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Директория для изображений '{parent_subdir_name}' не найдена: {images_folder_path}")

    # Собираем пути к изображениям "не дорог"
    not_road_images_folder_path = os.path.join(MASTER_DATASET_PATH, SOURCE_NOT_ROAD_IMG_PARENT_SUBDIR,
                                               JPEGIMAGES_SUBFOLDER_NAME)
    if os.path.isdir(not_road_images_folder_path):
        for ext_pattern in valid_extensions:
            not_road_image_paths.extend(glob.glob(os.path.join(not_road_images_folder_path, ext_pattern)))
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Директория для изображений 'не дорог' не найдена: {not_road_images_folder_path}")

    return road_image_paths, not_road_image_paths


def split_and_copy_files(image_paths, class_name, target_root_path, train_ratio):
    """Разделяет список путей к файлам и копирует их в целевые директории train/validation."""
    if not image_paths:
        print(f"  Информация: Нет изображений для копирования для класса '{class_name}'.")
        return 0, 0

    # Проверка на дубликаты перед перемешиванием (на всякий случай, если glob дал дубликаты из-за регистра)
    unique_image_paths = sorted(
        list(set(image_paths)))  # Сортировка для воспроизводимости random.shuffle при том же seed
    if len(unique_image_paths) != len(image_paths):
        print(
            f"  Информация: Удалено {len(image_paths) - len(unique_image_paths)} дубликатов путей для класса '{class_name}'.")

    random.seed(42)  # Для воспроизводимого разделения
    random.shuffle(unique_image_paths)

    num_train = int(len(unique_image_paths) * train_ratio)

    train_files = unique_image_paths[:num_train]
    val_files = unique_image_paths[num_train:]

    copied_train_count = 0
    copied_val_count = 0

    # Функция для копирования
    def copy_files_to_split(files_to_copy, split_name):
        nonlocal copied_train_count, copied_val_count  # Для изменения внешних счетчиков

        split_class_dir = os.path.join(target_root_path, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        copied_in_split = 0
        for file_path in files_to_copy:
            destination_path = os.path.join(split_class_dir, os.path.basename(file_path))
            # Проверка, чтобы не копировать файл сам в себя, если пути совпадают (маловероятно здесь)
            if os.path.abspath(file_path) == os.path.abspath(destination_path):
                print(f"  ПРЕДУПРЕЖДЕНИЕ: Исходный и целевой пути совпадают для {file_path}. Копирование пропущено.")
                if split_name == "train":
                    copied_train_count += 1
                elif split_name == "validation":
                    copied_val_count += 1
                copied_in_split += 1
                continue

            try:
                shutil.copy2(file_path, destination_path)  # copy2 сохраняет метаданные
                if split_name == "train":
                    copied_train_count += 1
                elif split_name == "validation":
                    copied_val_count += 1
                copied_in_split += 1
            except Exception as e:
                print(f"  ОШИБКА копирования {file_path} в {destination_path}: {e}")
        return copied_in_split

    print(f"\n  Копирование для класса '{class_name}':")
    num_copied_train = copy_files_to_split(train_files, "train")
    print(f"    Скопировано в train/{class_name}: {num_copied_train} файлов.")
    num_copied_val = copy_files_to_split(val_files, "validation")
    print(f"    Скопировано в validation/{class_name}: {num_copied_val} файлов.")

    return copied_train_count, copied_val_count


def main():
    print(f"--- Подготовка датасета для классификатора ---")
    print(f"Путь к исходному 'мастер' датасету (от друга): {MASTER_DATASET_PATH}")
    print(f"Целевой датасет для классификатора (куда будут скопированы файлы): {CLASSIFIER_TARGET_PATH_ABS}\n")

    if not os.path.isdir(MASTER_DATASET_PATH):
        print(f"ОШИБКА: Директория исходного датасета '{MASTER_DATASET_PATH}' не найдена.")
        print("Пожалуйста, проверьте путь в 'src/configs/base_config.yaml' -> 'master_dataset_path'.")
        return

    road_images, not_road_images = collect_image_paths_from_source()

    total_road_images = len(road_images)
    total_not_road_images = len(not_road_images)
    print(f"Найдено изображений 'дорог' (включая дефектные и нормальные): {total_road_images}")
    print(f"Найдено изображений 'не дорог': {total_not_road_images}")

    if total_road_images == 0 and total_not_road_images == 0:
        print("Не найдено ни одного изображения для обработки в указанных исходных директориях.")
        print(
            f"Проверьте содержимое подпапок в {MASTER_DATASET_PATH} (ожидаются {SOURCE_DEFECTIVE_ROAD_IMG_PARENT_SUBDIR}, {SOURCE_NORMAL_ROAD_IMG_PARENT_SUBDIR}, {SOURCE_NOT_ROAD_IMG_PARENT_SUBDIR} с JPEGImages внутри).")
        return

    # Очистка или создание целевой директории
    if os.path.exists(CLASSIFIER_TARGET_PATH_ABS):
        user_input = input(
            f"ПРЕДУПРЕЖДЕНИЕ: Целевая директория '{CLASSIFIER_TARGET_PATH_ABS}' уже существует. Удалить ее содержимое и создать заново? (yes/no): ").strip().lower()
        if user_input == 'yes':
            print(f"Очистка существующей целевой директории: {CLASSIFIER_TARGET_PATH_ABS}")
            shutil.rmtree(CLASSIFIER_TARGET_PATH_ABS)
            os.makedirs(CLASSIFIER_TARGET_PATH_ABS, exist_ok=True)
        else:
            print("Операция отменена пользователем. Существующая директория не будет изменена.")
            return
    else:
        os.makedirs(CLASSIFIER_TARGET_PATH_ABS, exist_ok=True)

    print("\n--- Подготовка и копирование изображений 'дорог' ---")
    split_and_copy_files(road_images, CLASS_ROAD, CLASSIFIER_TARGET_PATH_ABS, TRAIN_RATIO)

    print("\n--- Подготовка и копирование изображений 'не дорог' ---")
    split_and_copy_files(not_road_images, CLASS_NOT_ROAD, CLASSIFIER_TARGET_PATH_ABS, TRAIN_RATIO)

    print("\nПодготовка датасета для классификатора завершена.")
    print(f"Результаты находятся в: {CLASSIFIER_TARGET_PATH_ABS}")


if __name__ == "__main__":
    main()