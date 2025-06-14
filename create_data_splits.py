# RoadDefectDetector/create_data_splits.py
import os
import shutil
import glob
import random
import yaml
from pathlib import Path  # Для более удобной работы с путями

# --- Загрузка Конфигураций ---
_current_project_root = Path(__file__).parent.resolve()  # Корень проекта (где лежит этот скрипт)

_base_config_path = _current_project_root / 'src' / 'configs' / 'base_config.yaml'
# detector_config нам здесь не нужен, так как мы работаем с PASCAL VOC и еще не конвертировали в YOLO

try:
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict):
        print(f"ОШИБКА: base_config.yaml ({_base_config_path}) пуст или имеет неверный формат.")
        exit()
except FileNotFoundError:
    print(f"ОШИБКА: Файл base_config.yaml не найден по пути: {_base_config_path}")
    exit()
except yaml.YAMLError as e:
    print(f"ОШИБКА YAML при чтении base_config.yaml: {e}")
    exit()

# --- Параметры из Конфигов ---
MASTER_DATASET_PATH_FROM_CONFIG = BASE_CONFIG.get('master_dataset_path', '')
if not MASTER_DATASET_PATH_FROM_CONFIG:
    print("ОШИБКА: 'master_dataset_path' не указан в 'base_config.yaml'.")
    exit()

if not os.path.isabs(MASTER_DATASET_PATH_FROM_CONFIG):
    MASTER_DATASET_PATH_ABS = (_current_project_root / MASTER_DATASET_PATH_FROM_CONFIG).resolve()
else:
    MASTER_DATASET_PATH_ABS = Path(MASTER_DATASET_PATH_FROM_CONFIG).resolve()

# Имена подпапок в Master_Dataset_Path
SOURCE_SUBDIRS_KEYS = [
    'source_defective_road_img_parent_subdir',
    'source_normal_road_img_parent_subdir',
    'source_not_road_img_parent_subdir'
]
DEFAULT_SOURCE_SUBDIR_NAMES = {
    'source_defective_road_img_parent_subdir': "Defective_Road_Images",
    'source_normal_road_img_parent_subdir': "Normal_Road_Images",
    'source_not_road_img_parent_subdir': "Not_Road_Images"
}
IMAGES_SUBFOLDER_NAME_CFG = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBFOLDER_NAME_CFG = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')

# Целевая директория для разделенного датасета детектора
# Будем использовать структуру, которую ты показал: data/Detector_Dataset_Ready/
DETECTOR_TARGET_ROOT_FROM_CONFIG = "data/Detector_Dataset_Ready"  # Ты можешь вынести это в конфиг, если хочешь
DETECTOR_TARGET_ROOT_ABS = (_current_project_root / DETECTOR_TARGET_ROOT_FROM_CONFIG).resolve()

TRAIN_RATIO = 0.8  # 80% на обучение, 20% на валидацию
RANDOM_SEED = 42  # Для воспроизводимости разделения


def collect_all_master_data_paths():
    """Собирает абсолютные пути ко всем парам (изображение, XML-аннотация) из мастер-датасета."""
    all_image_annotation_pairs = []

    print(f"\n--- Сбор путей из мастер-датасета: {MASTER_DATASET_PATH_ABS} ---")

    for subfolder_key in SOURCE_SUBDIRS_KEYS:
        parent_subfolder_name = BASE_CONFIG.get(subfolder_key, DEFAULT_SOURCE_SUBDIR_NAMES.get(subfolder_key))
        if not parent_subfolder_name:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: Ключ для подпапки {subfolder_key} не найден в base_config.yaml. Пропускаем.")
            continue

        images_dir_path = MASTER_DATASET_PATH_ABS / parent_subfolder_name / IMAGES_SUBFOLDER_NAME_CFG
        annotations_dir_path = MASTER_DATASET_PATH_ABS / parent_subfolder_name / ANNOTATIONS_SUBFOLDER_NAME_CFG

        print(f"  Проверка категории: {parent_subfolder_name}")
        if not images_dir_path.is_dir():
            print(f"    ПРЕДУПРЕЖДЕНИЕ: Директория изображений {images_dir_path} не найдена. Пропускаем.")
            continue
        if not annotations_dir_path.is_dir():
            print(f"    ПРЕДУПРЕЖДЕНИЕ: Директория аннотаций {annotations_dir_path} не найдена. Пропускаем.")
            continue

        image_files_in_subdir = []
        for ext_pattern in ['*.jpg', '*.jpeg', '*.png']:  # Ищем без учета регистра через Path.glob
            image_files_in_subdir.extend(list(images_dir_path.glob(ext_pattern)))
            image_files_in_subdir.extend(list(images_dir_path.glob(ext_pattern.upper())))  # для .JPG и т.д.

        # Удаляем дубликаты, если glob нашел и .jpg и .JPG (Path.resolve() помогает)
        image_files_in_subdir = sorted(list(set([p.resolve() for p in image_files_in_subdir])))

        if not image_files_in_subdir:
            print(f"    Изображения не найдены в {images_dir_path}.")
            continue

        print(f"    Найдено {len(image_files_in_subdir)} изображений в {images_dir_path.name}")

        for img_path_obj in image_files_in_subdir:
            base_name = img_path_obj.stem  # Имя файла без расширения
            xml_path_obj = annotations_dir_path / (base_name + ".xml")

            if xml_path_obj.exists():
                all_image_annotation_pairs.append((str(img_path_obj), str(xml_path_obj)))
            else:
                print(
                    f"      ПРЕДУПРЕЖДЕНИЕ: XML-аннотация для {img_path_obj.name} не найдена в {annotations_dir_path}. Изображение будет пропущено.")

    if not all_image_annotation_pairs:
        print("\nОШИБКА: Не найдено ни одной валидной пары изображение/аннотация в мастер-датасете.")
    else:
        print(f"\nВсего найдено {len(all_image_annotation_pairs)} валидных пар изображение/аннотация.")

    return all_image_annotation_pairs


def split_and_copy_to_detector_dataset(all_pairs, target_root_dir, train_ratio, random_seed):
    """Разделяет данные и копирует их в целевые папки для детектора."""
    if not all_pairs:
        print("Нет данных для разделения.")
        return

    random.seed(random_seed)
    random.shuffle(all_pairs)

    num_train = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]

    print(f"\nРазделение датасета:")
    print(f"  Обучающая выборка: {len(train_pairs)} пар")
    print(f"  Валидационная выборка: {len(val_pairs)} пар")

    # Функция для копирования
    def copy_pairs_to_split(pairs_to_copy, split_name):
        target_images_dir = target_root_dir / split_name / IMAGES_SUBFOLDER_NAME_CFG
        target_annotations_dir = target_root_dir / split_name / ANNOTATIONS_SUBFOLDER_NAME_CFG

        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_annotations_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        print(f"\n  Копирование в '{split_name}' директории...")
        for img_src_path_str, xml_src_path_str in pairs_to_copy:
            img_src_path = Path(img_src_path_str)
            xml_src_path = Path(xml_src_path_str)

            img_dst_path = target_images_dir / img_src_path.name
            xml_dst_path = target_annotations_dir / xml_src_path.name

            try:
                shutil.copy2(img_src_path, img_dst_path)
                shutil.copy2(xml_src_path, xml_dst_path)
                copied_count += 1
            except Exception as e:
                print(f"    ОШИБКА копирования пары ({img_src_path.name}, {xml_src_path.name}): {e}")
        print(f"    Скопировано {copied_count} пар в '{split_name}'.")

    # Очистка или запрос на очистку целевой директории
    if DETECTOR_TARGET_ROOT_ABS.exists():
        user_input = input(
            f"ПРЕДУПРЕЖДЕНИЕ: Целевая директория '{DETECTOR_TARGET_ROOT_ABS}' уже существует. "
            f"Удалить ее содержимое и создать заново? (yes/no): "
        ).strip().lower()
        if user_input == 'yes':
            print(f"Очистка существующей целевой директории: {DETECTOR_TARGET_ROOT_ABS}")
            shutil.rmtree(DETECTOR_TARGET_ROOT_ABS)
            DETECTOR_TARGET_ROOT_ABS.mkdir(parents=True, exist_ok=True)
        else:
            print("Операция отменена пользователем. Существующая директория не будет изменена.")
            return
    else:
        DETECTOR_TARGET_ROOT_ABS.mkdir(parents=True, exist_ok=True)

    copy_pairs_to_split(train_pairs, "train")
    copy_pairs_to_split(val_pairs, "validation")

    print("\n--- Разделение и копирование данных для детектора завершено ---")
    print(f"Результаты находятся в: {DETECTOR_TARGET_ROOT_ABS}")


if __name__ == "__main__":
    print("--- Запуск скрипта create_data_splits.py ---")

    # Проверка существования мастер-датасета
    if not MASTER_DATASET_PATH_ABS.is_dir():
        print(f"\nОШИБКА: Директория мастер-датасета '{MASTER_DATASET_PATH_ABS}' не найдена.")
        print("Пожалуйста, проверьте путь в 'src/configs/base_config.yaml' -> 'master_dataset_path'.")
        print("Убедитесь, что датасет от друга (с подпапками Defective_Road_Images и т.д.) находится по этому пути.")
    else:
        all_data_pairs = collect_all_master_data_paths()
        if all_data_pairs:
            split_and_copy_to_detector_dataset(all_data_pairs, DETECTOR_TARGET_ROOT_ABS, TRAIN_RATIO, RANDOM_SEED)
        else:
            print("\nНе удалось собрать данные из мастер-датасета. Разделение не будет выполнено.")
            print("Проверьте структуру вашего мастер-датасета и пути в конфигурационных файлах.")

    print("\n--- Скрипт create_data_splits.py завершил работу ---")