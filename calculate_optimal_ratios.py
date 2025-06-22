import numpy as np
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# --- Настройка путей и импортов ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import parse_voc_xml
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)


def calculate_iou_1d(box_aspect_ratio, anchor_ratios):
    intersection = np.minimum(box_aspect_ratio, anchor_ratios)
    union = np.maximum(box_aspect_ratio, anchor_ratios)
    return intersection / union


def analyze_ratios(config, k_values_to_test=range(3, 8)):
    print("--- Запуск анализа для поиска оптимальных anchor_ratios ---")

    master_dataset_path = PROJECT_ROOT / config['master_dataset_path']
    all_xml_paths = list(master_dataset_path.glob("*/*.xml")) + list(master_dataset_path.glob("*/*/*.xml"))

    if not all_xml_paths:
        print(f"ОШИБКА: Не найдены XML файлы в {master_dataset_path}")
        return

    print(f"Найдено {len(all_xml_paths)} файлов аннотаций. Идет сбор данных...")

    aspect_ratios = []
    for xml_path in tqdm(all_xml_paths, desc="Анализ аннотаций"):
        try:
            boxes, _ = parse_voc_xml(xml_path)
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                width = float(xmax - xmin)
                height = float(ymax - ymin)

                # ===> ИСПРАВЛЕНИЕ ЗДЕСЬ <===
                # Игнорируем боксы с нулевой шириной или высотой, чтобы избежать деления на ноль
                if width > 0 and height > 0:
                    aspect_ratios.append(np.log(width / height))
        except Exception:
            continue

    aspect_ratios = np.array(aspect_ratios).reshape(-1, 1)
    print(f"Собрано {len(aspect_ratios)} корректных соотношений сторон из датасета.")

    # ... (остальной код остается без изменений) ...
    avg_ious = []
    all_found_ratios = {}

    print("\n--- Поиск оптимального количества кластеров (K) ---")
    for k in tqdm(k_values_to_test, desc="Тестирование K"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(aspect_ratios)

        optimal_ratios_log = kmeans.cluster_centers_
        optimal_ratios = np.exp(optimal_ratios_log).flatten()
        optimal_ratios.sort()

        all_found_ratios[k] = optimal_ratios

        ious = []
        for gt_ratio_log in aspect_ratios:
            gt_ratio = np.exp(gt_ratio_log)
            best_iou_for_gt = np.max(calculate_iou_1d(gt_ratio, optimal_ratios))
            ious.append(best_iou_for_gt)

        avg_ious.append(np.mean(ious))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values_to_test, avg_ious, 'o-')
    plt.title("Зависимость среднего IoU от количества Ratio (K)")
    plt.xlabel("Количество соотношений сторон (K)")
    plt.ylabel("Среднее IoU (1D)")
    plt.grid(True)

    graphs_dir = PROJECT_ROOT / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    save_path = graphs_dir / "optimal_ratios_analysis.png"
    plt.savefig(save_path)
    print(f"\nГрафик анализа сохранен в: {save_path}")
    plt.show()

    best_k = k_values_to_test[np.argmax(avg_ious)]
    best_ratios = all_found_ratios[best_k]

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА И РЕКОМЕНДАЦИИ".center(80))
    print("=" * 80)

    print("\nНайденные оптимальные ratios для разного K:")
    for k, ratios in all_found_ratios.items():
        ratios_str = [f"{r:.3f}" for r in ratios]
        print(f"  K = {k}: Ratios = [{', '.join(ratios_str)}]")

    print("\n--- РЕКОМЕНДАЦИЯ ---")
    print(f"Наилучшее среднее IoU ({max(avg_ious):.4f}) достигается при K = {best_k}.")
    print("\nВставьте этот блок в ваш `detector_config_v3_standard.yaml`:")
    print("-" * 30)
    print("anchor_ratios:")
    for r in best_ratios:
        print(f"- {r:.4f}")
    print(f"\nnum_anchors_per_level: {3 * best_k} # 3 scales * {best_k} ratios")
    print("-" * 30)


if __name__ == '__main__':
    try:
        config_path = PROJECT_ROOT / "src" / "configs" / "base_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Не удалось загрузить base_config.yaml: {e}")
        sys.exit(1)

    analyze_ratios(base_config)