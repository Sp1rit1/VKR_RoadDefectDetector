from ultralytics import YOLO
import os  # и другие необходимые импорты


# --- ВСЕ ТВОИ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ, ФУНКЦИИ (если есть) ---
# ... (например, загрузка конфигов, определение функций и т.д.) ...

def run_evaluation():
    # --- ЗДЕСЬ ТВОЙ КОД ДЛЯ ЗАГРУЗКИ МОДЕЛИ И ВАЛИДАЦИИ ---
    model_path = '../training_runs_yolov8/run1_pit_crack2/weights/best.pt'  # ЗАМЕНИ НА СВОЙ ПУТЬ
    data_yaml_path = '../src/configs/other_configs/road_defect_yolo_data.yaml'

    try:
        model = YOLO(model_path)
        print(f"Модель {model_path} успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    print(f"\nЗапуск валидации на данных из: {data_yaml_path}")
    metrics = model.val(
        data=data_yaml_path,
        imgsz=416,
        batch=8,  # Убедись, что это значение подходит для твоей VRAM
        conf=0.001,
        iou=0.5,
        split='val',  # Явно указываем, что это валидационная выборка
        # workers=0 # <-- ВАЖНО для Windows, если другие решения не помогают
    )

    print("\n--- Метрики на Валидации ---")
    if metrics and hasattr(metrics, 'box'):
        try:
            print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
            print(f"  mAP50 (box): {metrics.box.map50:.4f}")
            print(f"  mAP75 (box): {metrics.box.map75:.4f}")
            print(f"  Precision (box): {metrics.box.mp:.4f}")
            print(f"  Recall (box): {metrics.box.mr:.4f}")
            mp = metrics.box.mp;
            mr = metrics.box.mr
            f1 = 2 * (mp * mr) / (mp + mr + 1e-9) if (mp + mr) > 0 else 0.0
            print(f"  Примерный F1-score (box): {f1:.4f}")
        except AttributeError:
            print("  Не удалось извлечь детальные метрики из объекта metrics. Проверьте вывод в консоли.")
        except Exception as e_metrics:
            print(f"  Ошибка при извлечении метрик: {e_metrics}")
    else:
        print("  Не удалось получить объект метрик.")


# --- ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == '__main__':
    # Эту строку можно добавить, если программа будет "заморожена" в .exe, но обычно она не нужна для простых скриптов.
    # from multiprocessing import freeze_support
    # freeze_support()

    run_evaluation()