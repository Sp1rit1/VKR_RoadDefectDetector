# train_yolov8.py
from ultralytics import YOLO
import os  # Добавим os для работы с путями, если нужно

if __name__ == '__main__':
    # Загружаем pre-trained модель YOLOv8 Nano.
    model = YOLO('yolov8n.pt')

    data_yaml_path = 'src/configs/road_defect_yolo_data.yaml'

    project_name = 'training_runs_yolov8'
    experiment_name = 'run1_pit_crack2'  # Убедись, что это имя соответствует последнему запуску, или задай новое

    # Запуск обучения
    # model.train() возвращает объект метрик, а не объект Results с save_dir
    metrics_results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=416,
        batch=8,
        patience=25,
        project=project_name,
        name=experiment_name,
        # workers=4,
        # cache=True
    )

    print("\nОбучение YOLOv8 завершено.")

    # Формируем путь к директории результатов на основе project и name
    # (предполагаем, что скрипт запускается из RoadDefectDetector/)
    save_directory = os.path.join(project_name, experiment_name)
    print(f"Результаты (логи, графики, веса) должны быть сохранены в директории: {save_directory}")

    # Путь к лучшей модели (стандартное имя)
    best_model_path = os.path.join(save_directory, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        print(f"Лучшая модель сохранена как: {best_model_path}")
    else:
        # Если best.pt еще не создан (например, EarlyStopping на первой эпохе или ошибка),
        # last.pt может существовать
        last_model_path = os.path.join(save_directory, 'weights', 'last.pt')
        if os.path.exists(last_model_path):
            print(f"Последняя сохраненная модель: {last_model_path}")
        else:
            print("Файл с лучшей моделью (best.pt) не найден. Проверьте папку с результатами.")

    # Вывод финальных метрик из объекта metrics_results
    if metrics_results:
        print("\n--- Финальные Метрики (из объекта, возвращенного model.train()) ---")
        try:
            print(f"  mAP50-95 (box): {metrics_results.box.map:.4f}")
            print(f"  mAP50 (box): {metrics_results.box.map50:.4f}")
            print(f"  mAP75 (box): {metrics_results.box.map75:.4f}")
            print(f"  Precision (box): {metrics_results.box.mp:.4f}")
            print(f"  Recall (box): {metrics_results.box.mr:.4f}")
            # Для F1, если нужно посчитать:
            mp = metrics_results.box.mp
            mr = metrics_results.box.mr
            f1 = 2 * (mp * mr) / (mp + mr + 1e-9) if (mp + mr) > 0 else 0.0
            print(f"  Примерный F1-score (box): {f1:.4f}")
        except AttributeError:
            print(
                "  Не удалось извлечь детальные метрики из объекта results. Проверьте вывод в консоли во время обучения или файлы в папке результатов.")
        except Exception as e:
            print(f"  Ошибка при извлечении метрик: {e}")