import os

from ultralytics import YOLO
from PIL import Image
import cv2  # OpenCV для отображения, если нужно

# 1. Загрузка лучшей обученной модели
model_path = '../training_runs_yolov8/run1_pit_crack2/weights/best.pt'  # ЗАМЕНИ НА СВОЙ ПУТЬ К best.pt
try:
    model = YOLO(model_path)
    print(f"Модель успешно загружена из {model_path}")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

# 2. Путь к изображению, на котором хочешь сделать предсказание
image_to_predict_path = '/debug_data/China_Drone_000022.jpg'  # ЗАМЕНИ ЭТО

if not os.path.exists(image_to_predict_path):
    print(f"Ошибка: Изображение не найдено по пути {image_to_predict_path}")
    exit()

# 3. Выполнение предсказания
# Можно указать разные параметры, например:
# conf: порог уверенности (0.0 до 1.0), по умолчанию обычно 0.25
# iou: порог IoU для NMS (Non-Maximum Suppression), по умолчанию обычно 0.7 (или 0.45 для обучения)
# imgsz: размер изображения, на который оно будет изменено перед подачей в модель (например, 416 или 640)
# save: True, чтобы сохранить изображение с нарисованными рамками
# save_txt: True, чтобы сохранить .txt файл с координатами рамок в формате YOLO
# save_conf: True, чтобы в .txt файле также была уверенность

print(f"\nВыполнение предсказания для: {image_to_predict_path}")
results = model.predict(
    source=image_to_predict_path,
    imgsz=416,  # Используй тот же размер, на котором обучал, или стандартный для модели
    conf=0.25,  # Начни с этого порога, потом можно будет менять
    iou=0.45,  # Стандартный порог IoU для NMS
    save=True,  # Сохранит результат в папку runs/detect/predictX
    save_txt=True,  # Сохранит .txt файл с метками
    save_conf=True  # Добавит уверенность в .txt файл
)
print(
    f"Предсказания сохранены в директорию: {results[0].save_dir if results and results[0].save_dir else 'Не удалось определить директорию'}")

# 4. Анализ результатов (необязательно, но полезно)
# results - это список (обычно из одного элемента, если source - одно изображение)
# каждый элемент - это объект Results
if results and results[0]:
    result = results[0]
    print(f"\nНайдено объектов: {len(result.boxes)}")

    # Вывод информации о каждом найденном объекте
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls)
        class_name = model.names[class_id]  # Получаем имя класса по ID
        confidence = float(box.conf)
        coordinates_xyxy = box.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax] в пикселях исходного изображения
        coordinates_xywhn = box.xywhn[0].cpu().numpy()  # [x_center_norm, y_center_norm, width_norm, height_norm]

        print(f"  Объект {i + 1}:")
        print(f"    Класс: {class_name} (ID: {class_id})")
        print(f"    Уверенность: {confidence:.4f}")
        print(f"    Координаты (xyxy, пиксели): {coordinates_xyxy.astype(int)}")
        # print(f"    Координаты (xywh, нормализованные): {coordinates_xywhn}")

    # Изображение с нарисованными рамками уже сохранено, если save=True
    # Если хочешь отобразить его с помощью OpenCV:
    # try:
    #     img_with_boxes = result.plot() # Возвращает numpy array BGR
    #     cv2.imshow("YOLOv8 Prediction", img_with_boxes)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # except Exception as e_plot:
    #     print(f"Ошибка при отображении изображения с рамками: {e_plot}")
else:
    print("Предсказания не были получены.")