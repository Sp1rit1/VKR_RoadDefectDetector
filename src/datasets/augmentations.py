# RoadDefectDetector/src/datasets/augmentations.py
import albumentations as A
import cv2  # OpenCV нужен для некоторых border_mode в albumentations


def get_detector_train_augmentations(img_height, img_width):
    """
    Возвращает сбалансированную композицию аугментаций из albumentations
    для обучения детектора объектов.
    """
    return A.Compose([
        # --- Геометрические аугментации ---
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5), # Можно добавить, если объекты не имеют строгой ориентации "верх-низ"

        # Легкий сдвиг, масштабирование и поворот
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # сдвиг изображения на +/- 6.25%
            scale_limit=0.1,  # масштабирование на +/- 10%
            rotate_limit=10,  # поворот на +/- 10 градусов
            p=0.5,  # вероятность применения
            border_mode=cv2.BORDER_CONSTANT,
            value=0  # Заполнять черным, если изображение смещается
        ),

        # --- Аугментации, меняющие цвет/яркость (умеренные) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.15,  # яркость +/- 15%
            contrast_limit=0.15,  # контраст +/- 15%
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,  # сдвиг оттенка +/- 10
            sat_shift_limit=20,  # сдвиг насыщенности +/- 20
            val_shift_limit=10,  # сдвиг значения (яркости) +/- 10
            p=0.3
        ),
        # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2), # Можно добавить, если нужно больше вариативности цвета

        # --- Аугментации, добавляющие шум/размытие (легкие) ---
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),  # Легкий гауссовский шум

        A.OneOf([
            A.MotionBlur(p=0.5, blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),  # Одна из этих трех с вероятностью 20%

        # (Опционально и осторожно) Аугментации, которые могут удалить часть информации
        # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
        # A.RandomRain(p=0.1, brightness_coefficient=0.9, drop_length=10, drop_width=1, blur_value=3),

        # Важно: Не используем аугментации, которые сильно меняют глобальную геометрию или размер
        # перед основным ресайзом в data_loader, если только это не контролируется.
        # RandomSizedBBoxSafeCrop требует, чтобы после него был Resize до целевого размера.

    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # Наши bounding box'ы в формате [xmin, ymin, xmax, ymax]
        label_fields=['class_labels_for_albumentations'],
        min_visibility=0.25,  # Минимальная видимая часть рамки после аугментации
        min_area=25  # Минимальная площадь рамки в пикселях (например, 5x5)
    ))


if __name__ == '__main__':
    print("--- Тестирование augmentations.py (с НОРМАЛЬНЫМИ параметрами) ---")

    test_height = 416
    test_width = 416

    augs = get_detector_train_augmentations(test_height, test_width)
    print(f"Создан объект аугментаций: {type(augs)}")
    print("Список трансформаций:")
    if hasattr(augs, 'transforms'):
        for t in augs.transforms:
            print(
                f"  - {t.__class__.__name__} (p={t.p if hasattr(t, 'p') else (t.always_apply if hasattr(t, 'always_apply') else 'N/A')})")

    try:
        import numpy as np
        from PIL import Image as PILImage
        import matplotlib.pyplot as plt

        dummy_array = np.zeros((test_height, test_width, 3), dtype=np.uint8)
        center_y, center_x = test_height // 2, test_width // 2
        size = 100
        dummy_array[center_y - size // 2: center_y + size // 2,
        center_x - size // 2: center_x + size // 2, 0] = 200
        dummy_array[center_y - size // 4: center_y + size // 4,
        center_x - size // 4: center_x + size // 4, 1] = 150

        dummy_bboxes = [
            [center_x - size // 2, center_y - size // 2, center_x + size // 2, center_y + size // 2],
            [center_x - size // 4, center_y - size // 4, center_x + size // 4, center_y + size // 4]
        ]
        dummy_class_labels = ['obj1', 'obj2']

        print(f"\nПрименение аугментаций к фиктивному изображению...")
        print(f"  Исходные рамки: {dummy_bboxes}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        axes[0].imshow(dummy_array);
        axes[0].set_title("Оригинал")
        for xmin, ymin, xmax, ymax in dummy_bboxes:
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='blue', linewidth=1)
            axes[0].add_patch(rect)

        for i in range(1, 6):
            augmented = augs(image=dummy_array, bboxes=dummy_bboxes, class_labels_for_albumentations=dummy_class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']

            axes[i].imshow(aug_image);
            axes[i].set_title(f"Аугментация {i}")
            for xmin, ymin, xmax, ymax in aug_bboxes:
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='lime', linewidth=1)
                axes[i].add_patch(rect)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nДля тестового применения аугментаций нужны numpy, Pillow и matplotlib.")
    except Exception as e:
        print(f"\nОшибка при тестовом применении аугментаций: {e}")

    print("\n--- Тестирование augmentations.py завершено ---")