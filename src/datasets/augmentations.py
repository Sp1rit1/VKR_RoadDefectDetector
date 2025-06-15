# RoadDefectDetector/src/datasets/augmentations.py
import albumentations as A
import cv2
import numpy as np  # Может понадобиться для border_value в Affine, если передаем цвет


def get_detector_train_augmentations(img_height, img_width):
    """
    Возвращает сбалансированную композицию аугментаций из albumentations
    для обучения детектора объектов.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,  # Правильный параметр для режима
            border_value=0,  # Правильный параметр для значения заполнения (черный)
            # Можно также (0,0,0) для RGB или np.array([0,0,0])
            # cval_mask=0, # Если бы были маски
            keep_ratio=True
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),

        # Для GaussNoise, параметр var_limit (диапазон для дисперсии) должен быть валидным.
        # Предупреждение могло быть связано с одновременным использованием mean,
        # или это особенность конкретной версии albumentations.
        # Попробуем только с var_limit.
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        A.OneOf([
            A.MotionBlur(p=0.5, blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),

    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels_for_albumentations'],
        min_visibility=0.25,
        min_area=25
    ))


if __name__ == '__main__':
    print("--- Тестирование augmentations.py (с ИСПРАВЛЕННЫМИ параметрами для Affine/GaussNoise) ---")
    # ... (остальной тестовый код из if __name__ == '__main__' блока остается таким же) ...
    # Скопируй его из предыдущей версии augmentations.py
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
        # import numpy as np # Уже импортирован в начале файла
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
            for xmin, ymin, xmax, ymax in aug_bboxes:  # Рамки уже в пикселях
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='lime', linewidth=1)
                axes[i].add_patch(rect)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nДля тестового применения аугментаций нужны numpy, Pillow и matplotlib.")
    except Exception as e:
        print(f"\nОшибка при тестовом применении аугментаций: {e}")

    print("\n--- Тестирование augmentations.py завершено ---")