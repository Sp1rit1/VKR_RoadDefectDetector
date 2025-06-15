# RoadDefectDetector/src/datasets/augmentations.py
import albumentations as A
import cv2
import numpy as np
from pathlib import Path


def get_detector_train_augmentations(img_height, img_width):
    """
    Возвращает сбалансированную и "мягкую" композицию аугментаций
    из albumentations, стараясь избегать UserWarnings.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-7, 7),
            shear=(-3, 3),
            p=0.5,
            # Убираем параметры цвета/заполнения границы, полагаясь на дефолты Albumentations
            # для cv2.BORDER_CONSTANT (часто это 0 - черный).
            # Если нужно будет явно, придется смотреть документацию для твоей версии
            # на предмет правильного параметра: value, cval, border_value и т.д.
            border_mode=cv2.BORDER_CONSTANT,
            keep_ratio=True
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.4
        ),

        A.HueSaturationValue(
            hue_shift_limit=7,
            sat_shift_limit=15,
            val_shift_limit=7,
            p=0.3
        ),

        # Гауссов шум - самый простой вызов, полагаемся на дефолтные значения силы шума
        A.GaussNoise(p=0.2),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=0.5),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),

    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels_for_albumentations'],
        min_visibility=0.3,
        min_area=50
    ))


# ... остальной код if __name__ == '__main__': остается таким же, как в предыдущем ответе ...
# (я его скопирую сюда для полноты, чтобы ты мог просто заменить весь файл)

if __name__ == '__main__':
    print("--- Тестирование augmentations.py (самые простые параметры для Affine/GaussNoise) ---")

    _current_script_dir = Path(__file__).parent.resolve()
    _project_root_dir = _current_script_dir.parent.parent

    test_height = 416
    test_width = 416

    augs = get_detector_train_augmentations(test_height, test_width)
    if augs is None:
        print("ОШИБКА: Функция get_detector_train_augmentations вернула None.")
        exit()

    print(f"Создан объект аугментаций: {type(augs)}")
    print("Список применяемых трансформаций:")
    if hasattr(augs, 'transforms') and augs.transforms:
        for t_idx, t_transform in enumerate(augs.transforms):
            if isinstance(t_transform, A.OneOf):
                print(
                    f"  - {t_idx + 1}. {t_transform.__class__.__name__} (p={t_transform.p if hasattr(t_transform, 'p') else 'N/A'}) containing:")
                if hasattr(t_transform, 'transforms') and t_transform.transforms:
                    for child_t in t_transform.transforms:
                        print(
                            f"    - {child_t.__class__.__name__} (p_child={child_t.p if hasattr(child_t, 'p') else 'N/A'})")
            else:
                prob_attr = 'N/A'
                if hasattr(t_transform, 'p'):
                    prob_attr = t_transform.p
                elif hasattr(t_transform, 'always_apply') and t_transform.always_apply:
                    prob_attr = 'always_apply'
                print(f"  - {t_idx + 1}. {t_transform.__class__.__name__} (p={prob_attr})")
    else:
        print("  Не удалось получить список трансформаций или он пуст.")

    try:
        import matplotlib.pyplot as plt

        dummy_array_original = np.full((test_height, test_width, 3), 128, dtype=np.uint8)
        cv2.rectangle(dummy_array_original, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.rectangle(dummy_array_original, (200, 200), (300, 350), (0, 255, 0), -1)
        cv2.rectangle(dummy_array_original, (100, 300), (180, 380), (0, 0, 255), -1)
        dummy_array_rgb_original = cv2.cvtColor(dummy_array_original, cv2.COLOR_BGR2RGB)

        dummy_bboxes_original_pixels = [
            [50, 50, 150, 150], [200, 200, 300, 350], [100, 300, 180, 380]
        ]
        dummy_class_labels_for_alb = ['obj1', 'obj2', 'obj3']

        print(f"\nПрименение аугментаций к фиктивному изображению...")
        print(f"  Исходные рамки (пиксельные): {dummy_bboxes_original_pixels}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        axes[0].imshow(dummy_array_rgb_original)
        axes[0].set_title("Оригинал")
        for xmin, ymin, xmax, ymax in dummy_bboxes_original_pixels:
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='yellow', linewidth=2)
            axes[0].add_patch(rect)
        axes[0].axis('off')

        for i in range(1, 6):
            ax_current = axes[i]
            title_prefix = f"Аугментация {i}"
            try:
                bboxes_for_this_iteration = [list(map(int, bbox)) for bbox in dummy_bboxes_original_pixels]
                labels_for_this_iteration = list(dummy_class_labels_for_alb)
                augmented_data = augs(image=dummy_array_original.copy(),
                                      bboxes=bboxes_for_this_iteration,
                                      class_labels_for_albumentations=labels_for_this_iteration)
                aug_image_bgr = augmented_data['image']
                aug_image_rgb = cv2.cvtColor(aug_image_bgr, cv2.COLOR_BGR2RGB)
                aug_bboxes_list = augmented_data['bboxes']

                ax_current.imshow(aug_image_rgb)
                ax_current.set_title(title_prefix)
                if aug_bboxes_list and len(aug_bboxes_list) > 0:
                    for bbox_coords in aug_bboxes_list:
                        if len(bbox_coords) == 4:
                            xmin, ymin, xmax, ymax = map(int, bbox_coords)
                            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='lime',
                                                 linewidth=2)
                            ax_current.add_patch(rect)
                else:
                    ax_current.text(0.5, 0.5, 'No BBoxes Left', horizontalalignment='center',
                                    verticalalignment='center', transform=ax_current.transAxes, fontsize=10)
            except Exception as e_aug_loop:  # Ловим все ошибки на этапе аугментации
                error_msg_short = f"Error in Aug {i}:\n{str(e_aug_loop)[:100]}..."
                print(f"!!! ОШИБКА в Аугментации {i}: {e_aug_loop}")
                ax_current.set_title(f"Ошибка в {title_prefix}")
                ax_current.text(0.05, 0.5, error_msg_short, fontsize=8, color='red', va='center', wrap=True)
            finally:
                ax_current.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Тест Аугментаций (Оригинал + 5 Примеров)", fontsize=16)

        graphs_dir = _project_root_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        plot_save_path_aug = graphs_dir / "augmentations_test_visualization.png"
        try:
            plt.savefig(plot_save_path_aug)
            print(f"\nГрафик теста аугментаций сохранен в: {plot_save_path_aug}")
        except Exception as e_plot_save:
            print(f"Ошибка сохранения графика аугментаций: {e_plot_save}")

        # plt.show()
    except ImportError:
        print("\nДля тестового применения аугментаций нужны numpy, matplotlib и OpenCV (cv2).")
    except Exception as e_main_test:
        print(f"\nОбщая ошибка в тестовом блоке `if __name__ == '__main__':` в файле augmentations.py: {e_main_test}")
        import traceback

        traceback.print_exc()
    print("\n--- Тестирование augmentations.py завершено ---")