import sys

import cv2
import yaml
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Настройка путей и импортов ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import (
        parse_voc_xml, generate_all_anchors, assign_gt_to_anchors,
        encode_box_targets, calculate_iou_matrix
    )
    from src.models.detector_v3_standard import build_detector_v3_standard
    from src.losses.detection_losses_v3_standard import DetectorLoss
    from src.utils import plot_utils
    from src.utils.postprocessing import decode_predictions, perform_nms
    from src.datasets import augmentations
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}\nУбедитесь, что скрипт запускается из корня проекта.")
    sys.exit(1)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Кастомный Коллбэк для Визуализации ---
class BlockingVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, image_path_str, annot_path_str, all_anchors_tf, main_config, predict_config, aug_seed_for_viz,
                 freq=10):
        super().__init__()
        self.image_path = Path(image_path_str)
        self.annot_path = Path(annot_path_str)
        self.all_anchors = all_anchors_tf
        self.main_config = main_config
        self.predict_config = predict_config
        self.aug_seed = aug_seed_for_viz
        self.freq = freq

        self.output_dir = PROJECT_ROOT / "graphs" / "overfit_final"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_image_and_gt_for_viz(self):
        image_original = cv2.cvtColor(cv2.imread(str(self.image_path)), cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = image_original.shape
        gt_boxes_original, gt_class_names_original = parse_voc_xml(self.annot_path)

        h_target, w_target = self.main_config['input_shape'][:2]
        image_resized = cv2.resize(image_original, (w_target, h_target))

        gt_boxes_resized = []
        if gt_boxes_original:
            for box in gt_boxes_original:
                x1, y1, x2, y2 = box
                gt_boxes_resized.append([(x1 / w_orig) * w_target, (y1 / h_orig) * h_target, (x2 / w_orig) * w_target,
                                         (y2 / h_orig) * h_target])

        image_to_use_for_viz = image_resized
        gt_boxes_for_viz = gt_boxes_resized
        gt_labels_for_viz = gt_class_names_original

        # Аугментация для визуализации (если self.aug_seed не None)
        if self.aug_seed is not None:
            np.random.seed(self.aug_seed)  # Устанавливаем сид для воспроизводимости
            augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
            augmented = augmenter(image=image_resized, bboxes=gt_boxes_resized,
                                  class_labels_for_albumentations=gt_class_names_original)
            image_to_use_for_viz = augmented['image']
            gt_boxes_for_viz = augmented['bboxes']
            gt_labels_for_viz = augmented['class_labels_for_albumentations']

        return image_to_use_for_viz, gt_boxes_for_viz, gt_labels_for_viz

    def on_epoch_end(self, epoch, logs=None):
        is_last_epoch = (epoch + 1) == self.params['epochs']
        if (epoch + 1) % self.freq == 0 or is_last_epoch:
            logger.info(f"\nЭпоха {epoch + 1}: Остановка для визуализации. Закройте окно, чтобы продолжить...")

            image_augmented_for_plot, gt_boxes_for_plot, gt_labels_for_plot = self._prepare_image_and_gt_for_viz()

            image_for_predict_norm = image_augmented_for_plot.astype(np.float32) / 255.0
            image_batch = tf.expand_dims(image_for_predict_norm, axis=0)
            raw_predictions = self.model.predict(image_batch, verbose=0)

            decoded_boxes, decoded_scores = decode_predictions(raw_predictions, self.all_anchors, self.main_config)
            nms_boxes, nms_scores, nms_classes, valid_detections = perform_nms(
                decoded_boxes, decoded_scores, self.predict_config
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            num_dets = valid_detections[0].numpy()
            final_boxes = nms_boxes[0, :num_dets].numpy()
            final_scores = nms_scores[0, :num_dets].numpy()
            final_classes = nms_classes[0, :num_dets].numpy().astype(int)

            ax.set_title(f"Эпоха {epoch + 1} | Loss: {logs['loss']:.4f} | Изображение: {self.image_path.name}")
            plot_utils.plot_image(image_augmented_for_plot, ax)
            plot_utils.plot_boxes_on_image(ax, gt_boxes_for_plot, labels=gt_labels_for_plot, box_type='gt', linewidth=3)

            pred_labels = [self.main_config['class_names'][i] for i in final_classes]
            final_boxes_xyxy = final_boxes[:, [1, 0, 3, 2]]
            final_boxes_pixels = final_boxes_xyxy * np.array(
                [self.main_config['input_shape'][1], self.main_config['input_shape'][0]] * 2)
            plot_utils.plot_boxes_on_image(ax, final_boxes_pixels, labels=pred_labels, box_type='pred',
                                           scores=final_scores, linewidth=1.5)

            if is_last_epoch:
                save_path = self.output_dir / f"overfit_test_final_prediction_{self.image_path.stem}.png"
                plot_utils.save_plot(fig, str(save_path))
            plt.show()


# --- Основная функция ---
def train_on_one_image(main_config, predict_config, image_name, use_augmentation_for_train, fixed_aug_seed, epochs=100,
                       viz_freq=10):
    logger.info(f"--- Запуск теста на переобучение на изображении: {image_name} ---")
    if use_augmentation_for_train:
        logger.info(
            f"Аугментация для обучающего примера: ВКЛЮЧЕНА (сид: {fixed_aug_seed if fixed_aug_seed is not None else 'случайный'})")
    else:
        logger.info("Аугментация для обучающего примера: ВЫКЛЮЧЕНА")

    dataset_path = Path(main_config['dataset_path'])
    image_path_to_use = dataset_path / main_config['train_images_subdir'] / image_name
    annot_path_to_use = dataset_path / main_config['train_annotations_subdir'] / (Path(image_name).stem + ".xml")

    if not image_path_to_use.exists():
        logger.info(f"Не найдено в 'train', ищем в 'validation'...")
        image_path_to_use = dataset_path / main_config['val_images_subdir'] / image_name
        annot_path_to_use = dataset_path / main_config['val_annotations_subdir'] / (Path(image_name).stem + ".xml")
        if not image_path_to_use.exists():
            logger.error(f"Изображение '{image_name}' не найдено ни в одной из директорий.")
            return
    logger.info(f"Используется изображение: {image_path_to_use}")

    image_original = cv2.cvtColor(cv2.imread(str(image_path_to_use)), cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = image_original.shape
    gt_boxes_original, gt_class_names_original = parse_voc_xml(annot_path_to_use)

    h_target, w_target = main_config['input_shape'][:2]
    image_resized = cv2.resize(image_original, (w_target, h_target))
    gt_boxes_resized = [
        [(b[0] / w_orig) * w_target, (b[1] / h_orig) * h_target, (b[2] / w_orig) * w_target, (b[3] / h_orig) * h_target]
        for b in gt_boxes_original]

    image_for_training_final = image_resized
    gt_boxes_for_training_final = gt_boxes_resized
    gt_labels_for_training_final = gt_class_names_original

    if use_augmentation_for_train:
        if fixed_aug_seed is not None: np.random.seed(fixed_aug_seed)
        augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
        augmented = augmenter(image=image_resized, bboxes=gt_boxes_resized,
                              class_labels_for_albumentations=gt_class_names_original)
        image_for_training_final = augmented['image']
        gt_boxes_for_training_final = augmented['bboxes']
        gt_labels_for_training_final = augmented['class_labels_for_albumentations']

    image_for_training_norm = (image_for_training_final.astype(np.float32) / 255.0)

    all_anchors_np = generate_all_anchors(main_config['input_shape'], [8, 16, 32], main_config['anchor_scales'],
                                          main_config['anchor_ratios'])

    if gt_boxes_for_training_final:
        gt_boxes_norm_assign = np.array(gt_boxes_for_training_final, dtype=np.float32) / np.array(
            [w_target, h_target, w_target, h_target])
        gt_class_ids_assign = np.array(
            [main_config['class_names'].index(name) for name in gt_labels_for_training_final], dtype=np.int32)
    else:
        gt_boxes_norm_assign = np.empty((0, 4), dtype=np.float32);
        gt_class_ids_assign = np.empty((0,), dtype=np.int32)

    anchor_labels, matched_gt_boxes, _ = assign_gt_to_anchors(
        gt_boxes_norm_assign, all_anchors_np,
        main_config['anchor_positive_iou_threshold'],
        main_config['anchor_ignore_iou_threshold']
    )

    y_true_cls_np = np.zeros((len(all_anchors_np), main_config['num_classes']), dtype=np.float32)
    y_true_reg_np = np.zeros((len(all_anchors_np), 4), dtype=np.float32)
    positive_indices = np.where(anchor_labels == 1)[0]

    if len(positive_indices) > 0 and len(gt_boxes_norm_assign) > 0:
        iou_pos_anchors_vs_gt = calculate_iou_matrix(all_anchors_np[positive_indices], gt_boxes_norm_assign)
        best_gt_indices_for_pos_anchors = np.argmax(iou_pos_anchors_vs_gt, axis=1)
        positive_gt_class_ids = gt_class_ids_assign[best_gt_indices_for_pos_anchors]

        y_true_cls_np[positive_indices] = tf.keras.utils.to_categorical(positive_gt_class_ids,
                                                                        num_classes=main_config['num_classes'])
        y_true_reg_np[positive_indices] = encode_box_targets(all_anchors_np[positive_indices],
                                                             matched_gt_boxes[positive_indices])

    y_true_cls_np[np.where(anchor_labels == 0)[0]] = -1.0

    dataset_for_training = tf.data.Dataset.from_tensors(
        (tf.expand_dims(tf.constant(image_for_training_norm, dtype=tf.float32), axis=0),
         (tf.expand_dims(tf.constant(y_true_reg_np, dtype=tf.float32), axis=0),
          tf.expand_dims(tf.constant(y_true_cls_np, dtype=tf.float32), axis=0)))
    ).cache().repeat()

    logger.info(
        f"Датасет для обучения на '{image_name}' (с аугм. сидом {fixed_aug_seed if use_augmentation_for_train else 'N/A'}) успешно создан.")

    all_anchors_tf = tf.constant(all_anchors_np, dtype=tf.float32)
    main_config['freeze_backbone'] = False
    model = build_detector_v3_standard(main_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = DetectorLoss(main_config)
    model.compile(optimizer=optimizer, loss=loss_fn)

    visualizer_callback = BlockingVisualizer(
        str(image_path_to_use), str(annot_path_to_use),
        all_anchors_tf, main_config, predict_config,
        fixed_aug_seed if use_augmentation_for_train else None,  # Передаем сид в коллбэк
        freq=viz_freq
    )

    logger.info(f"Начинаем обучение на {epochs} эпохах...")
    model.fit(dataset_for_training, epochs=epochs, steps_per_epoch=1, verbose=1, callbacks=[visualizer_callback])
    logger.info("Обучение завершено.")


if __name__ == '__main__':
    try:
        config_path = PROJECT_ROOT / "src" / "configs" / "detector_config_v3_standard.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        predict_config_path = PROJECT_ROOT / "src" / "configs" / "predict_config.yaml"
        with open(predict_config_path, 'r', encoding='utf-8') as f:
            predict_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Ошибка загрузки конфига: {e}");
        sys.exit(1)

    IMAGE_TO_DEBUG = "China_Drone_000180.jpg"

    USE_AUGMENTATION_FOR_THIS_TEST = main_config.get('use_augmentation', True)
    AUG_SEED_IF_ENABLED = 42 if USE_AUGMENTATION_FOR_THIS_TEST else None

    EPOCHS_FOR_OVERFIT_TEST = 100
    VISUALIZATION_FREQUENCY = 10

    train_on_one_image(
        main_config=main_config,
        predict_config=predict_config,
        image_name=IMAGE_TO_DEBUG,
        use_augmentation_for_train=USE_AUGMENTATION_FOR_THIS_TEST,
        fixed_aug_seed=AUG_SEED_IF_ENABLED,
        epochs=EPOCHS_FOR_OVERFIT_TEST,
        viz_freq=VISUALIZATION_FREQUENCY
    )