import sys

import cv2
import yaml
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import random  # Для Python random

# --- Настройка путей и импортов ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.datasets.data_loader_v3_standard import (
        parse_voc_xml, generate_all_anchors, assign_gt_to_anchors,
        encode_box_targets, calculate_iou_matrix
    )
    from src.models.detector_v3_standard import build_detector_v3_standard  # Ожидаем модель с линейной регрессией
    from src.losses.detection_losses_v3_standard import DetectorLoss  # Ожидаем с simplified_bce_loss
    from src.utils import plot_utils
    from src.utils.postprocessing import decode_predictions, perform_nms  # Ожидает tx,ty,tw,th и якоря
    from src.datasets import augmentations
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}\nУбедитесь, что скрипт запускается из корня проекта.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Кастомный Коллбэк для Визуализации ---
class BlockingVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, image_path_str, annot_path_str, all_anchors_tf, main_config, predict_config,
                 use_augmentation_for_viz, aug_seed_for_viz, freq=10):
        super().__init__()
        self.image_path = Path(image_path_str)
        self.annot_path = Path(annot_path_str)
        self.all_anchors = all_anchors_tf  # Тензор всех якорей
        self.main_config = main_config
        self.predict_config = predict_config
        self.use_augmentation = use_augmentation_for_viz
        self.aug_seed = aug_seed_for_viz
        self.freq = freq

        self.output_dir = PROJECT_ROOT / "graphs" / "overfit_final"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir = PROJECT_ROOT / "graphs" / "overfit_progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

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

        image_to_show = image_resized
        gt_boxes_to_show = gt_boxes_resized
        gt_labels_to_show = gt_class_names_original

        if self.use_augmentation:
            if self.aug_seed is not None: np.random.seed(self.aug_seed)
            augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
            class_names_for_aug_cb = [self.main_config['class_names'][cid] if isinstance(cid, int) else cid for cid in
                                      gt_class_names_original]
            augmented = augmenter(image=image_resized, bboxes=gt_boxes_resized,
                                  class_labels_for_albumentations=class_names_for_aug_cb)
            image_to_show = augmented['image']
            gt_boxes_to_show = augmented['bboxes']
            gt_labels_to_show = augmented['class_labels_for_albumentations']

        return image_to_show, gt_boxes_to_show, gt_labels_to_show

    def on_epoch_end(self, epoch, logs=None):
        is_last_epoch = (epoch + 1) == self.params['epochs']
        if (epoch + 1) % self.freq == 0 or is_last_epoch:
            logger.info(f"\nЭпоха {epoch + 1}: Остановка для визуализации...")

            image_to_visualize, gt_boxes_to_visualize, gt_labels_to_visualize = self._prepare_image_and_gt_for_viz()
            image_for_predict_norm = image_to_visualize.astype(np.float32) / 255.0
            image_batch = tf.expand_dims(image_for_predict_norm, axis=0)
            raw_predictions_list = self.model.predict(image_batch, verbose=0)

            # decode_predictions теперь снова ожидает all_anchors
            decoded_boxes_norm, decoded_scores = decode_predictions(raw_predictions_list, self.all_anchors,
                                                                    self.main_config)
            nms_boxes, nms_scores, nms_classes, valid_detections = perform_nms(
                decoded_boxes_norm, decoded_scores, self.predict_config
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            num_dets = valid_detections[0].numpy()
            final_boxes_yxYX_norm = nms_boxes[0, :num_dets].numpy()
            final_scores_nms = nms_scores[0, :num_dets].numpy()
            final_classes_nms = nms_classes[0, :num_dets].numpy().astype(int)

            ax.set_title(f"Эпоха {epoch + 1} | Loss: {logs['loss']:.4f} | Изображение: {self.image_path.name}")
            plot_utils.plot_image(image_to_visualize, ax)
            plot_utils.plot_boxes_on_image(ax, gt_boxes_to_visualize, labels=gt_labels_to_visualize, box_type='gt',
                                           linewidth=3)

            pred_labels_nms = [self.main_config['class_names'][i] for i in final_classes_nms]
            final_boxes_xyxy_norm = final_boxes_yxYX_norm[:, [1, 0, 3, 2]]
            final_boxes_pixels = final_boxes_xyxy_norm * np.array(
                [self.main_config['input_shape'][1], self.main_config['input_shape'][0]] * 2)
            plot_utils.plot_boxes_on_image(ax, final_boxes_pixels, labels=pred_labels_nms, box_type='pred',
                                           scores=final_scores_nms, linewidth=1.5)

            save_dir = self.progress_dir if not is_last_epoch else self.output_dir
            filename = f"epoch_{epoch + 1:03d}_{self.image_path.stem}.png" if not is_last_epoch else f"overfit_test_final_prediction_{self.image_path.stem}.png"
            plot_utils.save_plot(fig, str(save_dir / filename))

            plt.show()  # Блокирующий вызов


# --- Основная функция ---
def train_on_one_image_final_attempt(main_config, predict_config, image_name, use_augmentation_for_train,
                                     seed_for_this_run, epochs=2000, initial_lr=1e-3, viz_freq=200):
    logger.info(f"--- Финальный тест переобучения для: {image_name} ---")

    if seed_for_this_run is not None:
        logger.info(f"Установка глобальных сидов на: {seed_for_this_run}")
        tf.random.set_seed(seed_for_this_run)
        np.random.seed(seed_for_this_run)
        random.seed(seed_for_this_run)

    if use_augmentation_for_train:
        logger.info(f"Аугментация для обучающего примера: ВКЛЮЧЕНА (сид: {seed_for_this_run})")
    else:
        logger.info("Аугментация для обучающего примера: ВЫКЛЮЧЕНА")

    # --- Загрузка и подготовка ОДНОГО изображения и его аннотации ---
    dataset_path = Path(main_config['dataset_path'])
    image_path_to_use = dataset_path / main_config['train_images_subdir'] / image_name
    annot_path_to_use = dataset_path / main_config['train_annotations_subdir'] / (Path(image_name).stem + ".xml")
    if not image_path_to_use.exists():
        image_path_to_use = dataset_path / main_config['val_images_subdir'] / image_name
        annot_path_to_use = dataset_path / main_config['val_annotations_subdir'] / (Path(image_name).stem + ".xml")
        if not image_path_to_use.exists():
            logger.error(f"Файл {image_name} не найден.");
            return

    image_original = cv2.cvtColor(cv2.imread(str(image_path_to_use)), cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = image_original.shape
    gt_boxes_original_pixels, gt_class_names_original = parse_voc_xml(annot_path_to_use)

    h_target, w_target = main_config['input_shape'][:2]
    image_resized_uint8 = cv2.resize(image_original, (w_target, h_target))
    gt_boxes_resized_pixels = [
        [(b[0] / w_orig) * w_target, (b[1] / h_orig) * h_target, (b[2] / w_orig) * w_target, (b[3] / h_orig) * h_target]
        for b in gt_boxes_original_pixels]

    # Применяем аугментацию, если нужно
    image_for_training_final_uint8, gt_boxes_for_training_final_pixels, gt_labels_for_training_final = image_resized_uint8, gt_boxes_resized_pixels, gt_class_names_original
    if use_augmentation_for_train:
        augmenter = augmentations.get_detector_train_augmentations(h_target, w_target)
        augmented = augmenter(image=image_resized_uint8, bboxes=gt_boxes_resized_pixels,
                              class_labels_for_albumentations=gt_class_names_original)
        image_for_training_final_uint8, gt_boxes_for_training_final_pixels, gt_labels_for_training_final = augmented[
            'image'], augmented['bboxes'], augmented['class_labels_for_albumentations']

    image_for_training_norm = (image_for_training_final_uint8.astype(np.float32) / 255.0)

    # --- Генерация y_true (целевых значений) ---
    all_anchors_np = generate_all_anchors(main_config['input_shape'], [8, 16, 32], main_config['anchor_scales'],
                                          main_config['anchor_ratios'])
    gt_boxes_norm_assign = np.array(gt_boxes_for_training_final_pixels, dtype=np.float32) / np.array(
        [w_target, h_target, w_target, h_target]) if gt_boxes_for_training_final_pixels else np.empty((0, 4))
    gt_class_ids_assign = np.array([main_config['class_names'].index(name) for name in gt_labels_for_training_final],
                                   dtype=np.int32) if gt_labels_for_training_final else np.empty((0,), dtype=np.int32)

    anchor_labels, matched_gt_boxes, matched_gt_class_ids, _ = assign_gt_to_anchors(
        gt_boxes_norm_assign, gt_class_ids_assign, all_anchors_np,
        main_config['anchor_positive_iou_threshold'], main_config['anchor_ignore_iou_threshold']
    )

    y_true_cls_flat, y_true_reg_flat = np.zeros((all_anchors_np.shape[0], main_config['num_classes']),
                                                dtype=np.float32), np.zeros((all_anchors_np.shape[0], 4),
                                                                            dtype=np.float32)
    positive_indices = np.where(anchor_labels == 1)[0]
    if len(positive_indices) > 0:
        y_true_cls_flat[positive_indices] = tf.keras.utils.to_categorical(matched_gt_class_ids[positive_indices],
                                                                          num_classes=main_config['num_classes'])
        y_true_reg_flat[positive_indices] = encode_box_targets(all_anchors_np[positive_indices],
                                                               matched_gt_boxes[positive_indices])
    y_true_cls_flat[np.where(anchor_labels == 0)[0]] = -1.0

    # --- [ИЗМЕНЕНИЕ] Разбиваем "плоские" y_true на 6 частей, как в data_loader ---
    logger.info("Разбиваем y_true на 6 частей для соответствия выходу модели...")
    anchor_counts_per_level, output_shapes_per_level = [], []
    num_base_anchors = main_config['num_anchors_per_level']
    fpn_strides = [8, 16, 32]
    for stride in fpn_strides:
        fh, fw = h_target // stride, w_target // stride
        anchor_counts_per_level.append(fh * fw * num_base_anchors)
        output_shapes_per_level.append((fh, fw, num_base_anchors))

    y_reg_split = tf.split(y_true_reg_flat, anchor_counts_per_level, axis=0)
    y_cls_split = tf.split(y_true_cls_flat, anchor_counts_per_level, axis=0)

    y_true_final_list = []
    for i in range(len(fpn_strides)):
        y_true_final_list.append(tf.reshape(y_reg_split[i], (*output_shapes_per_level[i], 4)))
    for i in range(len(fpn_strides)):
        y_true_final_list.append(tf.reshape(y_cls_split[i], (*output_shapes_per_level[i], main_config['num_classes'])))

    y_true_final_tuple = tuple(y_true_final_list)

    # --- [ИЗМЕНЕНИЕ] Создаем датасет с новой структурой y_true ---
    # Добавляем батч-измерение к каждому из 6 тензоров
    y_true_batched = tuple(tf.expand_dims(t, axis=0) for t in y_true_final_tuple)

    dataset_for_training = tf.data.Dataset.from_tensors(
        (tf.expand_dims(tf.constant(image_for_training_norm, dtype=tf.float32), axis=0),
         y_true_batched)
    ).cache().repeat()

    all_anchors_tf = tf.constant(all_anchors_np, dtype=tf.float32)

    # --- МОДЕЛЬ И ОБУЧЕНИЕ (без изменений) ---
    temp_config = main_config.copy()
    temp_config['freeze_backbone'] = True
    model = build_detector_v3_standard(temp_config)

    logger.info("Замораживаем слои BatchNormalization...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        if temp_config['freeze_backbone'] and 'efficientnetb0' in layer.name:
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    loss_fn = DetectorLoss(temp_config, all_anchors=all_anchors_np)

    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    visualizer_callback = BlockingVisualizer(
        str(image_path_to_use), str(annot_path_to_use),
        all_anchors_tf, temp_config, predict_config,
        use_augmentation_for_train,
        seed_for_this_run if use_augmentation_for_train else None,
        freq=viz_freq
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=100, min_lr=1e-7,
        verbose=1
    )

    logger.info(f"Начинаем обучение на {epochs} эпохах...")
    model.fit(dataset_for_training, epochs=epochs, steps_per_epoch=1, verbose=1,
              callbacks=[visualizer_callback, lr_scheduler])
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

    # --- Настройки для финального теста ---
    USE_AUGMENTATION = False  # ВЫКЛЮЧАЕМ АУГМЕНТАЦИЮ
    SEED = 42  # Фиксированный сид для всего, включая инициализацию весов

    EPOCHS = 2000  # Много шагов
    INITIAL_LR = 1e-3  # Начинаем с этого
    VIZ_FREQ = max(1, EPOCHS // 10)  # Визуализация ~10 раз

    # Убедимся, что в main_config правильный вес для регрессии
    main_config['loss_weights']['box_regression'] = 1.5  # или 2.0, 5.0 - можно поэкспериментировать
    logger.info(f"Для теста используется box_regression_weight: {main_config['loss_weights']['box_regression']}")

    train_on_one_image_final_attempt(
        main_config=main_config,
        predict_config=predict_config,
        image_name=IMAGE_TO_DEBUG,
        use_augmentation_for_train=USE_AUGMENTATION,
        seed_for_this_run=SEED,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        viz_freq=VIZ_FREQ
    )