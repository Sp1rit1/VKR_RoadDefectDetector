# RoadDefectDetector/train_detector.py
import random

import cv2
import tensorflow as tf
import yaml
import os
import datetime
import glob
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# --- Определяем корень проекта и добавляем src в sys.path ---
_project_root = Path(__file__).resolve().parent
_src_path = _project_root / 'src'
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# --- Импорты из твоих модулей в src ---
from datasets.other_loaders.detector_data_loader import (
    create_detector_tf_dataset,
    FPN_LEVELS_CONFIG_GLOBAL as DDL_FPN_LEVELS_CONFIG,
    CLASSES_LIST_GLOBAL_FOR_DETECTOR as DDL_CLASSES_LIST,
    NUM_CLASSES_DETECTOR as DDL_NUM_CLASSES,
    FPN_LEVEL_NAMES_ORDERED as DDL_FPN_LEVEL_NAMES,
    TARGET_IMG_HEIGHT as DDL_TARGET_IMG_HEIGHT,  # Для передачи в коллбэк
    TARGET_IMG_WIDTH as DDL_TARGET_IMG_WIDTH, parse_xml_annotation  # Для передачи в коллбэк
)
from models.other_models.object_detector import build_object_detector_v2_fpn
from losses.other_losses.detection_losses import compute_detector_loss_v2_fpn

# --- Попытка импорта функций для инференса и визуализации ---
PREDICT_FUNCS_LOADED_FOR_CALLBACK = False
PLOT_UTILS_LOADED_FOR_CALLBACK = False

# Для импорта из run_prediction_pipeline.py, который в корне проекта
if str(_project_root) not in sys.path:
    sys.path.insert(1, str(_project_root))  # Вставляем после src, чтобы не было конфликтов

try:
    from run_prediction_pipeline import (
        decode_single_level_predictions_generic,
        apply_nms_and_filter_generic,
        preprocess_image_for_model_tf  # Нужна для предсказаний в коллбэке
    )

    PREDICT_FUNCS_LOADED_FOR_CALLBACK = True
    print("INFO (train_detector.py): Функции инференса (decode, nms, preprocess) импортированы.")
except ImportError as e_imp_pred_train:
    print(f"ПРЕДУПРЕЖДЕНИЕ (train_detector.py): Не удалось импортировать функции инференса: {e_imp_pred_train}")


    # Заглушки, чтобы скрипт мог работать без этих функций в коллбэке
    def preprocess_image_for_model_tf(*args, **kwargs):
        return None


    def decode_single_level_predictions_generic(*args, **kwargs):
        return None, None, None


    def apply_nms_and_filter_generic(*args, **kwargs):
        return None, None, None, tf.constant([0])

try:
    from utils.plot_utils import visualize_fpn_detections_vs_gt

    PLOT_UTILS_LOADED_FOR_CALLBACK = True
    print("INFO (train_detector.py): Функция визуализации visualize_fpn_detections_vs_gt импортирована.")
except ImportError as e_imp_plot_train:
    PLOT_UTILS_LOADED_FOR_CALLBACK = False
    print(f"ПРЕДУПРЕЖДЕНИЕ (train_detector.py): Не удалось импортировать plot_utils: {e_imp_plot_train}")


    def visualize_fpn_detections_vs_gt(*args, **kwargs):
        print("  (Визуализация предсказаний недоступна)")

# --- Загрузка Конфигураций ---
# ... (код загрузки BASE_CONFIG и DETECTOR_CONFIG как был) ...
_base_config_path = _src_path / 'configs' / 'base_config.yaml'
_detector_config_path = _src_path / 'configs' / 'detector_config.yaml'
BASE_CONFIG = {};
DETECTOR_CONFIG = {};
CONFIG_LOAD_SUCCESS_TRAIN_DET = True
try:  # Base Config
    with open(_base_config_path, 'r', encoding='utf-8') as f:
        BASE_CONFIG = yaml.safe_load(f)
    if not isinstance(BASE_CONFIG, dict): BASE_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN_DET = False
except Exception:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False
try:  # Detector Config
    with open(_detector_config_path, 'r', encoding='utf-8') as f:
        DETECTOR_CONFIG = yaml.safe_load(f)
    if not isinstance(DETECTOR_CONFIG, dict): DETECTOR_CONFIG = {}; CONFIG_LOAD_SUCCESS_TRAIN_DET = False
except Exception:
    CONFIG_LOAD_SUCCESS_TRAIN_DET = False
if not CONFIG_LOAD_SUCCESS_TRAIN_DET:
    print("ОШИБКА: Загрузка конфигов в train_detector.py не удалась. Выход.");
    exit()

# --- Параметры из Конфигов ---
# ... (все твои переменные: IMAGES_SUBDIR_NAME_DET, ..., USE_AUGMENTATION_TRAIN_CFG) ...
IMAGES_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('images_dir', 'JPEGImages')
ANNOTATIONS_SUBDIR_NAME_DET = BASE_CONFIG.get('dataset', {}).get('annotations_dir', 'Annotations')
_detector_dataset_ready_path_rel = "../data/Detector_Dataset_Ready"
DETECTOR_DATASET_READY_ABS = _project_root / _detector_dataset_ready_path_rel
_fpn_params_train_global_vars = DETECTOR_CONFIG.get('fpn_detector_params',
                                                    {})  # Изменил имя, чтобы не конфликтовать с локальной в коллбэке
MODEL_BASE_NAME_CFG = _fpn_params_train_global_vars.get('model_name_prefix', 'RoadDefectDetector_v2_FPN_Default')
BACKBONE_LAYER_NAME_IN_MODEL_CFG = _fpn_params_train_global_vars.get('backbone_layer_name_in_model',
                                                                     'Backbone_MobileNetV2')
CONTINUE_FROM_CHECKPOINT_CFG = DETECTOR_CONFIG.get('continue_from_checkpoint', False)
PATH_TO_CHECKPOINT_REL_CFG = DETECTOR_CONFIG.get('path_to_checkpoint', None)
UNFREEZE_BACKBONE_CFG = DETECTOR_CONFIG.get('unfreeze_backbone', False)
UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG = DETECTOR_CONFIG.get('unfreeze_backbone_layers_from_end', 0)
FINETUNE_KEEP_BN_FROZEN_CFG = DETECTOR_CONFIG.get('finetune_keep_bn_frozen', True)
BATCH_SIZE_CFG = DETECTOR_CONFIG.get('batch_size', 4)
EPOCHS_CFG = DETECTOR_CONFIG.get('epochs', 150)
INITIAL_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('initial_learning_rate', 0.0001)
FINETUNE_LEARNING_RATE_CFG = DETECTOR_CONFIG.get('finetune_learning_rate', 1e-5)
EARLY_STOPPING_PATIENCE_CFG = DETECTOR_CONFIG.get('early_stopping_patience', 25)
REDUCE_LR_PATIENCE_CFG = DETECTOR_CONFIG.get('reduce_lr_patience', 10)
REDUCE_LR_FACTOR_CFG = DETECTOR_CONFIG.get('reduce_lr_factor', 0.2)
MIN_LR_ON_PLATEAU_CFG = DETECTOR_CONFIG.get('min_lr_on_plateau', 5e-8)
USE_AUGMENTATION_TRAIN_CFG = DETECTOR_CONFIG.get('use_augmentation', True)
FOCAL_LOSS_PARAMS_TRAIN = _fpn_params_train_global_vars.get('focal_loss_objectness_params', {})
USE_FOCAL_FOR_OBJECTNESS_TRAIN = FOCAL_LOSS_PARAMS_TRAIN.get('use_focal_loss', True)
FOCAL_ALPHA_TRAIN = FOCAL_LOSS_PARAMS_TRAIN.get('alpha', 0.25)
FOCAL_GAMMA_TRAIN = FOCAL_LOSS_PARAMS_TRAIN.get('gamma', 2.0)
LOGS_BASE_DIR_ABS = _project_root / BASE_CONFIG.get('logs_base_dir', 'logs')
WEIGHTS_BASE_DIR_ABS = _project_root / BASE_CONFIG.get('weights_base_dir', 'weights')
GRAPHS_DIR_ABS = _project_root / BASE_CONFIG.get('graphs_dir', 'graphs')
os.makedirs(GRAPHS_DIR_ABS, exist_ok=True);
os.makedirs(WEIGHTS_BASE_DIR_ABS, exist_ok=True);
os.makedirs(LOGS_BASE_DIR_ABS, exist_ok=True)
CALLBACK_CONF_THRESH = DETECTOR_CONFIG.get('predict_params', {}).get('confidence_threshold', 0.1)
CALLBACK_IOU_THRESH_NMS = DETECTOR_CONFIG.get('predict_params', {}).get('iou_threshold', 0.45)
CALLBACK_MAX_DETS = DETECTOR_CONFIG.get('predict_params', {}).get('max_detections', 100)


# --- Вспомогательные Функции ---
# ... (collect_split_data_paths, EpochTimeLogger, plot_training_history - КАК БЫЛИ)
def collect_split_data_paths(split_dir_abs_path, images_subdir, annotations_subdir):
    image_paths = [];
    xml_paths = []
    current_images_dir = os.path.join(split_dir_abs_path, images_subdir)
    current_annotations_dir = os.path.join(split_dir_abs_path, annotations_subdir)
    if not os.path.isdir(current_images_dir) or not os.path.isdir(
        current_annotations_dir): return image_paths, xml_paths
    valid_extensions = ['.jpg', '.jpeg', '.png'];
    image_files_in_split = []
    for ext in valid_extensions:
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.lower()}")))
        image_files_in_split.extend(glob.glob(os.path.join(current_images_dir, f"*{ext.upper()}")))
    image_files_in_split = sorted(list(set(image_files_in_split)))
    for img_path in image_files_in_split:
        base_name, _ = os.path.splitext(os.path.basename(img_path))
        xml_path = os.path.join(current_annotations_dir, base_name + ".xml")
        if os.path.exists(xml_path): image_paths.append(img_path); xml_paths.append(xml_path)
    return image_paths, xml_paths


class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        lr_val = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        val_loss_val = logs.get('val_loss', float('inf'))
        print(f" - ValLoss: {val_loss_val:.4f} - LR: {lr_val:.1e} - Время: {epoch_duration:.2f} сек")
        if logs is not None: logs['epoch_duration_sec'] = epoch_duration


def plot_training_history(history, save_path_plot, run_suffix=""):
    plt.figure(figsize=(12, 5));
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss']:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    title = f'История обучения ({run_suffix})' if run_suffix else 'История обучения детектора'
    plt.title(title);
    plt.ylabel('Loss');
    plt.xlabel('Epoch');
    plt.legend(loc='upper right');
    plt.grid(True)
    plt.tight_layout();
    try:
        plt.savefig(save_path_plot); print(f"График сохранен: {save_path_plot}")
    except Exception as e:
        print(f"ОШИБКА сохранения графика: {e}"); plt.close()


# --- Кастомный Коллбэк для Детального Логирования ---
class FullDetailEpochEndLogger(tf.keras.callbacks.Callback):
    def __init__(self,
                 val_image_paths_list_cb,
                 val_xml_paths_list_cb,
                 fpn_level_names_arg,
                 fpn_configs_arg,
                 classes_list_arg,
                 num_classes_arg,
                 target_h_arg,
                 target_w_arg,
                 log_freq_epochs=1,
                 num_samples_to_visualize=2,
                 conf_thresh_vis=CALLBACK_CONF_THRESH,
                 iou_thresh_nms_vis=CALLBACK_IOU_THRESH_NMS,
                 max_dets_vis=CALLBACK_MAX_DETS,
                 enable_visualization_arg=None  # Принимаем None, если ключ не найден
                 ):
        super().__init__()
        self.val_image_paths = val_image_paths_list_cb
        self.val_xml_paths = val_xml_paths_list_cb
        self.log_freq_epochs = log_freq_epochs
        self.num_samples_to_visualize = num_samples_to_visualize

        self.conf_thresh_vis = conf_thresh_vis
        self.iou_thresh_nms_vis = iou_thresh_nms_vis
        self.max_dets_vis = max_dets_vis

        self.fpn_level_names = fpn_level_names_arg
        self.fpn_configs = fpn_configs_arg
        self.classes_list = classes_list_arg
        self.num_classes = num_classes_arg
        self.target_h = target_h_arg
        self.target_w = target_w_arg

        self.use_focal_for_obj_cb_val = USE_FOCAL_FOR_OBJECTNESS_TRAIN
        self.focal_alpha_cb_val = FOCAL_ALPHA_TRAIN
        self.focal_gamma_cb_val = FOCAL_GAMMA_TRAIN

        self.predict_funcs_available_cb = PREDICT_FUNCS_LOADED_FOR_CALLBACK
        self.plot_utils_available_cb = PLOT_UTILS_LOADED_FOR_CALLBACK

        # Устанавливаем self.enable_visualization:
        # Если enable_visualization_arg передан (не None), используем его.
        # Иначе (если ключ в конфиге отсутствовал и был передан None), используем дефолт (например, False).
        if enable_visualization_arg is not None:
            self.enable_visualization = bool(enable_visualization_arg)  # Приводим к bool на всякий случай
        else:
            self.enable_visualization = False  # Дефолтное значение, если в конфиге нет и не передано

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq_epochs != 0:
            return

        print(f"\n--- Детальный Анализ в Конце Эпохи {epoch + 1} ---")
        logs = logs or {}
        current_lr_cb_val = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"  Текущий Learning Rate: {current_lr_cb_val:.3e}")
        print(f"  Keras Logs: Train Loss={logs.get('loss'):.4f}, Val Loss={logs.get('val_loss'):.4f}")

        # 1. Детальные компоненты потерь
        if self.val_image_paths and len(self.val_image_paths) > 0:
            # ... (код расчета детальных потерь остается таким же) ...
            print("  Расчет детальных потерь на случайных примерах из валидации...")
            original_debug_env_var_cb_val = os.environ.get("DEBUG_TRAINING_LOOP_ACTIVE")
            os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] = "1"
            num_samples_for_loss_calc = min(3 * BATCH_SIZE_CFG, len(self.val_image_paths))
            if num_samples_for_loss_calc > 0:
                selected_indices_loss_cb = random.sample(range(len(self.val_image_paths)), num_samples_for_loss_calc)
                loss_calc_img_paths_cb = [self.val_image_paths[i] for i in selected_indices_loss_cb]
                loss_calc_xml_paths_cb = [self.val_xml_paths[i] for i in selected_indices_loss_cb]
                temp_val_loss_dataset_cb = create_detector_tf_dataset(loss_calc_img_paths_cb, loss_calc_xml_paths_cb,
                                                                      batch_size=BATCH_SIZE_CFG, shuffle=False,
                                                                      augment=False)
                num_batches_to_avg_loss = 3;
                temp_val_loss_dataset_cb = temp_val_loss_dataset_cb.take(num_batches_to_avg_loss)
                all_batch_detailed_losses_cb_val = []
                try:
                    for val_images_cb_loss, val_y_true_tuple_cb_loss in temp_val_loss_dataset_cb:
                        val_y_pred_list_cb_loss = self.model(val_images_cb_loss, training=False)
                        detailed_loss_batch = compute_detector_loss_v2_fpn(val_y_true_tuple_cb_loss,
                                                                           val_y_pred_list_cb_loss,
                                                                           use_focal_for_obj_param=self.use_focal_for_obj_cb_val,
                                                                           focal_alpha_param=self.focal_alpha_cb_val,
                                                                           focal_gamma_param=self.focal_gamma_cb_val)
                        if isinstance(detailed_loss_batch, dict): all_batch_detailed_losses_cb_val.append(
                            detailed_loss_batch)
                except Exception as e_loss_cb_calc:
                    print(f"    ОШИБКА при расчете потерь в коллбэке: {e_loss_cb_calc}")
                if all_batch_detailed_losses_cb_val:
                    avg_detailed_losses_cb_val = {};
                    for key_cb_val in all_batch_detailed_losses_cb_val[0].keys():
                        valid_tensors = [d_cb_val[key_cb_val] for d_cb_val in all_batch_detailed_losses_cb_val if
                                         hasattr(d_cb_val[key_cb_val], 'numpy')]
                        if valid_tensors: avg_detailed_losses_cb_val[key_cb_val] = np.mean(
                            [t.numpy() for t in valid_tensors])
                    print("  Усредненные детальные VAL потерь (из коллбэка):")
                    for k_cb_val, v_avg_cb_val in avg_detailed_losses_cb_val.items():
                        print(f"    val_{k_cb_val}_cb: {v_avg_cb_val:.4f}")
                        if logs is not None: logs[f'val_{k_cb_val}_cb'] = v_avg_cb_val
            if original_debug_env_var_cb_val is None:
                if "DEBUG_TRAINING_LOOP_ACTIVE" in os.environ: del os.environ["DEBUG_TRAINING_LOOP_ACTIVE"]
            else:
                os.environ["DEBUG_TRAINING_LOOP_ACTIVE"] = original_debug_env_var_cb_val

        # 2. Визуализация предсказаний
        if self.enable_visualization and self.predict_funcs_available_cb and self.plot_utils_available_cb and self.val_image_paths and self.num_samples_to_visualize > 0:
            # ... (остальной код визуализации как в твоей последней версии, он использует self.enable_visualization) ...
            # ВАЖНО: Убедись, что вызов plt.show() внутри visualize_fpn_detections_vs_gt закомментирован,
            # а сама функция возвращает объект fig.
            print(f"\n  Визуализация предсказаний на {self.num_samples_to_visualize} случайных валидационных примерах:")
            num_to_show_actually_vis_cb = min(self.num_samples_to_visualize, len(self.val_image_paths))
            if num_to_show_actually_vis_cb > 0:
                selected_indices_vis_cb_vis = random.sample(range(len(self.val_image_paths)),
                                                            num_to_show_actually_vis_cb)
                for vis_idx_cb_vis in selected_indices_vis_cb_vis:
                    img_path_vis_cb = self.val_image_paths[vis_idx_cb_vis];
                    xml_path_vis_cb = self.val_xml_paths[vis_idx_cb_vis]
                    print(f"    Обработка для визуализации: {os.path.basename(img_path_vis_cb)}")
                    try:
                        original_bgr_img_cb_vis = cv2.imread(img_path_vis_cb)
                        if original_bgr_img_cb_vis is None: continue
                        temp_gt_dataset_vis = create_detector_tf_dataset([img_path_vis_cb], [xml_path_vis_cb],
                                                                         batch_size=1, shuffle=False, augment=False)
                        processed_img_for_gt_vis_tensor_batch, y_true_tuple_for_gt_vis_tensor_batch = next(
                            iter(temp_gt_dataset_vis))
                        processed_img_for_gt_vis_tensor = processed_img_for_gt_vis_tensor_batch[0]
                        y_true_tuple_for_gt_vis_tensor = (
                        y_true_tuple_for_gt_vis_tensor_batch[0][0], y_true_tuple_for_gt_vis_tensor_batch[1][0],
                        y_true_tuple_for_gt_vis_tensor_batch[2][0])
                        detector_input_batch_vis_cb = preprocess_image_for_model_tf(original_bgr_img_cb_vis,
                                                                                    self.target_h, self.target_w)
                        raw_model_preds_list_vis_cb = self.model.predict(detector_input_batch_vis_cb, verbose=0)
                        all_lvl_boxes_cb_vis, all_lvl_obj_cb_vis, all_lvl_cls_cb_vis = [], [], []
                        for i_lvl_vis_cb, lvl_key_vis_cb in enumerate(self.fpn_level_names):
                            raw_preds_lvl_cb_vis = raw_model_preds_list_vis_cb[i_lvl_vis_cb];
                            lvl_cfg_vis_cb = self.fpn_configs.get(lvl_key_vis_cb)
                            if lvl_cfg_vis_cb is None: continue
                            decoded_boxes_level, obj_confidence_level, class_probs_level = decode_single_level_predictions_generic(
                                raw_preds_lvl_cb_vis, lvl_cfg_vis_cb['anchors_wh_normalized'], lvl_cfg_vis_cb['grid_h'],
                                lvl_cfg_vis_cb['grid_w'], self.num_classes)
                            all_lvl_boxes_cb_vis.append(decoded_boxes_level);
                            all_lvl_obj_cb_vis.append(obj_confidence_level);
                            all_lvl_cls_cb_vis.append(class_probs_level)
                        if all_lvl_boxes_cb_vis:
                            nms_b, nms_s, nms_c, num_v = apply_nms_and_filter_generic(all_lvl_boxes_cb_vis,
                                                                                      all_lvl_obj_cb_vis,
                                                                                      all_lvl_cls_cb_vis,
                                                                                      self.num_classes,
                                                                                      self.conf_thresh_vis,
                                                                                      self.iou_thresh_nms_vis,
                                                                                      self.max_dets_vis)
                            num_preds_to_show_cb = int(num_v[0].numpy())
                            pred_boxes_for_viz_cb = nms_b[0][:num_preds_to_show_cb].numpy();
                            pred_scores_for_viz_cb = nms_s[0][:num_preds_to_show_cb].numpy();
                            pred_classes_for_viz_cb = nms_c[0][:num_preds_to_show_cb].numpy().astype(int)
                            gt_objs_orig_viz, orig_w_xml, orig_h_xml, _ = parse_xml_annotation(xml_path_vis_cb,
                                                                                               self.classes_list)
                            orig_gt_boxes_norm_for_ref_list = [];
                            orig_gt_ids_for_ref_list = []
                            if gt_objs_orig_viz:
                                img_w_to_use_ref = orig_w_xml if orig_w_xml and orig_w_xml > 0 else \
                                original_bgr_img_cb_vis.shape[1]
                                img_h_to_use_ref = orig_h_xml if orig_h_xml and orig_h_xml > 0 else \
                                original_bgr_img_cb_vis.shape[0]
                                for obj_gt_orig in gt_objs_orig_viz:
                                    if img_w_to_use_ref > 0 and img_h_to_use_ref > 0:
                                        xmin_n = obj_gt_orig['xmin'] / img_w_to_use_ref;
                                        ymin_n = obj_gt_orig['ymin'] / img_h_to_use_ref;
                                        xmax_n = obj_gt_orig['xmax'] / img_w_to_use_ref;
                                        ymax_n = obj_gt_orig['ymax'] / img_h_to_use_ref
                                        orig_gt_boxes_norm_for_ref_list.append([xmin_n, ymin_n, xmax_n, ymax_n]);
                                        orig_gt_ids_for_ref_list.append(obj_gt_orig['class_id'])
                            original_gt_tuple_for_viz = (np.array(orig_gt_boxes_norm_for_ref_list, dtype=np.float32),
                                                         np.array(orig_gt_ids_for_ref_list, dtype=np.int32))

                            fig_returned = visualize_fpn_detections_vs_gt(
                                image_np_processed=processed_img_for_gt_vis_tensor.numpy(),
                                y_true_fpn_tuple_np=(
                                y_true_tuple_for_gt_vis_tensor[0].numpy(), y_true_tuple_for_gt_vis_tensor[1].numpy(),
                                y_true_tuple_for_gt_vis_tensor[2].numpy()),
                                pred_boxes_norm_yxyx=pred_boxes_for_viz_cb, pred_scores=pred_scores_for_viz_cb,
                                pred_class_ids=pred_classes_for_viz_cb,
                                fpn_level_names=self.fpn_level_names, fpn_configs=self.fpn_configs,
                                classes_list=self.classes_list,
                                original_gt_boxes_for_reference=original_gt_tuple_for_viz,
                                title_prefix=f"Epoch {epoch + 1} - {os.path.basename(img_path_vis_cb)}")
                            plt.show()

                    except Exception as e_vis_cb_loop:
                        print(
                            f"    ОШИБКА виз. в коллбэке для {os.path.basename(img_path_vis_cb)}: {e_vis_cb_loop}"); import \
                            traceback; traceback.print_exc()
        print(f"--- Конец Детального Лога Эпохи {epoch + 1} ---")


# --- Основная функция обучения train_detector_main() ---
def train_detector_main():
    print(
        f"\n--- Запуск Обучения Детектора Объектов (Версия с Улучшенным Управлением Fine-tuning'ом и Детальным Логгингом) ---")

    # 1. Определение режима и параметров
    training_run_description = "initial_frozen_bb";
    initial_epoch = 0  # ИСПРАВЛЕНО: Определяем здесь
    current_lr = INITIAL_LEARNING_RATE_CFG;
    current_use_augmentation_train = USE_AUGMENTATION_TRAIN_CFG
    model_to_load_path_abs = None;
    perform_fine_tuning = False

    if PATH_TO_CHECKPOINT_REL_CFG: model_to_load_path_abs = WEIGHTS_BASE_DIR_ABS / PATH_TO_CHECKPOINT_REL_CFG

    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path_abs and model_to_load_path_abs.exists():
        print(f"Режим: Продолжение обучения / Fine-tuning с чекпоинта: {model_to_load_path_abs}")
        if UNFREEZE_BACKBONE_CFG:
            training_run_description = "finetune_bb";
            current_lr = FINETUNE_LEARNING_RATE_CFG;
            perform_fine_tuning = True
            print(f"  Backbone будет разморожен. Learning rate: {current_lr}")
        else:
            training_run_description = "continued_frozen_bb"
            print(f"  Backbone останется замороженным. Learning rate: {current_lr}")
    elif UNFREEZE_BACKBONE_CFG:
        training_run_description = "initial_unfrozen_bb_WARNING";
        current_lr = FINETUNE_LEARNING_RATE_CFG;
        perform_fine_tuning = True
        print(f"ПРЕДУПРЕЖДЕНИЕ: Начальное обучение с РАЗМОРОЖЕННЫМ backbone. LR: {current_lr}")
    else:
        print(f"Режим: Начальное обучение новой модели с ЗАМОРОЖЕННЫМ backbone. LR: {current_lr}")

    print(f"  Итоговый режим обучения для этого запуска: {training_run_description}")
    print(f"  Используемый Learning Rate для компиляции: {current_lr}")
    print(f"  Аугментация для обучающей выборки: {current_use_augmentation_train}")

    # 2. Подготовка данных
    train_split_dir = DETECTOR_DATASET_READY_ABS / "train"
    val_split_dir = DETECTOR_DATASET_READY_ABS / "validation"
    # ИСПРАВЛЕНО: val_image_paths и val_xml_paths теперь определяются здесь и видны коллбэку
    global val_image_paths, val_xml_paths  # Делаем их доступными для коллбэка, если он не вложенный
    train_image_paths, train_xml_paths = collect_split_data_paths(str(train_split_dir), IMAGES_SUBDIR_NAME_DET,
                                                                  ANNOTATIONS_SUBDIR_NAME_DET)
    val_image_paths, val_xml_paths = collect_split_data_paths(str(val_split_dir), IMAGES_SUBDIR_NAME_DET,
                                                              ANNOTATIONS_SUBDIR_NAME_DET)
    # ... (остальной код подготовки данных train_dataset, validation_dataset как был) ...
    if not train_image_paths: print("ОШИБКА: Обучающие данные не найдены."); return
    print(f"\nНайдено для обучения: {len(train_image_paths)}.");
    validation_dataset = None  # Инициализируем
    if val_image_paths:
        print(f"Найдено для валидации: {len(val_image_paths)}.")
        validation_dataset = create_detector_tf_dataset(val_image_paths, val_xml_paths,
                                                        batch_size=BATCH_SIZE_CFG, shuffle=False, augment=False)
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Валидационные данные не найдены.")
    print(f"\nСоздание TensorFlow датасетов...");
    train_dataset = create_detector_tf_dataset(train_image_paths, train_xml_paths,
                                               batch_size=BATCH_SIZE_CFG, shuffle=True,
                                               augment=current_use_augmentation_train)
    if train_dataset is None: print("Не удалось создать обучающий датасет. Выход."); return

    # 3. Создание или Загрузка Модели и ее Компиляция
    # ... (код создания/загрузки модели, разморозки, model.summary() и компиляции как был) ...
    model = None
    if CONTINUE_FROM_CHECKPOINT_CFG and model_to_load_path_abs and model_to_load_path_abs.exists():
        try:
            model = tf.keras.models.load_model(str(model_to_load_path_abs),
                                               custom_objects={
                                                   'compute_detector_loss_v2_fpn': compute_detector_loss_v2_fpn},
                                               compile=False)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}. Будет создана новая."); model = None
    if model is None:
        model = build_object_detector_v2_fpn(
            force_freeze_backbone_arg=(not perform_fine_tuning if not CONTINUE_FROM_CHECKPOINT_CFG else None))
    if model and perform_fine_tuning:
        try:
            backbone_model_internal = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)
            backbone_model_internal.trainable = True
            num_layers_to_finetune_from_end = UNFREEZE_BACKBONE_LAYERS_FROM_END_CFG
            if num_layers_to_finetune_from_end > 0:
                for i_bb_layer, layer_bb in enumerate(backbone_model_internal.layers):
                    layer_bb.trainable = (
                                i_bb_layer >= len(backbone_model_internal.layers) - num_layers_to_finetune_from_end)
                    if FINETUNE_KEEP_BN_FROZEN_CFG and isinstance(layer_bb,
                                                                  tf.keras.layers.BatchNormalization): layer_bb.trainable = False
            elif FINETUNE_KEEP_BN_FROZEN_CFG:  # Полная разморозка, но BN заморожены
                for layer_bb_all in backbone_model_internal.layers:
                    if isinstance(layer_bb_all, tf.keras.layers.BatchNormalization): layer_bb_all.trainable = False
        except Exception as e_unfreeze:
            print(f"ОШИБКА при разморозке backbone: {e_unfreeze}")
    elif model and not perform_fine_tuning and not CONTINUE_FROM_CHECKPOINT_CFG:
        backbone_model_internal = model.get_layer(BACKBONE_LAYER_NAME_IN_MODEL_CFG)
        if backbone_model_internal and backbone_model_internal.trainable:
            backbone_model_internal.trainable = False  # Убедимся, что он заморожен

    print("\nСтруктура модели (финальная перед компиляцией):");
    model.summary(line_length=120)
    loss_fn_compiled = lambda yt, yp: compute_detector_loss_v2_fpn(yt, yp,
                                                                   use_focal_for_obj_param=USE_FOCAL_FOR_OBJECTNESS_TRAIN,
                                                                   focal_alpha_param=FOCAL_ALPHA_TRAIN,
                                                                   focal_gamma_param=FOCAL_GAMMA_TRAIN)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                  loss=compute_detector_loss_v2_fpn)

    # 4. Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # ИСПРАВЛЕНО: training_run_description теперь определена
    log_dir_name_session = f"detector_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}"
    log_dir_abs_session = LOGS_BASE_DIR_ABS / log_dir_name_session
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir_abs_session), histogram_freq=1,
                                                          profile_batch=0)


    best_model_filename_session = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_best.keras'
    checkpoint_filepath_best_session = WEIGHTS_BASE_DIR_ABS / best_model_filename_session

    # ИСПРАВЛЕНО: validation_dataset теперь определена до этого блока
    callbacks_list = [tensorboard_callback, EpochTimeLogger()]

    if validation_dataset and val_image_paths:  # Убедимся, что есть и датасет, и пути
        enable_vis_from_cfg_value = DETECTOR_CONFIG.get('debug_callback_enable_visualization')
        detailed_logger = FullDetailEpochEndLogger(
            val_image_paths_list_cb=val_image_paths,  # Передаем списки путей
            val_xml_paths_list_cb=val_xml_paths,
            fpn_level_names_arg=DDL_FPN_LEVEL_NAMES,  # Из detector_data_loader
            fpn_configs_arg=DDL_FPN_LEVELS_CONFIG,  # Из detector_data_loader
            classes_list_arg=DDL_CLASSES_LIST,  # Из detector_data_loader
            num_classes_arg=DDL_NUM_CLASSES,  # Из detector_data_loader
            target_h_arg=DDL_TARGET_IMG_HEIGHT,  # Из detector_data_loader
            target_w_arg=DDL_TARGET_IMG_WIDTH,  # Из detector_data_loader
            log_freq_epochs=DETECTOR_CONFIG.get('debug_callback_log_freq', 1),
            num_samples_to_visualize=DETECTOR_CONFIG.get('debug_callback_num_samples', 2),
            conf_thresh_vis=CALLBACK_CONF_THRESH,  # Глобальные переменные из train_detector.py
            iou_thresh_nms_vis=CALLBACK_IOU_THRESH_NMS,
            max_dets_vis=CALLBACK_MAX_DETS,
            enable_visualization_arg=enable_vis_from_cfg_value
        )
        callbacks_list.append(detailed_logger)
        # ... (остальные коллбэки: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau - КАК БЫЛИ) ...
        model_cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_filepath_best_session),
                                                         save_weights_only=False, monitor='val_loss', mode='min',
                                                         save_best_only=True, verbose=1)
        callbacks_list.append(model_cp_cb)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_CFG,
                                                         verbose=1, restore_best_weights=True)
        callbacks_list.append(early_stop_cb)
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR_CFG,
                                                            patience=REDUCE_LR_PATIENCE_CFG, verbose=1,
                                                            min_lr=MIN_LR_ON_PLATEAU_CFG)
        callbacks_list.append(reduce_lr_cb)

    # 5. Запуск Обучения
    print(f"\n--- Запуск model.fit() ---");
    print(
        f"  Эпох: {EPOCHS_CFG}, Начальная эпоха: {initial_epoch}")  # initial_epoch определена в блоке определения режима

    try:
        history = model.fit(train_dataset, epochs=EPOCHS_CFG, validation_data=validation_dataset,
                            callbacks=callbacks_list, verbose=1,
                            initial_epoch=initial_epoch)  # Используем initial_epoch

        print(
            f"\n--- Обучение детектора (режим: {training_run_description}) завершено ---")  # training_run_description определена
        final_model_filename_session = f'{MODEL_BASE_NAME_CFG}_{training_run_description}_final_epoch{history.epoch[-1] + 1}_{timestamp}.keras'  # training_run_description определена
        final_model_save_path_session = WEIGHTS_BASE_DIR_ABS / final_model_filename_session
        model.save(final_model_save_path_session)
        print(f"Финальная модель сохранена в: {final_model_save_path_session}")
        if validation_dataset and checkpoint_filepath_best_session.exists():
            print(f"Лучшая модель по val_loss также сохранена/обновлена в: {checkpoint_filepath_best_session}")
        history_plot_filename_session = f'detector_history_{MODEL_BASE_NAME_CFG}_{training_run_description}_{timestamp}.png'  # training_run_description определена
        history_plot_save_path_session = GRAPHS_DIR_ABS / history_plot_filename_session
        plot_training_history(history, str(history_plot_save_path_session),
                              run_suffix=training_run_description)  # training_run_description определена

    except Exception as e_fit:
        print(f"ОШИБКА во время model.fit(): {e_fit}"); import traceback; traceback.print_exc()


if __name__ == '__main__':
    train_detector_main()