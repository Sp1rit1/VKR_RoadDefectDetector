# RoadDefectDetector/predict_detector.py (или predict_pipeline.py)
import tensorflow as tf
import numpy as np
import cv2  # OpenCV для загрузки/сохранения изображений и рисования
import yaml
import os
import argparse
import time
from pathlib import Path

# --- Добавляем src в sys.path ---
_project_root_predict = Path(__file__).parent.resolve()  # Корень проекта, где лежит этот скрипт
_src_path_predict = _project_root_predict / 'src'
import sys

if str(_src_path_predict) not in sys.path:
    sys.path.insert(0, str(_src_path_predict))

# --- Импорты из твоих модулей ---
# Для загрузки детектора может понадобиться его кастомная функция потерь, если она не стандартная Keras
# и если модель сохранялась не через model.export() / tf.saved_model.save()
# Если compute_detector_loss_v1 используется только при обучении, то для инференса она не нужна
# при загрузке модели, сохраненной через model.save() в .keras или .h5 формате, если compile=False
# Но если ты компилировал модель с этой функцией потерь и хочешь ее загрузить так же, то она нужна.
# Давай пока оставим, на случай если Keras ее потребует при tf.keras.models.load_model
try:
    from losses.detection_losses import compute_detector_loss_v1

    CUSTOM_OBJECTS_DETECTOR = {'compute_detector_loss_v1': compute_detector_loss_v1}
    print("INFO (predict_detector.py): Кастомная функция потерь для детектора загружена.")
except ImportError:
    CUSTOM_OBJECTS_DETECTOR = {}
    print("ПРЕДУПРЕЖДЕНИЕ (predict_detector.py): Кастомная функция потерь для детектора не найдена. "
          "Модель будет загружаться без нее (compile=False рекомендуется).")


# --- Загрузка ВСЕХ Конфигураций ---
def load_config_predict(config_path_obj, default_on_error=None):
    if default_on_error is None: default_on_error = {}
    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            print(f"ПРЕДУПРЕЖДЕНИЕ: {config_path_obj.name} пуст или имеет неверный формат. Используются дефолты.")
            return default_on_error
        return cfg
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {config_path_obj.name} не найден по пути: {config_path_obj}. Используются дефолты.")
        return default_on_error
    except yaml.YAMLError as e:
        print(f"ОШИБКА YAML при чтении {config_path_obj.name}: {e}. Используются дефолты.")
        return default_on_error


_base_config_path_obj = _src_path_predict / 'configs' / 'base_config.yaml'
_classifier_config_path_obj = _src_path_predict / 'configs' / 'classifier_config.yaml'
_detector_config_path_obj = _src_path_predict / 'configs' / 'detector_config.yaml'
_predict_config_path_obj = _src_path_predict / 'configs' / 'predict_config.yaml'

BASE_CONFIG_PREDICT = load_config_predict(_base_config_path_obj)
CLASSIFIER_CONFIG_PREDICT = load_config_predict(_classifier_config_path_obj,
                                                {'input_shape': [224, 224, 3], 'num_classes': 2,
                                                 'class_names_ordered': ['not_road', 'road']})
DETECTOR_CONFIG_PREDICT = load_config_predict(_detector_config_path_obj,
                                              {'input_shape': [416, 416, 3], 'classes': ['pit', 'crack'],
                                               'anchors_wh_normalized': [[0.05, 0.1], [0.1, 0.05], [0.1, 0.1]],
                                               'num_anchors_per_location': 3, 'num_classes': 2})
PREDICT_PARAMS_CONFIG = load_config_predict(_predict_config_path_obj,
                                            {'default_conf_thresh': 0.25, 'default_iou_thresh': 0.45,
                                             'default_max_dets': 100, 'classifier_model_path': '',
                                             'detector_model_path': ''})

# --- Параметры из Конфигов ---
CLS_INPUT_SHAPE = tuple(CLASSIFIER_CONFIG_PREDICT.get('input_shape'))
CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH = CLS_INPUT_SHAPE[0], CLS_INPUT_SHAPE[1]
CLS_CLASS_NAMES = CLASSIFIER_CONFIG_PREDICT.get('class_names_ordered', ['not_road', 'road'])
ROAD_CLASS_INDEX_FOR_CLASSIFIER = CLS_CLASS_NAMES.index('road') if 'road' in CLS_CLASS_NAMES else 1

DET_INPUT_SHAPE = tuple(DETECTOR_CONFIG_PREDICT.get('input_shape'))
DET_TARGET_IMG_HEIGHT, DET_TARGET_IMG_WIDTH = DET_INPUT_SHAPE[0], DET_INPUT_SHAPE[1]
DET_CLASSES_LIST = DETECTOR_CONFIG_PREDICT.get('classes', ['pit', 'crack'])
DET_ANCHORS_WH_NORM = np.array(DETECTOR_CONFIG_PREDICT.get('anchors_wh_normalized'), dtype=np.float32)
DET_NUM_ANCHORS = DETECTOR_CONFIG_PREDICT.get('num_anchors_per_location', DET_ANCHORS_WH_NORM.shape[0])
DET_NUM_CLASSES = len(DET_CLASSES_LIST)
DET_NETWORK_STRIDE = 16  # Предполагаем для нашей архитектуры
DET_GRID_HEIGHT = DET_TARGET_IMG_HEIGHT // DET_NETWORK_STRIDE
DET_GRID_WIDTH = DET_TARGET_IMG_WIDTH // DET_NETWORK_STRIDE


# --- Вспомогательные Функции ---
def preprocess_image_for_model(image_bgr, target_height, target_width):
    # ... (Код этой функции остается таким же, как ты предоставил) ...
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, [target_height, target_width])
    image_normalized = image_resized / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    return image_batch


def decode_predictions(raw_predictions_tensor, anchors_wh_normalized, grid_h, grid_w, num_classes_detector, stride):
    # ... (Код этой функции остается таким же, как ты предоставил) ...
    # raw_predictions_tensor форма: (batch, grid_h, grid_w, num_anchors, 4_coords + 1_obj + num_classes)
    batch_size = tf.shape(raw_predictions_tensor)[0]

    # Разделяем выходной тензор
    pred_xy_raw = raw_predictions_tensor[..., 0:2]  # Относительно центра ячейки
    pred_wh_raw = raw_predictions_tensor[..., 2:4]  # Логарифм отношения к якорю
    pred_obj_logit = raw_predictions_tensor[..., 4:5]  # Логит objectness
    pred_class_logits = raw_predictions_tensor[..., 5:]  # Логиты классов

    # Создаем сетку координат центров ячеек
    # gy_indices: [[0],[1]...[grid_h-1]] -> (grid_h, 1)
    # gx_indices: [[0,1...grid_w-1]]     -> (1, grid_w)
    gy_indices = tf.tile(tf.range(grid_h, dtype=tf.float32)[:, tf.newaxis], [1, grid_w])
    gx_indices = tf.tile(tf.range(grid_w, dtype=tf.float32)[tf.newaxis, :], [grid_h, 1])
    grid_coords_xy = tf.stack([gx_indices, gy_indices], axis=-1)  # (grid_h, grid_w, 2) (x,y)

    # Расширяем для батча и якорей
    grid_coords_xy = grid_coords_xy[tf.newaxis, :, :, tf.newaxis, :]  # (1, grid_h, grid_w, 1, 2)
    grid_coords_xy = tf.tile(grid_coords_xy, [batch_size, 1, 1, anchors_wh_normalized.shape[0], 1])  # (B, Gh, Gw, A, 2)

    # Декодируем координаты центра (tx, ty -> bx, by)
    # bx = sigmoid(tx) + cx
    # by = sigmoid(ty) + cy
    pred_xy_on_grid = (tf.sigmoid(pred_xy_raw) + grid_coords_xy)  # Координаты центра в масштабе сетки
    # Нормализуем координаты центра относительно всего изображения
    pred_xy_normalized = pred_xy_on_grid / tf.constant([grid_w, grid_h],
                                                       dtype=tf.float32)  # (B,Gh,Gw,A,2) -> (x_center_norm, y_center_norm)

    # Декодируем ширину и высоту (tw, th -> bw, bh)
    # bw = anchor_w * exp(tw)
    # bh = anchor_h * exp(th)
    anchors_tensor = tf.constant(anchors_wh_normalized, dtype=tf.float32)  # (Num_Anchors, 2) -> (width, height)
    anchors_reshaped = anchors_tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # (1,1,1,A,2)
    pred_wh_normalized = (tf.exp(pred_wh_raw) * anchors_reshaped)  # (B,Gh,Gw,A,2) -> (width_norm, height_norm)

    # Собираем декодированные рамки: [x_center_norm, y_center_norm, width_norm, height_norm]
    decoded_boxes_xywh_norm = tf.concat([pred_xy_normalized, pred_wh_normalized], axis=-1)  # (B,Gh,Gw,A,4)

    # Уверенность в объекте и вероятности классов
    pred_obj_confidence = tf.sigmoid(pred_obj_logit)  # (B,Gh,Gw,A,1)
    pred_class_probs = tf.sigmoid(pred_class_logits)  # (B,Gh,Gw,A,Num_Classes) (если sigmoid на выходе)
    # или tf.nn.softmax(pred_class_logits, axis=-1) если логиты без активации

    return decoded_boxes_xywh_norm, pred_obj_confidence, pred_class_probs


def apply_nms_and_filter(decoded_boxes_xywh_norm, obj_confidence, class_probs,
                         gh, gw, num_anchors, num_classes_detector,
                         confidence_threshold=0.25, iou_threshold=0.45, max_detections=100):
    # ... (Код этой функции остается таким же, как ты предоставил) ...
    batch_size = tf.shape(decoded_boxes_xywh_norm)[0]  # Должен быть 1 для инференса одного изображения

    # Решейпим все входы в плоские списки для tf.image.combined_non_max_suppression
    # (batch_size, num_total_boxes, ...)
    num_total_boxes_per_image = gh * gw * num_anchors

    # [x_center, y_center, width, height] -> [ymin, xmin, ymax, xmax] для NMS
    # ymin = yc - h/2, xmin = xc - w/2, ymax = yc + h/2, xmax = xc + w/2
    boxes_flat_xywh = tf.reshape(decoded_boxes_xywh_norm, [batch_size, num_total_boxes_per_image, 4])
    boxes_ymin_xmin_ymax_xmax = tf.concat([
        boxes_flat_xywh[..., 1:2] - boxes_flat_xywh[..., 3:4] / 2.0,  # y_center - height/2 (ymin)
        boxes_flat_xywh[..., 0:1] - boxes_flat_xywh[..., 2:3] / 2.0,  # x_center - width/2  (xmin)
        boxes_flat_xywh[..., 1:2] + boxes_flat_xywh[..., 3:4] / 2.0,  # y_center + height/2 (ymax)
        boxes_flat_xywh[..., 0:1] + boxes_flat_xywh[..., 2:3] / 2.0  # x_center + width/2  (xmax)
    ], axis=-1)
    boxes_ymin_xmin_ymax_xmax = tf.clip_by_value(boxes_ymin_xmin_ymax_xmax, 0.0,
                                                 1.0)  # Убедимся, что координаты в [0,1]

    obj_conf_flat = tf.reshape(obj_confidence, [batch_size, num_total_boxes_per_image, 1])
    class_probs_flat = tf.reshape(class_probs, [batch_size, num_total_boxes_per_image, num_classes_detector])

    # Финальные скоры для каждого класса = objectness_confidence * class_probability
    # (B, num_total_boxes, num_classes)
    final_scores_per_class = obj_conf_flat * class_probs_flat

    # combined_non_max_suppression ожидает boxes формы [batch_size, num_boxes, num_classes, 4] или [batch_size, num_boxes, 1, 4]
    # и scores формы [batch_size, num_boxes, num_classes]
    # Если у нас одна рамка на якорь, но она может принадлежать разным классам (с разными скорами)
    # то нам нужно "размножить" рамки для каждого класса или использовать немного другую логику.
    # Для tf.image.combined_non_max_suppression:
    #   boxes: A 4-D float Tensor of shape [batch_size, num_boxes, q, 4].
    #          If q is 1 then same boxes are used for all classes otherwise if q is equal to number of classes,
    #          class-specific boxes are used.
    #   scores: A 3-D float Tensor of shape [batch_size, num_boxes, num_classes]

    # Проще всего сделать q=1, то есть использовать одни и те же координаты рамки для всех классов
    boxes_for_nms = tf.expand_dims(boxes_ymin_xmin_ymax_xmax, axis=2)  # -> (B, num_total_boxes, 1, 4)

    # Применяем NMS
    nms_boxes, nms_scores, nms_classes, nms_valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes_for_nms,  # (B, num_total_boxes, 1, 4)
        scores=final_scores_per_class,  # (B, num_total_boxes, num_classes)
        max_output_size_per_class=max_detections // num_classes_detector if num_classes_detector > 0 else max_detections,
        # Макс. объектов на класс
        max_total_size=max_detections,  # Общее макс. количество объектов
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold,
        clip_boxes=False  # Координаты уже должны быть нормализованы
    )
    # nms_boxes: (B, max_total_detections, 4) -> ymin, xmin, ymax, xmax (нормализованные)
    # nms_scores: (B, max_total_detections)
    # nms_classes: (B, max_total_detections) -> ID класса
    # nms_valid_detections: (B,) -> количество валидных детекций в батче

    return nms_boxes, nms_scores, nms_classes, nms_valid_detections


def draw_detections(image_bgr_input, boxes_norm_yminxminymaxxmax, scores, classes_ids,
                    class_names_list_detector, original_img_w, original_img_h):
    # ... (Код этой функции остается таким же, как ты предоставил) ...
    # Убедись, что boxes_norm_yminxminymaxxmax действительно в формате [ymin, xmin, ymax, xmax]
    image_bgr_output = image_bgr_input.copy()
    num_valid_detections = tf.shape(boxes_norm_yminxminymaxxmax)[
        0]  # Если это уже отфильтрованные для одного изображения

    for i in range(num_valid_detections):
        if scores[i] < 0.001:  # Пропускаем очень слабые (хотя NMS уже должен был отфильтровать по score_threshold)
            continue

        ymin_n, xmin_n, ymax_n, xmax_n = boxes_norm_yminxminymaxxmax[i]

        # Преобразование в пиксельные координаты
        xmin = int(xmin_n * original_img_w)
        ymin = int(ymin_n * original_img_h)
        xmax = int(xmax_n * original_img_w)
        ymax = int(ymax_n * original_img_h)

        class_id = int(classes_ids[i])
        score_val = scores[i]

        # Определение цвета и метки
        if 0 <= class_id < len(class_names_list_detector):
            label_text = f"{class_names_list_detector[class_id]}: {score_val:.2f}"
            if class_names_list_detector[class_id] == 'pit':
                color = (0, 0, 255)  # Красный для ям
            elif class_names_list_detector[class_id] == 'crack':  # или 'treshina'
                color = (0, 255, 0)  # Зеленый для трещин
            else:
                color = (255, 0, 0)  # Синий для других (если будут)
        else:
            label_text = f"Unknown({class_id}): {score_val:.2f}"
            color = (128, 128, 128)  # Серый

        cv2.rectangle(image_bgr_output, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_bgr_output, label_text, (xmin, ymin - 10 if ymin - 10 > 10 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Увеличил толщину текста

    return image_bgr_output


# --- КОНЕЦ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

def run_complete_pipeline(image_path_arg, classifier_model_path_arg, detector_model_path_arg,
                          output_path_arg, conf_thresh_arg, iou_thresh_arg, max_dets_arg):
    # 1. Загрузка исходного изображения
    # ... (как было) ...
    if not os.path.exists(image_path_arg): print(f"Ошибка: Изображение не найдено: {image_path_arg}"); return
    original_bgr_image = cv2.imread(image_path_arg)
    if original_bgr_image is None: print(f"Ошибка: Не удалось прочитать: {image_path_arg}"); return
    original_h, original_w = original_bgr_image.shape[:2]

    # 2. Загрузка моделей
    # ... (как было, но добавим compile=False для инференса) ...
    classifier_model_full_path = os.path.join(_project_root_predict, classifier_model_path_arg)
    detector_model_full_path = os.path.join(_project_root_predict, detector_model_path_arg)
    try:
        print(f"Загрузка классификатора: {classifier_model_full_path}")
        classifier_model = tf.keras.models.load_model(classifier_model_full_path, compile=False)
        print(f"Загрузка детектора: {detector_model_full_path}")
        # Если модель сохранена без compile=False и требует кастомный лосс, он нужен.
        # Если сохранена с compile=False или как SavedModel, то custom_objects не обязателен для инференса.
        detector_model = tf.keras.models.load_model(detector_model_full_path, custom_objects=CUSTOM_OBJECTS_DETECTOR,
                                                    compile=False)
        print("Модели успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}"); return

    # 3. Этап Классификации
    # ... (как было) ...
    print("\n--- Этап 1: Классификация 'Дорога / Не дорога' ---")
    classifier_input_batch = preprocess_image_for_model(original_bgr_image, CLS_TARGET_IMG_HEIGHT, CLS_TARGET_IMG_WIDTH)
    classifier_prediction = classifier_model.predict(classifier_input_batch)
    predicted_class_name_cls = "Unknown";
    confidence_cls = 0.0
    if CLASSIFIER_CONFIG_PREDICT.get('num_classes', 2) == 2 and classifier_prediction.shape[-1] == 1:
        confidence_road = classifier_prediction[0][0]
        predicted_class_name_cls = "road" if confidence_road > 0.5 else "not_road"
        confidence_cls = confidence_road if confidence_road > 0.5 else 1.0 - confidence_road
        print(
            f"Предсказание классификатора: '{predicted_class_name_cls}' с уверенностью {confidence_cls:.4f} (raw 'road' score: {confidence_road:.4f})")
    # ... (ветка для >2 классов, если нужна) ...

    # 4. Принятие решения и Детекция
    output_image_display = original_bgr_image.copy()
    if predicted_class_name_cls == "not_road":
        print("\nРЕЗУЛЬТАТ: Дрон сбился с пути! (Обнаружена не дорога)")
        cv2.putText(output_image_display, "DRONE OFF COURSE (NOT A ROAD)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    elif predicted_class_name_cls == "road":
        print("\n--- Этап 2: Детекция дефектов на дороге ---")
        detector_input_batch = preprocess_image_for_model(original_bgr_image, DET_TARGET_IMG_HEIGHT,
                                                          DET_TARGET_IMG_WIDTH)

        print("  Предсказание детектора...")
        start_time_det = time.time()
        raw_detector_predictions = detector_model.predict(detector_input_batch)  # Форма (1, Gh, Gw, A, 5+C)
        end_time_det = time.time()
        print(f"  Время инференса детектора: {end_time_det - start_time_det:.4f} секунд")
        print(f"  Форма сырых предсказаний детектора: {raw_detector_predictions.shape}")

        print("  Декодирование предсказаний...")
        decoded_boxes_xywh, obj_conf, class_probs = decode_predictions(
            raw_detector_predictions, DET_ANCHORS_WH_NORM, DET_GRID_HEIGHT, DET_GRID_WIDTH,
            DET_NUM_CLASSES, DET_NETWORK_STRIDE
        )

        # --- ОТЛАДОЧНЫЙ ВЫВОД СЫРЫХ ДЕТЕКЦИЙ ДО NMS ---
        print(f"\n  --- ДЕТАЛЬНЫЕ ПРЕДСКАЗАНИЯ ДО NMS (objectness > 0.01) ---")
        obj_conf_squeezed_np = tf.squeeze(obj_conf[0], axis=-1).numpy()  # (Gh, Gw, A)
        potential_indices = np.argwhere(obj_conf_squeezed_np > 0.01)
        print(f"  Найдено {len(potential_indices)} потенциальных якорей с objectness > 0.01")

        # Выведем топ-N по objectness score для примера
        top_k_to_show = 5
        if len(potential_indices) > 0:
            # Собираем данные для сортировки
            temp_list_for_sort = []
            for idx_gh, idx_gw, idx_anchor in potential_indices:
                obj_s = obj_conf_squeezed_np[idx_gh, idx_gw, idx_anchor]
                box_s = decoded_boxes_xywh[0, idx_gh, idx_gw, idx_anchor, :].numpy()
                cls_s = class_probs[0, idx_gh, idx_gw, idx_anchor, :].numpy()
                temp_list_for_sort.append(
                    {'obj': obj_s, 'box': box_s, 'cls': cls_s, 'cell': (idx_gh, idx_gw), 'anchor': idx_anchor})

            # Сортируем по убыванию objectness
            sorted_detections = sorted(temp_list_for_sort, key=lambda x: x['obj'], reverse=True)

            for k_idx, det_info in enumerate(sorted_detections[:top_k_to_show]):
                pred_cls_id_raw = np.argmax(det_info['cls'])
                pred_cls_name_raw = DET_CLASSES_LIST[pred_cls_id_raw] if pred_cls_id_raw < len(
                    DET_CLASSES_LIST) else "Unknown"
                print(f"    Топ {k_idx + 1}: Ячейка{det_info['cell']}, Якорь {det_info['anchor']}: "
                      f"ObjConf={det_info['obj']:.3f}, "
                      f"Box_XYWH_n={np.round(det_info['box'], 2)}, "
                      f"MaxCls='{pred_cls_name_raw}'(Score={det_info['cls'][pred_cls_id_raw]:.3f})")
        # --- КОНЕЦ ОТЛАДОЧНОГО ВЫВОДА ---

        print(f"\n  Применение NMS с conf_thresh={conf_thresh_arg:.2f}, iou_thresh={iou_thresh_arg:.2f}...")
        final_boxes_norm, final_scores, final_classes_ids, num_valid_dets = apply_nms_and_filter(
            decoded_boxes_xywh, obj_conf, class_probs,
            DET_GRID_HEIGHT, DET_GRID_WIDTH, DET_NUM_ANCHORS, DET_NUM_CLASSES,
            confidence_threshold=conf_thresh_arg,
            iou_threshold=iou_thresh_arg,
            max_detections=max_dets_arg
        )

        num_found_defects = int(num_valid_dets[0].numpy())  # num_valid_dets это тензор [B], берем первый элемент
        print(f"  Найдено {num_found_defects} дефектов после NMS.")

        if num_found_defects > 0:
            # Извлекаем только валидные детекции для первого (и единственного) изображения в батче
            boxes_to_draw_norm = final_boxes_norm[0][:num_found_defects].numpy()
            scores_to_draw = final_scores[0][:num_found_defects].numpy()
            classes_ids_to_draw = final_classes_ids[0][:num_found_defects].numpy()

            output_image_display = draw_detections(
                original_bgr_image, boxes_to_draw_norm, scores_to_draw, classes_ids_to_draw,
                DET_CLASSES_LIST, original_w, original_h
            )
            print("РЕЗУЛЬТАТ: Обнаружены дефекты. См. изображение.")
        else:
            print("РЕЗУЛЬТАТ: Дорога в норме (дефекты не обнаружены детектором с текущими порогами).")
            cv2.putText(output_image_display, "ROAD OK - NO DEFECTS DETECTED", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # ... (остальная логика сохранения и if __name__ == '__main__' как была) ...
    else:  # Случай, если классификатор выдал что-то кроме "road" или "not_road" (маловероятно с текущей моделью)
        print("РЕЗУЛЬТАТ: Не удалось определить тип поверхности классификатором (не 'road' и не 'not_road').")
        cv2.putText(output_image_display, "SURFACE UNCLASSIFIED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

    # Сохранение результата
    final_output_path = ""
    if output_path_arg:
        final_output_path = output_path_arg
        if not os.path.isabs(final_output_path):
            final_output_path = str(_project_root_predict / final_output_path)
    elif PREDICT_PARAMS_CONFIG.get("output_path_template"):
        template = PREDICT_PARAMS_CONFIG["output_path_template"]
        img_path_obj = Path(image_path_arg)
        image_name = img_path_obj.stem;
        ext = img_path_obj.suffix[1:]
        output_filename = template.format(image_name=image_name, ext=ext)
        output_dir = _project_root_predict / Path(output_filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = str(_project_root_predict / output_filename)
    else:
        base, ext_in = os.path.splitext(image_path_arg)
        final_output_path = base + "_pipeline_result" + ext_in

    try:
        cv2.imwrite(final_output_path, output_image_display)
        print(f"\nИтоговое изображение сохранено в: {final_output_path}")
    except Exception as e_write:
        print(f"ОШИБКА: Не удалось сохранить итоговое изображение в {final_output_path}: {e_write}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный пайплайн: классификация Дорога/Не дорога + Детекция дефектов.")
    parser.add_argument("--image_path", type=str, required=True, help="Путь к входному изображению.")
    parser.add_argument("--classifier_model_path", type=str, default=PREDICT_PARAMS_CONFIG.get("classifier_model_path"),
                        help="Путь к модели классификатора.")
    parser.add_argument("--detector_model_path", type=str, default=PREDICT_PARAMS_CONFIG.get("detector_model_path"),
                        help="Путь к модели детектора.")
    parser.add_argument("--output_path", type=str, default=None, help="Путь для сохранения результата.")
    parser.add_argument("--conf_thresh", type=float, default=PREDICT_PARAMS_CONFIG.get("default_conf_thresh"),
                        help="Порог уверенности для NMS.")
    parser.add_argument("--iou_thresh", type=float, default=PREDICT_PARAMS_CONFIG.get("default_iou_thresh"),
                        help="Порог IoU для NMS.")
    parser.add_argument("--max_dets", type=int, default=PREDICT_PARAMS_CONFIG.get("default_max_dets"),
                        help="Макс. детекций после NMS.")
    args_pipeline = parser.parse_args()

    if not args_pipeline.classifier_model_path: print("ОШИБКА: Путь к модели классификатора не указан."); exit()
    if not args_pipeline.detector_model_path: print("ОШИБКА: Путь к модели детектора не указан."); exit()

    run_complete_pipeline(
        args_pipeline.image_path,
        args_pipeline.classifier_model_path,
        args_pipeline.detector_model_path,
        args_pipeline.output_path,
        args_pipeline.conf_thresh,
        args_pipeline.iou_thresh,
        args_pipeline.max_dets
    )