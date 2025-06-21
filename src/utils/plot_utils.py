# RoadDefectDetector/src/utils/plot_utils.py

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import yaml
from pathlib import Path

# --- Установка глобального сида ---
def set_global_seed(seed):
    """
    Устанавливает сид для воспроизводимости случайных процессов в random, numpy и tensorflow.
    Albumentations использует сид numpy.random.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # print(f"Глобальный сид установлен в {seed}")
    else:
        # print("Глобальный сид НЕ установлен (используются случайные сиды)")
        pass # Ничего не делаем, если сид None

# --- Цвета для отрисовки ---
# Список различных цветов для отрисовки ограничивающих рамок
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)
]

def get_color(index):
    """Циклически выбирает цвета из предопределенного списка."""
    return tuple(c/255.0 for c in COLORS[index % len(COLORS)]) # Конвертируем в float для matplotlib (ожидает цвета в диапазоне [0.0, 1.0])

# --- Основные функции отрисовки ---

def plot_image(image_np, ax, title=""):
    """Отрисовывает изображение на заданном matplotlib axis."""
    if image_np is None:
        ax.text(0.5, 0.5, "Изображение недоступно", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title)
        ax.axis('off')
        return

    # Убедимся, что изображение в формате RGB для matplotlib
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        # Если 3 канала, предполагаем RGB или BGR. Проверяем тип данных.
        # Если uint8, скорее всего BGR из OpenCV, конвертируем в RGB.
        # Если float, предполагаем, что уже в RGB (например, после нормализации).
        if image_np.dtype == np.uint8:
             # Albumentations по умолчанию возвращает RGB uint8
             display_image = image_np
        else:
             # Предполагаем вход float (например, нормализованный), трактуем как RGB
             display_image = image_np
    else:
        # Обрабатываем оттенки серого или другие форматы, если нужно
        # Для оттенков серого matplotlib ожидает 2D массив (H, W)
        if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[-1] == 1):
             display_image = np.squeeze(image_np) # Убираем последний размер, если он 1
             ax.imshow(display_image, cmap='gray') # Используем цветовую карту для оттенков серого
             ax.set_title(title)
             ax.axis('off')
             return # Специальная обработка для оттенков серого

        # Если формат не распознан
        ax.text(0.5, 0.5, f"Неподдерживаемый формат изображения: {image_np.shape}", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
        ax.set_title(title)
        ax.axis('off')
        return


    ax.imshow(display_image)
    ax.set_title(title)
    ax.axis('off')


def plot_boxes_on_image(ax, boxes, labels=None, scores=None, extra_info=None, color_index_base=None, linewidth=2, fontsize=8):
    """
    Отрисовывает ограничивающие рамки на заданном matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): Ось для отрисовки.
        boxes (list): Список ограничивающих рамок, формат [[xmin, ymin, xmax, ymax], ...].
                      Координаты должны быть в пиксельном пространстве изображения, отображаемого на ax.
        labels (list, optional): Список строк меток для каждой рамки.
        scores (list, optional): Список значений уверенности (float) для каждой рамки.
        extra_info (list, optional): Список словарей или строк с дополнительной информацией
                                     для каждой рамки, например, [{'iou': 0.75, 'level': 'P3'}, ...].
                                     Будет добавлено к строке метки.
        color_index_base (int, optional): Базовый индекс для циклического выбора цвета.
        linewidth (int): Толщина линии рамки.
        fontsize (int): Размер шрифта для текстовых меток.
    """
    if not boxes:
        return

    # Выравниваем длины списков, чтобы избежать ошибок индексации
    num_boxes = len(boxes)
    if labels is None:
        labels = [None] * num_boxes
    if scores is None:
        scores = [None] * num_boxes
    if extra_info is None:
        extra_info = [None] * num_boxes

    if not (len(labels) == len(scores) == len(extra_info) == num_boxes):
         print("ПРЕДУПРЕЖДЕНИЕ: Несоответствие длин списков boxes, labels, scores или extra_info.")
         # Обрезаем по самой короткой длине
         min_len = min(num_boxes, len(labels), len(scores), len(extra_info))
         boxes = boxes[:min_len]
         labels = labels[:min_len]
         scores = scores[:min_len]
         extra_info = extra_info[:min_len]
         num_boxes = min_len


    for i in range(num_boxes):
        box = boxes[i]
        # Убедимся, что координаты целые числа для отрисовки пикселей
        xmin, ymin, xmax, ymax = map(int, box)

        # Вычисляем ширину и высоту прямоугольника
        width = xmax - xmin
        height = ymax - ymin

        # Получаем цвет
        # Если color_index_base задан, используем его для старта цикла, иначе начинаем с 0
        box_color = get_color((color_index_base if color_index_base is not None else 0) + i)

        # Создаем прямоугольный патч
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=linewidth, edgecolor=box_color, facecolor='none')

        # Добавляем патч на ось
        ax.add_patch(rect)

        # Создаем текст метки
        text_parts = []
        if labels[i] is not None:
            text_parts.append(str(labels[i]))
        if scores[i] is not None:
            text_parts.append(f"{scores[i]:.2f}") # Форматируем скор до 2 знаков после запятой
        if extra_info[i] is not None:
             # Форматируем словарь extra_info в строку "ключ: значение, ..."
             if isinstance(extra_info[i], dict):
                 info_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in extra_info[i].items()])
             else:
                 # Если extra_info не словарь, просто преобразуем его в строку
                 info_str = str(extra_info[i])
             text_parts.append(f"({info_str})") # Добавляем доп. инфо в скобках

        label_text = " ".join(text_parts)

        # Добавляем текстовую метку
        if label_text:
            # Располагаем текст немного выше и левее верхнего левого угла рамки
            text_x = xmin
            text_y = ymin - 5 # Небольшое смещение вверх от рамки
            ax.text(text_x, text_y, label_text,
                    verticalalignment='bottom', horizontalalignment='left',
                    color='white', fontsize=fontsize, weight='bold',
                    bbox={'facecolor': box_color, 'alpha': 0.5, 'pad': 0.5},
                    clip_on=True) # Обрезаем текст, если он выходит за границы области построения

# --- Вспомогательные функции отрисовки для отладки Data Loader'а ---

def plot_original_gt(ax, image_np_original, gt_objects):
    """Отрисовывает оригинальное изображение и рамки Ground Truth."""
    plot_image(image_np_original, ax, title="Исходное изображение + GT")
    if gt_objects:
        boxes = [obj['bbox'] for obj in gt_objects]
        # Предполагаем, что gt_objects['class'] содержит строковое имя класса
        labels = [obj['class'] for obj in gt_objects]
        plot_boxes_on_image(ax, boxes, labels=labels, color_index_base=0, linewidth=2)


def plot_augmented_gt(ax, image_np_augmented, augmented_gt_objects):
    """Отрисовывает аугментированное изображение и аугментированные рамки Ground Truth."""
    # Примечание: image_np_augmented здесь ожидается как uint8 RGB из Albumentations
    plot_image(image_np_augmented, ax, title="Аугментированное изображение + GT")
    if augmented_gt_objects:
        boxes = [obj['bbox'] for obj in augmented_gt_objects]
        labels = [obj['class'] for obj in augmented_gt_objects]
        plot_boxes_on_image(ax, boxes, labels=labels, color_index_base=0, linewidth=2)

def plot_specific_anchors_on_image(ax, image_np_augmented, anchors_info_list, title="Якоря"):
    """
    Отрисовывает заданные якоря на аугментированном изображении с дополнительной информацией.

    Args:
        ax (matplotlib.axes.Axes): Ось для отрисовки.
        image_np_augmented (np.ndarray): Аугментированное изображение (RGB, uint8).
        anchors_info_list (list): Список словарей, например:
                                  {'bbox': [xmin, ymin, xmax, ymax], 'level': 'P3',
                                   'iou': 0.75, 'type': 'positive'/'ignored'/'negative', ...}.
        title (str): Заголовок для графика.
    """
    # Примечание: image_np_augmented здесь ожидается как uint8 RGB из Albumentations
    plot_image(image_np_augmented, ax, title=title)
    if anchors_info_list:
        boxes_to_plot = []
        labels_to_plot = [] # Будет тип якоря (positive, ignored, negative)
        scores_to_plot = [] # Будет IoU
        extra_info_to_plot = [] # Будет уровень FPN
        colors_indices_to_plot = [] # Индексы цветов на основе типа

        # Определение цвета на основе типа якоря для наглядности
        type_colors = {'positive': 1, 'ignored': 3, 'negative': 0, 'anchor': 5} # Зеленый, Желтый, Красный/Синий, Серый
        # Используем индекс 0 для негативных (красный) или 2 (синий). Давайте 0.

        for i, info in enumerate(anchors_info_list):
             bbox = info.get('bbox')
             anchor_type = info.get('type', 'anchor') # Default to 'anchor' type
             iou_val = info.get('iou', None) # IoU может отсутствовать для некоторых типов
             level_val = info.get('level', 'N/A')

             if bbox is None:
                 continue # Пропускаем, если нет координат

             boxes_to_plot.append(bbox)
             labels_to_plot.append(anchor_type)
             scores_to_plot.append(iou_val)
             extra_info_to_plot.append({'уровень': level_val}) # Используем 'уровень' для отображения
             colors_indices_to_plot.append(type_colors.get(anchor_type, 5)) # Выбираем индекс цвета по типу

        # Отрисовываем рамки одну за другой, чтобы явно задать цвет каждой
        for i in range(len(boxes_to_plot)):
            plot_boxes_on_image(ax, [boxes_to_plot[i]], labels=[labels_to_plot[i]], scores=[scores_to_plot[i]],
                                extra_info=[extra_info_to_plot[i]], color_index_base=colors_indices_to_plot[i], linewidth=1)

    else:
         ax.text(0.5, 0.5, "Нет якорей для отрисовки", horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')


def show_plot():
    """Отображает окно(окна) графика matplotlib."""
    plt.show()

def save_plot(fig, filepath):
    """Сохраняет график matplotlib в файл."""
    try:
        # Убедимся, что директория существует
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        print(f"График сохранен в: {filepath}")
    except Exception as e:
        print(f"Ошибка сохранения графика в {filepath}: {e}")


# --- Тестовый Блок ---
if __name__ == '__main__':
    print("--- Тестирование plot_utils.py ---")

    # Фиксируем сид для воспроизводимости *стиля* отрисовки (например, цикл цветов).
    # Это НЕ фиксирует сид для аугментации, которая происходит в другом месте.
    # Фиксация здесь просто гарантирует, что элементы графика выглядят одинаково при повторных запусках ЭТОГО скрипта.
    PLOT_TEST_SEED = 42
    set_global_seed(PLOT_TEST_SEED)
    print(f"Используется сид теста отрисовки: {PLOT_TEST_SEED}")

    # Создаем фиктивное изображение (например, серое)
    img_height, img_width = 512, 512
    dummy_image_rgb = np.full((img_height, img_width, 3), 180, dtype=np.uint8) # Светло-серый фон

    # Добавляем несколько фиктивных объектов (как в тесте аугментаций)
    dummy_image_bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR) # Конвертируем в BGR для рисования с помощью cv2
    cv2.rectangle(dummy_image_bgr, (50, 50), (150, 150), (255, 0, 0), -1) # Синий квадрат
    cv2.rectangle(dummy_image_bgr, (200, 200), (300, 350), (0, 255, 0), -1) # Зеленый прямоугольник
    cv2.circle(dummy_image_bgr, (400, 100), 50, (0, 0, 255), -1) # Красный круг
    dummy_image_rgb = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2RGB) # Конвертируем обратно в RGB

    # Создаем фиктивные данные GT
    dummy_gt_boxes = [
        [50, 50, 150, 150],
        [200, 200, 300, 350],
        [350, 50, 450, 150] # Примерно соответствует кругу
    ]
    dummy_gt_labels = ["яма", "трещина", "яма"] # Русские метки
    dummy_gt_objects = [{'bbox': b, 'class': l} for b, l in zip(dummy_gt_boxes, dummy_gt_labels)]

    # Создаем фиктивные предсказания (имитация вывода модели после декодирования и NMS)
    dummy_pred_boxes = [
        [52, 53, 148, 149], # Хорошее предсказание для ямы 1
        [205, 203, 298, 348], # Хорошее предсказание для трещины 1
        [355, 55, 445, 145], # Хорошее предсказание для ямы 2
        [140, 140, 160, 160], # Ложное срабатывание рядом с ямой 1
        [100, 100, 200, 200] # Ложное срабатывание с низкой уверенностью
    ]
    dummy_pred_labels = ["яма", "трещина", "яма", "яма", "трещина"] # Русские метки предсказаний
    dummy_pred_scores = [0.98, 0.95, 0.91, 0.6, 0.25] # Пример значений уверенности

    # Создаем фиктивные данные о якорях (имитация вывода data_loader_v3_standard в режиме отладки)
    # Пример для одного GT объекта [200, 200, 300, 350] (трещина), назначенного на P4
    # Предполагаем размер изображения 512x512, шаг P4 = 16. Ячейка, содержащая центр (250, 275), это (15, 17).
    # Размер якоря (из конфига P4, индекс 0): [0.171035, 0.120275] * 512 = [87.57, 61.58]
    # Центр якоря в ячейке (15, 17) -> центр (15.5*16, 17.5*16) = (248, 280)
    # Границы якоря: [248-87.57/2, 280-61.58/2, 248+87.57/2, 280+61.58/2] = [204, 249, 291, 311]
    # Это лишь концептуальный пример, реальная генерация якорей будет в data_loader.
    dummy_anchor_info = [
        {'bbox': [204, 249, 291, 311], 'уровень': 'P4', 'iou': 0.85, 'type': 'positive'}, # Пример якоря с высоким IoU
        {'bbox': [150, 200, 250, 300], 'уровень': 'P4', 'iou': 0.48, 'type': 'ignored'}, # Пример игнорируемого якоря
        {'bbox': [280, 280, 380, 380], 'уровень': 'P4', 'iou': 0.15, 'type': 'negative'}, # Пример отрицательного якоря
        {'bbox': [400, 400, 450, 450], 'уровень': 'P5', 'iou': 0.01, 'type': 'negative'}, # Пример отрицательного на другом уровне
        {'bbox': [50, 50, 70, 70], 'уровень': 'P3', 'iou': 0.9, 'type': 'positive'} # Пример положительного на P3
    ]


    # --- Создаем фигуру с несколькими подграфиками ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel() # Разворачиваем массив 2x3 в одномерный массив осей

    # График 1: Исходный GT
    plot_original_gt(axes[0], dummy_image_rgb, dummy_gt_objects) # Используем русские метки из dummy_gt_objects


    # График 2: Предсказания (имитация вывода оценки/инференса)
    plot_image(dummy_image_rgb, axes[1], title="Фиктивные предсказания (со скорами)")
    # Фильтруем предсказания по порогу уверенности (например, predict_conf_threshold из конфига)
    pred_threshold = 0.3
    filtered_pred_boxes = [b for i, b in enumerate(dummy_pred_boxes) if dummy_pred_scores[i] >= pred_threshold]
    filtered_pred_labels = [l for i, l in enumerate(dummy_pred_labels) if dummy_pred_scores[i] >= pred_threshold]
    filtered_pred_scores = [s for i, s in enumerate(dummy_pred_scores) if dummy_pred_scores[i] >= pred_threshold]
    # Используем color_index_base=1 для других цветов, чтобы отличать от GT
    plot_boxes_on_image(axes[1], filtered_pred_boxes, labels=filtered_pred_labels, scores=filtered_pred_scores, color_index_base=1)


    # График 3: Отрисовка якорей с IoU и Уровнем
    # Передаем dummy_anchor_info, где уже есть ключи 'уровень', 'iou', 'type'
    plot_specific_anchors_on_image(axes[2], dummy_image_rgb, dummy_anchor_info, title="Фиктивные якоря (IoU, Уровень, Тип)")


    # График 4: Другой пример якорей - только положительные
    dummy_positive_anchors = [info for info in dummy_anchor_info if info.get('type') == 'positive']
    plot_specific_anchors_on_image(axes[3], dummy_image_rgb, dummy_positive_anchors, title="Фиктивные положительные якоря")


    # График 5 и 6: Пустые графики или другие тесты
    plot_image(None, axes[4], title="Пустой график 1")
    plot_boxes_on_image(axes[4], []) # Тест отрисовки пустого списка рамок
    plot_image(dummy_image_rgb, axes[5], title="Пустой график 2")


    # --- Отображение/Сохранение графиков ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Корректируем расположение элементов, чтобы заголовки не перекрывались
    plt.suptitle("Тестовые графики plot_utils.py", fontsize=16, y=0.98) # Общий заголовок для всей фигуры

    # Убедимся, что директория 'graphs' существует в корне проекта
    _current_script_dir = Path(__file__).parent.resolve()
    # Предполагаем, что plot_utils находится в src/utils, поэтому идем на два уровня выше
    _project_root_dir = _current_script_dir.parent.parent

    graphs_dir = _project_root_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True) # Создаем директорию, если ее нет

    plot_save_path = graphs_dir / "plot_utils_test.png"
    save_plot(fig, plot_save_path) # Сохраняем график в файл

    # Вы можете выбрать: показать окно с графиком или только сохранить его
    show_plot() # Раскомментируйте, чтобы отобразить окно с графиком

    print("\n--- Тестирование plot_utils.py завершено ---")
    print("Пожалуйста, проверьте сгенерированный файл графика:", plot_save_path)