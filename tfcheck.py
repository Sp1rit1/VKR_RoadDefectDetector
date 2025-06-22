# check_gpu_with_detailed_logging.py

import os
import sys
import logging
from tensorflow.python.client import  device_lib
from pathlib import Path

# --- Настройка логирования скрипта ---
# Используем стандартный модуль logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Скрипт запущен. Начинаем проверку наличия GPU с подробным логированием TensorFlow.")

# --- Включение подробного логирования TensorFlow ---
# Установка переменной окружения TF_CPP_MIN_LOG_LEVEL в '0' включает все логи (INFO, WARNING, ERROR, FATAL).
# Это должно быть сделано ДО импорта TensorFlow.
# Возможные значения:
# '0': Выводить все (INFO, WARNING, ERROR, FATAL) - самый подробный режим
# '1': Выводить WARNING, ERROR, FATAL (подавить INFO)
# '2': Выводить ERROR, FATAL (подавить INFO и WARNING)
# '3': Выводить FATAL (подавить все, кроме FATAL)
logger.info("Установка переменной окружения TF_CPP_MIN_LOG_LEVEL = '0' для подробных логов TensorFlow.")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# --- Импорт TensorFlow ---
try:
    logger.info("Попытка импорта TensorFlow...")
    import tensorflow as tf
    logger.info(f"TensorFlow успешно импортирован (версия: {tf.__version__}).")
except ImportError as e:
    logger.error(f"Ошибка импорта TensorFlow: {e}")
    logger.error("Убедитесь, что TensorFlow установлен в вашем виртуальном окружении.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Неизвестная ошибка при импорте TensorFlow: {e}")
    sys.exit(1)


# --- Проверка наличия GPU ---
logger.info("Попытка получить список доступных физических GPU устройств...")
try:
    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        logger.info("!!! GPU УСПЕШНО ОБНАРУЖЕНЫ TensorFlow !!!")
        print("\n" + "="*80)
        print("СПИСОК ДОСТУПНЫХ ФИЗИЧЕСКИХ GPU УСТРОЙСТВ:".center(80))
        print("="*80)
        for gpu in gpu_devices:
            print(f"  - {gpu}")
        print("="*80 + "\n")
        logger.info(f"Обнаружено {len(gpu_devices)} GPU устройств.")
        logger.info("Теперь можно запускать обучение на GPU.")
    else:
        logger.warning("--- GPU НЕ ОБНАРУЖЕНЫ TensorFlow ---")
        print("\n" + "="*80)
        print("!!! GPU НЕ ОБНАРУЖЕНЫ TensorFlow !!!".center(80))
        print("="*80)
        print("Список доступных физических GPU устройств пуст.")
        print("Это означает, что TensorFlow не может найти или использовать GPU.")
        print("Пожалуйста, внимательно изучите логи TensorFlow выше для выявления причин.")
        print("Обычно проблемы связаны с:")
        print(" - Неправильной установкой CUDA Toolkit или cuDNN SDK.")
        print(" - Несовместимостью версий TensorFlow, CUDA, cuDNN и драйвера NVIDIA.")
        print(" - Ошибками в переменных окружения (PATH).")
        print(" - Конфликтами между разными установками CUDA/cuDNN.")
        print(" - Проблемами с самой видеокартой или ее драйвером.")
        print("="*80 + "\n")
        logger.warning("GPU не обнаружены.")

except Exception as e:
    logger.error(f"Произошла ошибка при попытке получить список GPU: {e}")
    logger.error("Пожалуйста, изучите логи TensorFlow выше для выявления причин.")
    sys.exit(1)


logger.info("Скрипт завершил выполнение.")