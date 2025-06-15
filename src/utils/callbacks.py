# src/utils/callbacks.py
import tensorflow as tf
import time

class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{self.params['epochs']} - Начало", end='') # Убрал \n для вывода на той же строке

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        # Логи Keras (loss, val_loss, etc.) уже печатаются после этого сообщения по умолчанию,
        # поэтому просто добавим время.
        # Если verbose=0 в model.fit(), то нужно будет печатать логи здесь.
        # Мы используем verbose=1, так что Keras сам напечатает метрики.
        # Просто добавим наше время в конец строки, которую начал Keras.
        # Keras progress bar (если verbose=1) обычно заканчивается на ..., поэтому это может выглядеть немного криво.
        # Более чистый способ - установить verbose=2 в model.fit() и печатать все метрики здесь.
        # Но для начала попробуем так:
        print(f" - Время эпохи: {epoch_duration:.2f} сек") # Печатаем на новой строке после стандартного вывода Keras

# Если хочешь более контролируемый вывод (когда verbose=2 в model.fit):
# class EpochTimeLoggerPretty(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.epoch_start_time = time.time()
#         print(f"Epoch {epoch+1}/{self.params['epochs']}")

#     def on_epoch_end(self, epoch, logs=None):
#         epoch_duration = time.time() - self.epoch_start_time
#         log_string = ""
#         if logs:
#             for k, v in logs.items():
#                 log_string += f" - {k}: {v:.4f}"
#         print(f"Время эпохи: {epoch_duration:.2f} сек{log_string}")