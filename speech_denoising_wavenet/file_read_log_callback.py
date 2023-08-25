import tensorflow as tf


class FileReadLogCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        with open("./file_read_log.txt", "a") as frlf:
            frlf.write(
                "===============================\nEPOCH END\n==============================="
            )
