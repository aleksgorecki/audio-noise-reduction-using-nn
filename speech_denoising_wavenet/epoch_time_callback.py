from timeit import default_timer as timer
import tensorflow as tf
import pandas as pd
import os


class EpochTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, dir):
        super().__init__()
        self.starttime = None
        self.dir = dir
        self.df = pd.DataFrame(columns=["times"])

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        record = pd.DataFrame(data=[{"times": timer() - self.starttime}])
        self.df = pd.concat([self.df, record])
        self.df.to_csv(os.path.join(self.dir, f"epoch_times.csv"), index=False)

