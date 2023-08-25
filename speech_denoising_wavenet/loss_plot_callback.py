import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, fname):
        super().__init__()
        self.loss = []
        self.val_loss = []
        self.epochs = []
        self.fname = fname

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.epochs.append(epoch)

        fig, axs = plt.subplots(2, figsize=(8, 8))

        fig.tight_layout()
        axs[0].plot(self.epochs, self.loss, color="y")
        axs[0].set_title("Loss")
        axs[1].plot(self.epochs, self.val_loss, color="b")
        axs[1].set_title("Val Loss")

        for ax in axs.flat:
            ax.set(xlabel="Epoch", ylabel="Loss")
            ax.grid(visible=True)
            ax.xaxis.set_major_locator(MultipleLocator(1))

        fig.tight_layout()

        fig.savefig(fname=self.fname, dpi=400)
