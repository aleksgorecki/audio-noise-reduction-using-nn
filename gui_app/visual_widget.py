from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
    QTabBar,
    QTabWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from datasets import visual
from matplotlib import pyplot as plt
from matplotlib import rcParams


class VisualWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        rcParams.update({"figure.autolayout": True})

        figure_size = (300, 200)

        self.time_noisy_fig, self.time_noisy_ax = plt.subplots(figsize=figure_size)
        self.time_pred_fig, self.time_pred_ax = plt.subplots(figsize=figure_size)
        self.time_clean_fig, self.time_clean_ax = plt.subplots(figsize=figure_size)
        self.spec_noisy_fig, self.spec_noisy_ax = plt.subplots(figsize=figure_size)
        self.spec_pred_fig, self.spec_pred_ax = plt.subplots(figsize=figure_size)
        self.spec_clean_fig, self.spec_clean_ax = plt.subplots(figsize=figure_size)

        # self.time_noisy_fig.tight_layout()
        # self.spec_noisy_fig.tight_layout()
        # self.time_pred_fig.tight_layout()
        # self.spec_pred_fig.tight_layout()
        # self.time_clean_fig.tight_layout()
        # self.spec_clean_fig.tight_layout()

        # self.time_noisy_fig.constrained_layout()
        # self.spec_noisy_fig.constrained_layout()
        # self.time_pred_fig.constrained_layout()
        # self.spec_pred_fig.constrained_layout()
        # self.time_clean_fig.constrained_layout()
        # self.spec_clean_fig.constrained_layout()

        self.time_noisy = FigureCanvasQTAgg(self.time_noisy_fig)
        self.time_pred = FigureCanvasQTAgg(self.time_pred_fig)
        self.time_clean = FigureCanvasQTAgg(self.time_clean_fig)
        self.spec_noisy = FigureCanvasQTAgg(self.spec_noisy_fig)
        self.spec_pred = FigureCanvasQTAgg(self.spec_pred_fig)
        self.spec_clean = FigureCanvasQTAgg(self.spec_clean_fig)

        self.time_tab = QWidget()
        self.spec_tab = QWidget()
        self.metrics_group = QGroupBox("Metrics")
        self.tabs = QTabWidget()

        self.time_tab.setLayout(QVBoxLayout())
        self.spec_tab.setLayout(QVBoxLayout())

        self.time_tab.layout().addWidget(self.time_noisy)
        self.time_tab.layout().addWidget(self.time_pred)
        self.time_tab.layout().addWidget(self.time_clean)

        self.spec_tab.layout().addWidget(self.spec_noisy)
        self.spec_tab.layout().addWidget(self.spec_pred)
        self.spec_tab.layout().addWidget(self.spec_clean)

        self.tabs.addTab(self.time_tab, "Time domain")
        self.tabs.addTab(self.spec_tab, "Frequency domain")

        self.layout().addWidget(self.tabs)
        self.layout().addWidget(self.metrics_group)

        self.clear_all()

    def plot_noisy(self, data):
        self.time_noisy_fig.clear()
        self.spec_noisy_fig.clear()

        time_ax = self.time_noisy_fig.add_subplot(111)
        spec_ax = self.spec_noisy_fig.add_subplot(111)

        visual.plot_waveform(
            data,
            fs=16000,
            show=False,
            fig=self.time_noisy_fig,
            ax=time_ax,
            eng=True,
            title="Noisy",
        )
        visual.plot_spectrogram(
            data,
            fs=16000,
            show=False,
            fig=self.spec_noisy_fig,
            ax=spec_ax,
            eng=True,
            title="Noisy",
        )

        self.time_noisy.draw()
        self.spec_noisy.draw()

    def plot_predicted(self, data):
        self.time_pred_fig.clear()
        self.spec_pred_fig.clear()

        time_ax = self.time_pred_fig.add_subplot(111)
        spec_ax = self.spec_pred_fig.add_subplot(111)

        visual.plot_waveform(
            data,
            fs=16000,
            show=False,
            fig=self.time_pred_fig,
            ax=time_ax,
            eng=True,
            title="Predicted",
        )
        visual.plot_spectrogram(
            data,
            fs=16000,
            show=False,
            fig=self.spec_pred_fig,
            ax=spec_ax,
            eng=True,
            title="Predicted",
        )

        self.time_pred.draw()
        self.spec_pred.draw()

    def plot_clean(self, data):
        self.time_clean_fig.clear()
        self.spec_clean_fig.clear()

        time_ax = self.time_clean_fig.add_subplot(111)
        spec_ax = self.spec_clean_fig.add_subplot(111)

        visual.plot_waveform(
            data,
            fs=16000,
            show=False,
            fig=self.time_clean_fig,
            ax=time_ax,
            eng=True,
            title="Clean",
        )
        visual.plot_spectrogram(
            data,
            fs=16000,
            show=False,
            fig=self.spec_clean_fig,
            ax=spec_ax,
            eng=True,
            title="Clean",
        )

        self.time_clean.draw()
        self.spec_clean.draw()

    def clear_results(self):
        self.time_pred_fig.clear()
        self.spec_pred_fig.clear()
        self.time_clean_fig.clear()
        self.spec_clean_fig.clear()
        self.time_pred.draw()
        self.spec_pred.draw()
        self.time_clean.draw()
        self.spec_clean.draw()

    def clear_all(self):
        self.time_noisy_fig.clear()
        self.spec_noisy_fig.clear()
        self.time_pred_fig.clear()
        self.spec_pred_fig.clear()
        self.time_clean_fig.clear()
        self.spec_clean_fig.clear()
        self.time_noisy.draw()
        self.spec_noisy.draw()
        self.time_pred.draw()
        self.spec_pred.draw()
        self.time_clean.draw()
        self.spec_clean.draw()
