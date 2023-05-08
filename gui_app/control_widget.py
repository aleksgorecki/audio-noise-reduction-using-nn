import os.path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel
from PyQt5.QtCore import pyqtSlot
import pathlib


class ControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.run_button = QPushButton("Run prediction", parent=self)
        self.noisy_input_button = QPushButton("Select noisy input", parent=self)
        self.clean_input_button = QPushButton("Select clean input", parent=self)
        self.model_config_button = QPushButton("Select model config", parent=self)
        self.model_weights_button = QPushButton("Select model weights", parent=self)

        self.noisy_input_dialog = QFileDialog(caption="Noisy input", parent=self)
        self.noisy_input_dialog.setFileMode(QFileDialog.ExistingFile)
        self.clean_input_dialog = QFileDialog(caption="Noisy input", parent=self)
        self.clean_input_dialog.setFileMode(QFileDialog.ExistingFile)
        # self.model_config_dialog = QFileDialog("Model config")
        # self.model_weights_dialog = QFileDialog("Model weights")

        self.noisy_input_button.clicked.connect(self.noisy_input_onclick)
        self.clean_input_button.clicked.connect(self.clean_input_onclick)

        self.debug_label = QLabel("default", parent=self)

        self.layout().addWidget(self.debug_label)
        self.layout().addWidget(self.run_button)
        self.layout().addWidget(self.noisy_input_button)
        self.layout().addWidget(self.clean_input_button)
        self.layout().addWidget(self.model_config_button)
        self.layout().addWidget(self.model_weights_button)

        self.noisy_input_path = ""
        self.clean_input_path = ""
        self.model_config_path = ""
        self.model_weights_path = ""

    @pyqtSlot()
    def noisy_input_onclick(self):
        options = QFileDialog.Options()
        last_dir = ""
        if self.noisy_input_path is not None:
            last_dir = os.path.dirname(self.noisy_input_path)
        if last_dir != "":
            self.noisy_input_path = \
            QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a file to denoise"), options=options,
                                        filter=self.tr("Wav File (*.wav)"))[0]
        else:
            self.noisy_input_path = QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a file to denoise"), directory=last_dir, options=options, filter=self.tr("Wav File (*.wav)"))[0]
        self.debug_label.setText(str(self.noisy_input_path))

    @pyqtSlot()
    def clean_input_onclick(self):
        options = QFileDialog.Options()
        last_dir = ""
        if self.clean_input_path is not None:
            last_dir = os.path.dirname(self.clean_input_path)
        if last_dir != "":
            self.clean_input_path = \
            QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a clean reference file"), directory=last_dir, options=options,
                                        filter=self.tr("Wav File (*.wav)"))[0]
        else:
            self.clean_input_path = QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a clean reference file"), options=options, filter=self.tr("Wav File (*.wav)"))[0]
        self.debug_label.setText(str(self.noisy_input_path))

    @pyqtSlot()
    def model_config_onclick(self):
        pass

    @pyqtSlot()
    def model_weights_onclick(self):
        pass

