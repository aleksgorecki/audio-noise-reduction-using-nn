import json
import os.path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel, QCheckBox, QRadioButton, QGroupBox, QHBoxLayout, QMessageBox
from PyQt5.QtCore import pyqtSlot
import pathlib
from speech_denoising_wavenet.models import DenoisingWavenet
from custom_model_evaluation import predict_example
import librosa
from visual_widget import VisualWidget


class ControlWidget(QWidget):
    def __init__(self, visual_widget: VisualWidget):
        super().__init__()
        self.setFixedWidth(300)
        self.setFixedHeight(700)
        self.setLayout(QVBoxLayout())

        self.run_button = QPushButton("Run prediction", parent=self)
        self.load_model_button = QPushButton("Load model", parent=self)

        # self.clean_input_button = QPushButton("Select clean input", parent=self)
        # self.model_config_button = QPushButton("Select model config", parent=self)
        # self.model_weights_button = QPushButton("Select model weights", parent=self)
        self.noisy_input_group = QGroupBox("Noisy input", parent=self)
        self.model_session_group = QGroupBox("Model session", parent=self)

        self.noisy_input_group.setLayout(QHBoxLayout())
        self.model_session_group.setLayout(QHBoxLayout())

        self.noisy_input_button = QPushButton("Select", parent=self.noisy_input_group)
        self.noisy_input_label = QLabel("", parent=self.noisy_input_group)
        self.noisy_input_group.layout().addWidget(self.noisy_input_label)
        self.noisy_input_group.layout().addWidget(self.noisy_input_button)

        self.model_session_label = QLabel("", parent=self.model_session_group)
        self.model_session_button = QPushButton("Select", parent=self.model_session_group)
        self.model_session_group.layout().addWidget(self.model_session_label)
        self.model_session_group.layout().addWidget(self.model_session_button)

        self.weights_group = QGroupBox("Weights selection", parent=self)

        self.best_weights_radio = QRadioButton("Use best weights", parent=self.weights_group)
        self.latest_weights_radio = QRadioButton("Use latest weights", parent=self.weights_group)
        self.custom_weights_radio = QRadioButton("Use selected weights", parent=self.weights_group)
        self.custom_weights_button = QPushButton("Select", parent=self.weights_group)
        self.weights_label = QLabel("")

        self.weights_group.setLayout(QVBoxLayout())
        self.weights_group.layout().addWidget(self.weights_label)
        self.weights_group.layout().addWidget(self.best_weights_radio)
        self.weights_group.layout().addWidget(self.latest_weights_radio)
        self.weights_group.layout().addWidget(self.custom_weights_radio)

        self.weights_group.layout().addWidget(self.custom_weights_button)

        self.general_groupbox = QGroupBox("General settings", parent=self)
        self.general_groupbox.setLayout(QVBoxLayout())
        self.metrics_checkbox = QCheckBox("Calculate metrics", parent=self.general_groupbox)
        self.save_result_checkbox = QCheckBox("Save result", parent=self.general_groupbox)
        self.show_extra_checkbox = QCheckBox("Show extra options", parent=self.general_groupbox)
        self.general_groupbox.layout().addWidget(self.metrics_checkbox)
        self.general_groupbox.layout().addWidget(self.show_extra_checkbox)
        self.general_groupbox.layout().addWidget(self.save_result_checkbox)

        self.noisy_input_path = ""
        self.last_noisy_dir = ""
        self.session_path = ""
        self.last_session_dir = ""
        self.custom_weights_path = ""
        self.calculate_metrics = False
        self.save_output = False
        self.noisy_input_data = None

        ControlWidget.set_label_empty(self.model_session_label)
        ControlWidget.set_label_empty(self.noisy_input_label)
        ControlWidget.set_label_empty(self.weights_label)

        self.noisy_input_button.clicked.connect(self.noisy_input_onclick)
        self.model_session_button.clicked.connect(self.session_onclick)

        self.best_weights_radio.clicked.connect(self.best_weights_toggle)
        self.latest_weights_radio.clicked.connect(self.latest_weights_toggle)
        self.custom_weights_radio.clicked.connect(self.custom_weights_toggle)
        self.custom_weights_button.clicked.connect(self.select_weights_onclick)
        self.run_button.clicked.connect(self.run_prediction_onclick)
        self.load_model_button.clicked.connect(self.load_model_onclick)

        self.layout().addWidget(self.run_button)
        self.layout().addWidget(self.noisy_input_group)
        self.layout().addWidget(self.load_model_button)
        self.layout().addWidget(self.model_session_group)
        self.layout().addWidget(self.weights_group)
        self.layout().addWidget(self.general_groupbox)

        self.weights_group.setEnabled(False)
        self.custom_weights_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.load_model_button.setEnabled(False)

        self.model = None
        self.visual_widget = visual_widget

    def set_default_settings(self):
        pass

    @pyqtSlot()
    def noisy_input_onclick(self):
        if self.last_noisy_dir != "":
            directory = self.last_noisy_dir
        else:
            directory = os.getcwd()
        options = QFileDialog.Options()
        noisy_input = \
            QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a file to denoise"), options=options,
                                        filter=self.tr("Wav File (*.wav)"), directory=directory)[0]
        if noisy_input != "":
            self.noisy_input_path = noisy_input
            self.last_noisy_dir = os.path.dirname(noisy_input)
            self.noisy_input_data = librosa.core.load(self.noisy_input_path, sr=16000, mono=True)
            ControlWidget.set_label_text_enable(self.noisy_input_label, str(pathlib.Path(self.noisy_input_path).name))
            self.visual_widget.plot_noisy(self.noisy_input_data[0])
            self.visual_widget.clear_results()
            if self.model is not None:
                self.run_button.setEnabled(True)

    @pyqtSlot()
    def session_onclick(self):
        directory = os.path.join("../speech_denoising_wavenet/sessions")
        session_dir = QFileDialog.getExistingDirectory(self, 'Select session directory', directory=directory)
        if session_dir != "":
            dir_contents = os.listdir(session_dir)
            if "checkpoints" not in dir_contents or "config.json" not in dir_contents:
                dialog = QMessageBox()
                dialog.setWindowTitle("Error")
                dialog.setText("Selected directory is not a valid model session")
                dialog.exec()
                return
            self.session_path = session_dir
            self.last_session_dir = os.path.dirname(session_dir)
            ControlWidget.set_label_text_enable(self.model_session_label, str(pathlib.Path(self.session_path).name))
            self.weights_group.setEnabled(True)
            self.custom_weights_path = ""
            self.model = None
            self.run_button.setEnabled(False)
            self.visual_widget.clear_results()

    @pyqtSlot()
    def select_weights_onclick(self):
        options = QFileDialog.Options()
        weights = \
            QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select weights file"), directory=os.path.join(self.session_path, "checkpoints"), options=options,
                                        filter=self.tr("HDF5 File (*.hdf5)"))[0]
        if weights != "":
            self.custom_weights_path = weights
            ControlWidget.set_label_text_enable(self.weights_label, str(pathlib.Path(self.custom_weights_path).name))
            self.load_model_button.setEnabled(True)

    def best_weights_toggle(self):
        if self.best_weights_radio.isChecked():
            self.custom_weights_button.setEnabled(False)
            self.load_model_button.setEnabled(True)

    def latest_weights_toggle(self):
        if self.latest_weights_radio.isChecked():
            self.custom_weights_button.setEnabled(False)
            self.load_model_button.setEnabled(True)

    def custom_weights_toggle(self):
        if self.custom_weights_radio.isChecked():
            self.custom_weights_button.setEnabled(True)
            if self.custom_weights_path == "":
                self.load_model_button.setEnabled(False)

    def load_model_onclick(self):
        try:
            with open(os.path.join(self.session_path, "config.json"), "r") as cf:
                config = json.load(cf)
                config["training"]["path"] = self.session_path
                checkpoint = None
                if self.best_weights_radio.isChecked():
                    checkpoint = ControlWidget.get_best_checkpoint(os.path.join(self.session_path, "checkpoints"))
                elif self.latest_weights_radio.isChecked():
                    checkpoint = ControlWidget.get_latest_checkpoint(os.path.join(self.session_path, "checkpoints"))
                elif self.custom_weights_radio.isChecked() and self.custom_weights_path != "":
                    checkpoint = self.custom_weights_path
                else:
                    raise RuntimeError
                self.model = DenoisingWavenet(config, load_checkpoint=checkpoint)
                if self.noisy_input_data is not None:
                    self.run_button.setEnabled(True)
        except RuntimeError:
            dialog = QMessageBox()
            dialog.setWindowTitle("Error")
            dialog.setText("Something went wrong while loading the model")
            dialog.exec()

    def run_prediction_onclick(self):
        noisy_path = pathlib.Path(self.noisy_input_path)
        clean_name = noisy_path.name
        clean_parent = str(noisy_path.parents[1])
        set_name = str(noisy_path.parents[0].name).replace("noisy", "clean")
        clean_path = os.path.join(clean_parent, set_name, clean_name)

        if not os.path.exists(clean_path):
            dialog = QMessageBox()
            dialog.setWindowTitle("Error")
            dialog.setText("Clean example doesnt exist")
            dialog.exec()
            return

        clean_data = librosa.core.load(clean_path, sr=16000, mono=True)

        predicted, metrics, clean_aligned = predict_example(self.noisy_input_data[0], clean_data[0], self.model, self.metrics_checkbox.isChecked())

        self.visual_widget.plot_predicted(predicted)
        self.visual_widget.plot_clean(clean_aligned)

        print(metrics)


    @staticmethod
    def get_latest_checkpoint(checkpoints_dir):
        checkpoints = os.listdir(checkpoints_dir)
        try:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x[11:16]))
            return os.path.join(checkpoints_dir, latest_checkpoint)
        except ValueError:
            return checkpoints[0]

    @staticmethod
    def get_best_checkpoint(checkpoints_dir):
        checkpoints = os.listdir(checkpoints_dir)
        try:
            best_checkpoint = min(checkpoints, key=lambda x: float(x[17:22]))
            return os.path.join(checkpoints_dir, best_checkpoint)
        except ValueError:
            return checkpoints[0]

    @staticmethod
    def set_label_empty(label: QLabel):
        label.setText("None")
        label.setEnabled(False)

    @staticmethod
    def set_label_text_enable(label: QLabel, text: str):
        label.setText(text)
        label.setEnabled(True)
