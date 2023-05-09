import os.path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel, QCheckBox, QRadioButton, QGroupBox, QHBoxLayout
from PyQt5.QtCore import pyqtSlot
import pathlib


class ControlWidget(QWidget):

    WEIGHTS_SETTINGS = ["best", "latest", "custom"]

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.run_button = QPushButton("Run prediction", parent=self)

        # self.clean_input_button = QPushButton("Select clean input", parent=self)
        # self.model_config_button = QPushButton("Select model config", parent=self)
        # self.model_weights_button = QPushButton("Select model weights", parent=self)
        self.noisy_input_group = QGroupBox("Noisy input", parent=self)
        self.model_session_group = QGroupBox("Model session", parent=self)

        self.noisy_input_group.setLayout(QHBoxLayout())
        self.model_session_group.setLayout(QHBoxLayout())

        self.noisy_input_button = QPushButton("Select", parent=self.noisy_input_group)
        self.noisy_input_label = QLabel("None", parent=self.noisy_input_group)
        self.noisy_input_group.layout().addWidget(self.noisy_input_label)
        self.noisy_input_group.layout().addWidget(self.noisy_input_button)

        self.model_session_label = QLabel("None", parent=self.model_session_group)
        self.model_session_button = QPushButton("Select", parent=self.model_session_group)
        self.model_session_group.layout().addWidget(self.model_session_label)
        self.model_session_group.layout().addWidget(self.model_session_button)

        self.weights_group = QGroupBox("Weights selection", parent=self)

        self.best_weights_radio = QRadioButton("Use best weights", parent=self.weights_group)
        self.latest_weights_radio = QRadioButton("Use latest weights", parent=self.weights_group)
        self.custom_weights = QRadioButton("Use selected weights", parent=self.weights_group)
        self.custom_weights_button = QPushButton("Select", parent=self.weights_group)

        self.weights_group.setLayout(QVBoxLayout())
        self.weights_group.layout().addWidget(self.best_weights_radio)
        self.weights_group.layout().addWidget(self.latest_weights_radio)
        self.weights_group.layout().addWidget(self.custom_weights)

        self.custom_weights_label = QLabel("None")
        self.custom_weights_widget = QWidget(parent=self.weights_group)
        self.custom_weights_widget.setLayout(QHBoxLayout())
        self.custom_weights_widget.layout().addWidget(self.custom_weights_label)
        self.custom_weights_widget.layout().addWidget(self.custom_weights_button)

        self.weights_group.layout().addWidget(self.custom_weights_widget)

        self.custom_weights_widget.hide()

        self.general_groupbox = QGroupBox("General settings", parent=self)
        self.general_groupbox.setLayout(QVBoxLayout())
        self.metrics_checkbox = QCheckBox("Calculate metrics", parent=self.general_groupbox)
        self.save_result_checkbox = QCheckBox("Save result", parent=self.general_groupbox)
        self.show_extra_checkbox = QCheckBox("Show extra options", parent=self.general_groupbox)
        self.general_groupbox.layout().addWidget(self.metrics_checkbox)
        self.general_groupbox.layout().addWidget(self.show_extra_checkbox)
        self.general_groupbox.layout().addWidget(self.save_result_checkbox)
        # self.noisy_input_dialog = QFileDialog(caption="Noisy input", parent=self)
        # self.noisy_input_dialog.setFileMode(QFileDialog.ExistingFile)
        # self.clean_input_dialog = QFileDialog(caption="Noisy input", parent=self)
        # self.clean_input_dialog.setFileMode(QFileDialog.ExistingFile)
        # self.model_config_dialog = QFileDialog("Model config")
        # self.model_weights_dialog = QFileDialog("Model weights")

        self.noisy_input_button.clicked.connect(self.noisy_input_onclick)
        # self.clean_input_button.clicked.connect(self.clean_input_onclick)
        self.model_session_button.clicked.connect(self.session_onclick)

        self.layout().addWidget(self.run_button)
        self.layout().addWidget(self.noisy_input_group)
        self.layout().addWidget(self.model_session_group)
        self.layout().addWidget(self.weights_group)
        self.layout().addWidget(self.general_groupbox)
        # self.layout().addWidget(self.clean_input_button)
        # self.layout().addWidget(self.model_config_button)
        # self.layout().addWidget(self.model_weights_button)


        self.noisy_input_path = ""
        self.last_noisy_dir = ""
        # self.clean_input_path = ""
        # self.model_config_path = ""
        # self.model_weights_path = ""
        self.session_path = ""
        self.last_session_dir = ""

    @pyqtSlot()
    def noisy_input_onclick(self):
        options = QFileDialog.Options()
        noisy_input = QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a file to denoise"), options=options, filter=self.tr("Wav File (*.wav)"))[0]
        if noisy_input != "":
            self.noisy_input_path = noisy_input
            self.last_noisy_dir = os.path.dirname(noisy_input)
        self.noisy_input_label.setText(str(pathlib.Path(self.noisy_input_path).name))


    @pyqtSlot()
    def session_onclick(self):
        session_dir = QFileDialog.getExistingDirectory(self, 'Select session directory')
        if session_dir != "":
            self.session_path = session_dir
            self.last_session_dir = os.path.dirname(session_dir)
        self.model_session_label.setText(str(pathlib.Path(self.session_path).name))
    # @pyqtSlot()
    # def clean_input_onclick(self):
    #     options = QFileDialog.Options()
    #     last_dir = ""
    #     if self.clean_input_path is not None:
    #         last_dir = os.path.dirname(self.clean_input_path)
    #     if last_dir != "":
    #         self.clean_input_path = \
    #         QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a clean reference file"), directory=last_dir, options=options,
    #                                     filter=self.tr("Wav File (*.wav)"))[0]
    #     else:
    #         self.clean_input_path = QFileDialog.getOpenFileName(parent=self, caption=self.tr("Select a clean reference file"), options=options, filter=self.tr("Wav File (*.wav)"))[0]
    #     self.debug_label.setText(str(self.noisy_input_path))
    #
    # @pyqtSlot()
    # def model_config_onclick(self):
    #     pass
    #
    # @pyqtSlot()
    # def model_weights_onclick(self):
    #     pass

