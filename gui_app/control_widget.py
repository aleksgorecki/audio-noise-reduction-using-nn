from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel


class ControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.run_button = QPushButton("Run prediction")
        self.noisy_input_button = QPushButton("Choose noisy input")
        self.clean_input_button = QPushButton("Choose clean input")
        self.model_config_button = QPushButton("Choose model config")
        self.model_weights_button = QPushButton("Choose model weights")

        self.noisy_input_dialog = QFileDialog(caption="Noisy input")
        # self.clean_input_dialog = QFileDialog("Clean input")
        # self.model_config_dialog = QFileDialog("Model config")
        # self.model_weights_dialog = QFileDialog("Model weights")

        self.noisy_input_button.clicked.connect(self.noisy_input_onclick)

        self.debug_label = QLabel("default")

        self.layout().addWidget(self.debug_label)
        self.layout().addWidget(self.run_button)
        self.layout().addWidget(self.noisy_input_button)
        self.layout().addWidget(self.clean_input_button)
        self.layout().addWidget(self.model_config_button)
        self.layout().addWidget(self.model_weights_button)

    def noisy_input_onclick(self):
        self.debug_label.setText("noisy clicked")
        self.noisy_input_dialog.show()
        filename = self.noisy_input_dialog.getOpenFileName()
        self.debug_label.setText(filename)

    def clean_input_onclick(self):
        pass

    def model_config_onclick(self):
        pass

    def model_weights_onclick(self):
        pass

