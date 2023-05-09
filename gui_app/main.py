from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QGroupBox, QVBoxLayout
import sys
from control_widget import ControlWidget
from visual_widget import VisualWidget


class MainWindow(QWidget):
    def __init__(self, window_name):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.control_group = QGroupBox("Control")
        self.visual_group = QGroupBox("Visual")

        self.control_group.setLayout(QVBoxLayout())
        self.visual_group.setLayout(QVBoxLayout())

        self.control_widget = ControlWidget()
        self.visual_widget = VisualWidget()

        self.control_group.layout().addWidget(self.control_widget)
        self.visual_group.layout().addWidget(self.visual_widget)

        self.layout().addWidget(self.control_group)
        self.layout().addWidget(self.visual_group)

        self.setWindowTitle(window_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(window_name="Inference GUI")
    window.show()
    app.exec()