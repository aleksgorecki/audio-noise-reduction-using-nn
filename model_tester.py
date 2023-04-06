import speech_denoising_wavenet
from PyQt5.QtWidgets import QApplication, QWidget
import sys


def main():
    app = QApplication(sys.argv)
    window = QWidget()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
