from PyQt5 import QtWidgets, QtGui, QtCore

class AboutApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("About")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #2C2F33; color: white;")

        layout = QtWidgets.QVBoxLayout()

        title_label = QtWidgets.QLabel("About This Application")
        title_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        about_text = QtWidgets.QLabel(
            "This application is designed for feature extraction and classification using deep learning models.\n"
            "Developed by: Your Name\n"
            "Version: 1.0"
        )
        about_text.setFont(QtGui.QFont("Arial", 10))
        about_text.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(about_text)

        self.setLayout(layout)
