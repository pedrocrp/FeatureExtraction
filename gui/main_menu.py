import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from gui.FeatureExtractionGUI import FeatureExtractionApp

class MainApp(QtWidgets.QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Extraction and Classification")
        self.setFixedSize(800, 600)
        self.setStyleSheet("background-color: #2C2F33; color: white;")
        
        self.main_menu = MainMenu(self)
        self.extract_features = FeatureExtractionApp(self)
        
        self.addWidget(self.main_menu)
        self.addWidget(self.extract_features)
        
        self.setCurrentWidget(self.main_menu)

class MainMenu(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        title_label = QtWidgets.QLabel("Feature Extraction and Classification Tool")
        title_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        button_layout = QtWidgets.QVBoxLayout()
        
        self.extract_button = QtWidgets.QPushButton("Extract Features")
        self.extract_button.setFixedSize(200, 40)
        self.extract_button.setStyleSheet("font-size: 12px; background-color: #7289DA; color: white;")
        self.extract_button.clicked.connect(lambda: self.parent.setCurrentWidget(self.parent.extract_features))
        button_layout.addWidget(self.extract_button, alignment=QtCore.Qt.AlignCenter)
        
        self.classify_button = QtWidgets.QPushButton("Classify")
        self.classify_button.setFixedSize(200, 40)
        self.classify_button.setStyleSheet("font-size: 12px; background-color: #7289DA; color: white;")
        button_layout.addWidget(self.classify_button, alignment=QtCore.Qt.AlignCenter)
        
        self.about_button = QtWidgets.QPushButton("About")
        self.about_button.setFixedSize(200, 40)
        self.about_button.setStyleSheet("font-size: 12px; background-color: #7289DA; color: white;")
        button_layout.addWidget(self.about_button, alignment=QtCore.Qt.AlignCenter)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
