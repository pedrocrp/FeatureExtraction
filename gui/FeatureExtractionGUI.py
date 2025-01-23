import os
from PyQt5 import QtWidgets, QtGui, QtCore
from feature_extraction.extractor import FeatureExtractor

class FeatureExtractionApp(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Extract Features")
        self.setFixedSize(800, 600)  # Define um tamanho fixo para a janela
        self.setStyleSheet("background-color: #2C2F33; color: white;")

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(30)  # Aumenta o espaçamento geral entre os elementos

        title_label = QtWidgets.QLabel("Extract Features")
        title_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label, alignment=QtCore.Qt.AlignTop)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setVerticalSpacing(30)  # Aumenta o espaçamento vertical entre os campos
        form_layout.setHorizontalSpacing(20)  # Aumenta o espaçamento horizontal entre os campos

        model_label = QtWidgets.QLabel("Feature Extraction Model:")
        model_label.setMinimumWidth(220)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(['Select', 'ResNet50', 'VGG16', 'VGG19', 'EfficientNetV2L', 'InceptionV3', 'All'])
        self.model_combo.setCurrentIndex(0)  # Inicialmente seleciona 'Select'
        self.model_combo.setMinimumWidth(300)
        self.model_combo.setStyleSheet("padding: 8px; font-size: 12px;")
        form_layout.addWidget(model_label, 0, 0, QtCore.Qt.AlignRight)
        form_layout.addWidget(self.model_combo, 0, 1, QtCore.Qt.AlignLeft)

        path_label = QtWidgets.QLabel("Image Directory:")
        path_label.setMinimumWidth(220)
        self.path_display = QtWidgets.QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setMinimumWidth(300)
        self.path_display.setStyleSheet("background-color: white; padding: 8px;")
        self.path_button = QtWidgets.QPushButton("Browse")
        self.path_button.setFixedSize(120, 35)
        self.path_button.clicked.connect(self.browse_folder)
        form_layout.addWidget(path_label, 1, 0, QtCore.Qt.AlignRight)
        form_layout.addWidget(self.path_display, 1, 1, QtCore.Qt.AlignLeft)
        form_layout.addWidget(self.path_button, 1, 2, QtCore.Qt.AlignLeft)

        layout.addLayout(form_layout)
        layout.addSpacing(40)  # Adiciona espaçamento extra antes dos botões

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setAlignment(QtCore.Qt.AlignCenter)
        button_layout.setSpacing(20)  # Define um espaçamento maior entre os botões
        
        self.back_button = QtWidgets.QPushButton("Back")
        self.back_button.setFixedSize(140, 45)
        self.back_button.setStyleSheet("font-size: 12px; background-color: #ff4747; color: white;")
        self.back_button.clicked.connect(lambda: self.parent.setCurrentWidget(self.parent.main_menu))
        button_layout.addWidget(self.back_button)
        
        self.extract_button = QtWidgets.QPushButton("Start Extraction")
        self.extract_button.setFixedSize(200, 45)
        self.extract_button.setStyleSheet("font-size: 12px; background-color: #43B581; color: white;")
        self.extract_button.clicked.connect(self.extract_features)
        button_layout.addWidget(self.extract_button)

        layout.addLayout(button_layout)

        layout.addSpacing(20)  # Adiciona espaçamento extra antes da status label

        self.status_label = QtWidgets.QLabel("Status: Waiting...")
        self.status_label.setFont(QtGui.QFont("Arial", 12))
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.path_display.setText(folder)

    def extract_features(self):
        selected_model = self.model_combo.currentText()
        directory = self.path_display.text()
        
        if selected_model == 'Select':
            self.status_label.setText("Error: Please select a model!")
            return
        
        if not directory:
            self.status_label.setText("Error: Please select a folder!")
            return
        
        self.status_label.setText(f"Status: Extracting features using {selected_model}...")
        
        if selected_model == 'All':
            models = ['ResNet50', 'VGG16', 'VGG19', 'EfficientNetV2L', 'InceptionV3']
            for model in models:
                extractor = FeatureExtractor(model_name=model)
                extractor.save_features_with_labels(directory)
        else:
            extractor = FeatureExtractor(model_name=selected_model)
            extractor.save_features_with_labels(directory)
        
        self.status_label.setText("Status: Feature extraction completed!")
