import sys
import os
from PyQt5 import QtWidgets
from feature_extraction.extractor import FeatureExtractor
from classification.classifier import ClassificationPipeline

class FeatureExtractionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Feature Extraction and Classification")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QtWidgets.QVBoxLayout()
        
        self.model_label = QtWidgets.QLabel("Choose Feature Extraction Model:")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(['ResNet50', 'VGG16', 'VGG19', 'EfficientNetV2L', 'InceptionV3'])
        
        self.path_label = QtWidgets.QLabel("Select Image Directory:")
        self.path_button = QtWidgets.QPushButton("Browse")
        self.path_button.clicked.connect(self.browse_folder)
        self.path_display = QtWidgets.QLabel("No folder selected")
        
        self.extract_button = QtWidgets.QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.extract_features)
        
        self.train_folder_label = QtWidgets.QLabel("Select Training Data Folder:")
        self.train_folder_button = QtWidgets.QPushButton("Browse")
        self.train_folder_button.clicked.connect(self.browse_train_folder)
        self.train_folder_display = QtWidgets.QLabel("No folder selected")
        
        self.test_folder_label = QtWidgets.QLabel("Select Testing Data Folder:")
        self.test_folder_button = QtWidgets.QPushButton("Browse")
        self.test_folder_button.clicked.connect(self.browse_test_folder)
        self.test_folder_display = QtWidgets.QLabel("No folder selected")
        
        self.classify_button = QtWidgets.QPushButton("Run Classification")
        self.classify_button.clicked.connect(self.run_classification)
        
        self.status_label = QtWidgets.QLabel("Status: Waiting...")
        
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.path_label)
        layout.addWidget(self.path_button)
        layout.addWidget(self.path_display)
        layout.addWidget(self.extract_button)
        layout.addWidget(self.train_folder_label)
        layout.addWidget(self.train_folder_button)
        layout.addWidget(self.train_folder_display)
        layout.addWidget(self.test_folder_label)
        layout.addWidget(self.test_folder_button)
        layout.addWidget(self.test_folder_display)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.path_display.setText(folder)

    def browse_train_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if folder:
            self.train_folder_display.setText(folder)

    def browse_test_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if folder:
            self.test_folder_display.setText(folder)

    def extract_features(self):
        selected_model = self.model_combo.currentText()
        directory = self.path_display.text()
        if directory == "No folder selected":
            self.status_label.setText("Status: Please select a folder!")
            return

        self.status_label.setText(f"Status: Extracting features using {selected_model}...")
        extractor = FeatureExtractor(model_name=selected_model)
        extractor.save_features_with_labels(directory)
        self.status_label.setText("Status: Feature extraction completed!")

    def run_classification(self):
        train_folder = self.train_folder_display.text()
        test_folder = self.test_folder_display.text()
        if train_folder == "No folder selected" or test_folder == "No folder selected":
            self.status_label.setText("Status: Please select training and testing folders!")
            return

        self.status_label.setText("Status: Running classification...")
        pipeline = ClassificationPipeline(train_folder, test_folder)
        pipeline.run()
        self.status_label.setText("Status: Classification completed!")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec_())
