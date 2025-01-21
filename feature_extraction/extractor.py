import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import pandas as pd
import tensorflow as tf
from feature_extraction.models import get_model

class FeatureExtractor:
    def __init__(self, model_name='ResNet50'):
        self.model, self.preprocess_input, self.target_size = get_model(model_name)
        self.model_name = model_name

    def extract_with_labels(self, main_dir):
        features = []
        labels = []
        class_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
        class_labels = {class_dir: idx for idx, class_dir in enumerate(class_dirs)}

        for class_dir, class_idx in class_labels.items():
            class_path = os.path.join(main_dir, class_dir)
            image_files = os.listdir(class_path)

            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                feature = self._extract_from_image(img)
                features.append(feature)
                labels.append(class_idx)

        return features, labels

    def save_features_with_labels(self, main_dir):
        features, labels = self.extract_with_labels(main_dir)
        df = pd.DataFrame(features)
        df['label'] = labels
        output_csv = f'{self.model_name}_output.csv'
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")

    def _extract_from_image(self, img):
        img = img.resize(self.target_size)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
