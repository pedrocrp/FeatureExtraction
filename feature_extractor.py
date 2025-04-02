import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import logging
import yaml
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50, VGG16, VGG19, EfficientNetV2L, InceptionV3
)
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.models import Model


class GPUManager:
    @staticmethod
    def configure_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)


class FeatureExtractor:
    def __init__(self, model_name, config):
        GPUManager.configure_gpu()
        self.model_name = model_name
        self.config = config
        self.output_dir = config['dataset']['output_dir']
        self.test_dir = os.path.join(self.output_dir, 'Test')
        self.reports_dir = os.path.join(self.output_dir, 'Reports')
        os.makedirs(self.reports_dir, exist_ok=True)

        self.class_labels = None
        self.target_size = (224, 224)
        self._load_model()

        logging.basicConfig(filename=self.config['paths']['log_file'],
                            level=getattr(logging, self.config['logging']['log_level']),
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def _load_model(self):
        if self.model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_resnet
        elif self.model_name == 'VGG16':
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            self.preprocess_input = preprocess_input_vgg16
        elif self.model_name == 'VGG19':
            base_model = VGG19(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            self.preprocess_input = preprocess_input_vgg19
        elif self.model_name == 'EfficientNetV2L':
            base_model = EfficientNetV2L(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_effnet
            self.target_size = (480, 480)
        elif self.model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_inception
            self.target_size = (299, 299)
        else:
            raise ValueError("Unsupported model: " + self.model_name)

    def extract_with_labels(self, batch_size=32):
        features, labels = [], []
        self.class_directories = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        self.class_labels = {class_dir: idx for idx, class_dir in enumerate(self.class_directories)}

        all_images = []
        all_labels = []

        for class_dir, class_idx in self.class_labels.items():
            class_path = os.path.join(self.test_dir, class_dir)
            image_files = os.listdir(class_path)
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                all_images.append(img_path)
                all_labels.append(class_idx)

        dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        dataset = dataset.batch(batch_size)

        for batch_paths, batch_labels in tqdm(dataset, total=len(all_images) // batch_size + 1, desc=f"Extracting {self.model_name}"):
            batch_imgs = []
            for img_path in batch_paths.numpy():
                img = Image.open(img_path.decode('utf-8')).resize(self.target_size).convert('RGB')
                x = image.img_to_array(img)
                batch_imgs.append(x)

            x_batch = np.array(batch_imgs)
            x_batch = self.preprocess_input(x_batch)
            batch_features = self.model.predict(x_batch, verbose=0)
            batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)

            features.extend(batch_features)
            labels.extend(batch_labels.numpy())

        return features, labels

    def save_features_with_labels(self):
        features, labels = self.extract_with_labels()
        features = np.array(features)
        labels = np.array(labels)

        npy_dir = os.path.join(self.output_dir, 'NumpyFeatures')
        os.makedirs(npy_dir, exist_ok=True)

        output_features = os.path.join(npy_dir, f'{self.model_name}_features.npy')
        output_labels = os.path.join(npy_dir, f'{self.model_name}_labels.npy')
        output_parquet = os.path.join(npy_dir, f'{self.model_name}_features.parquet')
        output_csv = os.path.join(npy_dir, f'{self.model_name}_features.csv')

        np.save(output_features, features)
        np.save(output_labels, labels)

        df = pd.DataFrame(features)
        df['label'] = labels
        df.to_parquet(output_parquet, index=False)
        df.to_csv(output_csv, index=False)

        print(f"Features salvos em {output_features}, {output_labels}, {output_parquet} e {output_csv}")
        logging.info(f"Features salvos em {output_features}, {output_labels}, {output_parquet} e {output_csv}")
        self._write_log()

    def _write_log(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_filename = os.path.join(self.reports_dir, f"{self.model_name}_feature_extraction_log.txt")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"Feature Extraction Log - {timestamp}\n")
            log_file.write(f"Model Used: {self.model_name}\n")
            log_file.write(f"Main Directory: {self.test_dir}\n")
            for class_dir in self.class_directories:
                class_idx = self.class_labels[class_dir]
                num_images = len(os.listdir(os.path.join(self.test_dir, class_dir)))
                log_file.write(f"Class '{class_dir}' (Label Index: {class_idx}) - Images Processed: {num_images}\n")
            log_file.write("\n")
        logging.info(f"Log written to {log_filename}")


class FeatureExtractionPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def run(self):
        for model_name in self.config['feature_extraction']['models']:
            extractor = FeatureExtractor(model_name, self.config)
            extractor.save_features_with_labels()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]
    pipeline = FeatureExtractionPipeline(config_path)
    pipeline.run()
