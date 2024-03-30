import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, EfficientNetV2L, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.models import Model
import pandas as pd
import tensorflow as tf
import datetime


class FeatureExtractor:
    def __init__(self, model_name='ResNet50'):
        # Verifica se o TensorFlow foi construído com suporte a CUDA (GPU support)
        print("Is Built with CUDA: ", tf.test.is_built_with_cuda())

        # Lista as GPUs disponíveis para o TensorFlow
        print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Define a primeira GPU disponível para o TensorFlow
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # A exceção é lançada quando a visibilidade do dispositivo é definida após a inicialização do TensorFlow
                print(e)
        
        self.model_name = model_name
        self.class_labels = None

        if model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_resnet
            self.target_size = (224, 224)
        elif model_name == 'VGG16':
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            self.preprocess_input = preprocess_input_vgg16
            self.target_size = (224, 224)
        elif model_name == 'VGG19':
            base_model = VGG19(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
            self.preprocess_input = preprocess_input_vgg19
            self.target_size = (224, 224)
        elif model_name == 'EfficientNetV2L':
            base_model = EfficientNetV2L(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_effnet
            self.target_size = (480, 480)
        elif model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.preprocess_input = preprocess_input_inception
            self.target_size = (299, 299)
        else:
            raise ValueError("Unsupported model. Choose 'ResNet50', 'VGG16', 'VGG19', 'EfficientNetV2L', or 'InceptionV3'")

    def extract_with_labels(self, main_dir):
        features = []
        labels = []
        self.class_directories = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
        self.class_labels = {class_dir: idx for idx, class_dir in enumerate(self.class_directories)}

        for class_dir, class_idx in self.class_labels.items():
            class_path = os.path.join(main_dir, class_dir)
            image_files = os.listdir(class_path)
            total_images = len(image_files)

            for i, img_name in enumerate(image_files):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                feature = self._extract_from_image(img)
                features.append(feature)
                labels.append(class_idx)
                print(f"{class_dir}: {i+1}/{total_images} images processed")

        return features, labels

    def write_log(self, main_dir, class_images_processed):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_filename = f"{self.model_name}_feature_extraction_log.txt"
        with open(log_filename, 'a') as log_file:
            log_file.write(f"Feature Extraction Log - {timestamp}\n")
            log_file.write(f"Model Used: {self.model_name}\n")
            log_file.write(f"Main Directory: {main_dir}\n")
            for class_dir, num_images in class_images_processed.items():
                class_idx = self.class_labels[class_dir]
                log_file.write(f"Class '{class_dir}' (Label Index: {class_idx}) - Images Processed: {num_images}\n")
            log_file.write("\n")
        print(f"Log written to {log_filename}")

    def save_features_with_labels(self, main_dir):
        features, labels = self.extract_with_labels(main_dir)
        df = pd.DataFrame(features)
        df['label'] = labels
        output_csv = f'{self.model_name}_output.csv'
        #output_excel = f"{self.model_name}_output.xlsx"
        df.to_csv(output_csv, index=False)
        #df.to_excel(output_excel, index=False)
        print(f"Features saved to {output_csv}") # and {output_excel}")
        
        # Update the log after saving the features
        self.write_log(main_dir, {class_dir: len(os.listdir(os.path.join(main_dir, class_dir))) 
                                              for class_dir in self.class_directories})


    def _extract_from_image(self, img):
        img = img.resize(self.target_size)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

cnns = [
        'ResNet50', 
       'VGG16', 
       'VGG19', 
       'EfficientNetV2L',
       'InceptionV3']

for cnn in cnns:
    extractor = FeatureExtractor(cnn)
    extractor.save_features_with_labels('COVID_Dataset_331_RGB_split/Test')
    
# extractor = FeatureExtractor('ResNet50')
# extractor.save_features_with_labels('Brain_Cancer_RGB_3classes_axial_256x256_2024/Train')

