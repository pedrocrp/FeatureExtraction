from tensorflow.keras.applications import (
    ResNet50, VGG16, VGG19, EfficientNetV2L, InceptionV3
)
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_effnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.models import Model

def get_model(model_name):
    if model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output), preprocess_resnet, (224, 224)
    elif model_name == 'VGG16':
        base_model = VGG16(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output), preprocess_vgg16, (224, 224)
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output), preprocess_vgg19, (224, 224)
    elif model_name == 'EfficientNetV2L':
        base_model = EfficientNetV2L(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output), preprocess_effnet, (480, 480)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output), preprocess_inception, (299, 299)
    else:
        raise ValueError("Unsupported model. Choose from available models.")
