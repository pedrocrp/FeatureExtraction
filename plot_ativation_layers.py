import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, InceptionV3, EfficientNetV2L
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pre
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_pre

# =======================
# CONFIGURAÇÃO
# =======================
img_path = 'datasets/COVID_Dataset_331_RGB/Normal/1.png'
output_dir = 'grids_multicamadas'
os.makedirs(output_dir, exist_ok=True)

# =======================
# MODELOS E CAMADAS
# =======================
model_configs = {
    "VGG16": {
        "model": VGG16(weights="imagenet", include_top=False),
        "layers": ["block1_conv1", "block2_conv1", "block3_conv1"],
        "preprocess": vgg16_pre,
        "size": (224, 224)
    },
    "VGG19": {
        "model": VGG19(weights="imagenet", include_top=False),
        "layers": ["block1_conv1", "block2_conv1", "block3_conv1"],
        "preprocess": vgg19_pre,
        "size": (224, 224)
    },
    "ResNet50": {
        "model": ResNet50(weights="imagenet", include_top=False),
        "layers": ["conv1_conv", "conv2_block1_out", "conv3_block1_out"],
        "preprocess": resnet_pre,
        "size": (224, 224)
    },
    "InceptionV3": {
        "model": InceptionV3(weights="imagenet", include_top=False),
        "layers": ["conv2d_1", "conv2d_3", "mixed0"],
        "preprocess": inception_pre,
        "size": (299, 299)
    },
    "EfficientNetV2L": {
        "model": EfficientNetV2L(weights="imagenet", include_top=False),
        "layers": ["stem_conv", "block2a_expand_activation", "block3a_expand_activation"],
        "preprocess": effnet_pre,
        "size": (480, 480)
    }
}

# =======================
# FUNÇÕES AUXILIARES
# =======================
def load_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def plot_feature_maps(feature_maps, title, save_path, num_filters=16):
    fmap = feature_maps[0]
    num_filters = min(num_filters, fmap.shape[-1])
    plt.figure(figsize=(12, 12))
    for i in range(num_filters):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(fmap[:, :, i], cmap='viridis')
        plt.axis("off")
        plt.title(f"Filtro {i + 1}", fontsize=8)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =======================
# EXECUÇÃO PARA CADA MODELO
# =======================
for model_name, cfg in model_configs.items():
    print(f"\n🔍 Processando {model_name}...")
    model = cfg["model"]
    size = cfg["size"]
    layers = cfg["layers"]
    preprocess = cfg["preprocess"]

    # Verifica se o caminho da imagem existe
    if not os.path.exists(img_path):
        print(f"❌ Imagem não encontrada em: {img_path}")
        continue

    # Carrega e pré-processa a imagem
    img = load_image(img_path, size)
    img = preprocess(img)

    for layer in layers:
        if not layer in [l.name for l in model.layers]:
            print(f"⚠️ Camada '{layer}' não encontrada no modelo {model_name}. Pulando.")
            continue

        print(f" → Gerando ativação para camada: {layer}")
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
        features = intermediate_model.predict(img)

        save_path = os.path.join(output_dir, f"{model_name}_{layer}_grid.png")
        plot_feature_maps(features, f"{model_name} - {layer}", save_path)
        print(f" ✅ Salvo em: {save_path}")
