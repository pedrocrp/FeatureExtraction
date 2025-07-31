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
output_dir = 'ativacoes_todas_camadas'
model_name = "VGG19"  # Escolha: "VGG16", "VGG19", "ResNet50", "InceptionV3", "EfficientNetV2L"
os.makedirs(output_dir, exist_ok=True)

# =======================
# MODELOS DISPONÍVEIS
# =======================
model_configs = {
    "VGG16": {
        "model": VGG16(weights="imagenet", include_top=False),
        "preprocess": vgg16_pre,
        "size": (224, 224)
    },
    "VGG19": {
        "model": VGG19(weights="imagenet", include_top=False),
        "preprocess": vgg19_pre,
        "size": (224, 224)
    },
    "ResNet50": {
        "model": ResNet50(weights="imagenet", include_top=False),
        "preprocess": resnet_pre,
        "size": (224, 224)
    },
    "InceptionV3": {
        "model": InceptionV3(weights="imagenet", include_top=False),
        "preprocess": inception_pre,
        "size": (299, 299)
    },
    "EfficientNetV2L": {
        "model": EfficientNetV2L(weights="imagenet", include_top=False),
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
# EXECUÇÃO PARA UM MODELO
# =======================
if model_name not in model_configs:
    print(f"❌ Modelo '{model_name}' não está configurado.")
else:
    print(f"\n🔍 Gerando ativações para todas as camadas de: {model_name}")
    cfg = model_configs[model_name]
    model = cfg["model"]
    preprocess = cfg["preprocess"]
    size = cfg["size"]

    if not os.path.exists(img_path):
        print(f"❌ Imagem não encontrada em: {img_path}")
    else:
        img = load_image(img_path, size)
        img = preprocess(img)

        # Coleta camadas com saída 4D (altura x largura x canais)
        layers = [layer.name for layer in model.layers if len(layer.output.shape) == 4]

        for layer in layers:
            try:
                print(f" → Camada: {layer}")
                intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
                features = intermediate_model.predict(img)

                save_path = os.path.join(output_dir, f"{model_name}_{layer}_grid.png")
                plot_feature_maps(features, f"{model_name} - {layer}", save_path)
                print(f"   ✅ Salvo em: {save_path}")
            except Exception as e:
                print(f"⚠️ Erro na camada {layer}: {e}")
