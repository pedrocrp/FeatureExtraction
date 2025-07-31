# 🧩 Full Classification Pipeline (Feature Extraction + Split + Classification)

Este pipeline foi desenvolvido para processamento completo de datasets de imagens, incluindo:

1. **Divisão do Dataset** (Train/Test com stratified split)
2. **Extração de Features com CNN pré-treinadas (GPU otimizada)**
3. **Classificação com modelos clássicos de Machine Learning (scikit-learn, XGBoost, CatBoost, LightGBM)**

---

## 🚀 Estrutura do Projeto

```
📂 project/
├── dataset_splitter.py
├── feature_extraction_pipeline.py
├── classification_pipeline.py
├── run_full_pipeline.sh
├── config.yaml
├── README.md
└── 📂 Results/
    ├── Confusion Matrices/
    ├── Final Scores/
    └── Logs/
```

---

## ⚙️ Requisitos

**Python 3.8+**

**Principais bibliotecas:**
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- tqdm
- catboost
- lightgbm
- xgboost
- PyYAML

Instale tudo com:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Configuração

A configuração do pipeline é feita via arquivo `config.yaml`.

Exemplo:
```yaml
dataset:
  source_dir: "/path/to/CompleteDataset"
  output_dir: "/path/to/SplittedDataset"
  test_size: 0.2

feature_extraction:
  models:
    - ResNet50
    - VGG16
    - VGG19
    - EfficientNetV2L
    - InceptionV3

classification:
  train_folder: "/path/to/train/"
  test_folder: "/path/to/test/"

logging:
  log_file: "Results/Logs/pipeline.log"
  log_level: "INFO"
```

---

## 🔥 Como executar o pipeline completo

Execute o pipeline completo com apenas um comando:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

O pipeline executará automaticamente:
1. Split do dataset.
2. Extração de features otimizadas para GPU.
3. Classificação com hyperparameter tuning.

---

## 📊 Resultados

Após a execução, você encontrará:

- Resultados de classificação em: `Results/Final Scores/`
- Matrizes de confusão em: `Results/Confusion Matrices/`
- Logs completos em: `Results/Logs/pipeline.log`
- Features extraídas em `.npy` e `.parquet`

---

## ✅ Melhorias aplicadas

- Pipeline totalmente configurável via `config.yaml`
- Logs centralizados e automatizados
- Split estratificado
- Extração de features paralelizada via GPU com batching
- Salvamento de features em `.npy` e `.parquet`
- Pipeline modular, reutilizável e escalável

---

