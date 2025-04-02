#!/bin/bash

set -e  # Faz o script parar imediatamente se qualquer comando falhar

# Caminho do arquivo de configuração
config_path="config.yaml"

echo "🚀 Iniciando pipeline completo..."

# echo "➡️  Split do dataset..."
# if ! python dataset_splitter.py "$config_path"; then
#     echo "❌ Erro na etapa de Split."
#     exit 1
# fi

# echo "➡️  Extração de features..."
# if ! python feature_extractor.py "$config_path"; then
#     echo "❌ Erro na etapa de Extração de Features."
#     exit 1
# fi

# echo "➡️  Classificação..."
# if ! python classifier.py "$config_path"; then
#     echo "❌ Erro na etapa de Classificação."
#     exit 1
# fi

echo "➡️  Geração de relatório..."
if ! python generate_report.py "$config_path"; then
    echo "❌ Erro na etapa de Relatório."
    exit 1
fi

echo "✅ Pipeline completo concluído com sucesso!"
