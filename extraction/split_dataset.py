import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset_into_test_and_train_sets(all_data_dir, output_dir, test_size=0.2):
    training_data_dir = os.path.join(output_dir, 'Train')
    testing_data_dir = os.path.join(output_dir, 'Test')

    for class_dir in os.listdir(all_data_dir):
        class_dir_full_path = os.path.join(all_data_dir, class_dir)
        
        # Certifica-se de que é um diretório
        if not os.path.isdir(class_dir_full_path):
            continue

        # Filtra para apenas imagens
        file_names = [fi for fi in os.listdir(class_dir_full_path) if fi.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

        if len(file_names) == 0:
            # Não há imagens, apenas continue com a próxima classe
            continue

        # Divide as imagens em treino e teste
        train_files, test_files = train_test_split(file_names, test_size=test_size)

        # Cria os diretórios de treino e teste para cada classe
        train_class_dir = os.path.join(training_data_dir, class_dir)
        test_class_dir = os.path.join(testing_data_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copia os arquivos para os diretórios correspondentes
        for file_name in train_files:
            shutil.copy2(os.path.join(class_dir_full_path, file_name), os.path.join(train_class_dir, file_name))

        for file_name in test_files:
            shutil.copy2(os.path.join(class_dir_full_path, file_name), os.path.join(test_class_dir, file_name))


split_dataset_into_test_and_train_sets('COVID_Dataset_331_RGB', '')
