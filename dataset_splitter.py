import os
import shutil
import logging
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.source_dir = self.config['dataset']['source_dir']
        self.output_dir = self.config['dataset']['output_dir']
        self.test_size = self.config['dataset']['test_size']

        self.train_dir = os.path.join(self.output_dir, 'Train')
        self.test_dir = os.path.join(self.output_dir, 'Test')
        self.logs_dir = os.path.join(self.output_dir, 'Logs')

        os.makedirs(self.logs_dir, exist_ok=True)

        self.config['paths']['train_folder'] = self.train_dir
        self.config['paths']['test_folder'] = self.test_dir
        self.config['paths']['log_file'] = os.path.join(self.logs_dir, 'pipeline.log')

        logging.basicConfig(filename=self.config['paths']['log_file'],
                            level=getattr(logging, self.config['logging']['log_level']),
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Salva config atualizada
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)

    def split(self):
        for class_dir in tqdm(os.listdir(self.source_dir), desc="Splitting dataset"):
            class_path = os.path.join(self.source_dir, class_dir)

            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

            if not images:
                continue

            train_files, test_files = train_test_split(images, test_size=self.test_size, stratify=[class_dir]*len(images), random_state=42)

            train_class_dir = os.path.join(self.train_dir, class_dir)
            test_class_dir = os.path.join(self.test_dir, class_dir)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            for file in train_files:
                shutil.copy2(os.path.join(class_path, file), os.path.join(train_class_dir, file))

            for file in test_files:
                shutil.copy2(os.path.join(class_path, file), os.path.join(test_class_dir, file))

            logging.info(f"Classe '{class_dir}' - {len(train_files)} imagens para treino, {len(test_files)} para teste.")

        print(f"Dataset split concluído. Treino em '{self.train_dir}', Teste em '{self.test_dir}'.")
        logging.info("Dataset split concluído.")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1]
    splitter = DatasetSplitter(config_path)
    splitter.split()
