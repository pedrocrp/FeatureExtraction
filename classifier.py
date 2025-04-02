import os
import gc
import yaml
import time
import logging
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
import sys

class DirectoryManager:
    @staticmethod
    def create_directories(base_dir):
        os.makedirs(os.path.join(base_dir, 'Results/Confusion Matrices'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'Results/Final Scores'), exist_ok=True)


class HyperparameterSearcher:
    @staticmethod
    def search(X_train, y_train, X_val, y_val, classifier, param_grid):
        logging.info(f"Buscando hiperparâmetros para {type(classifier).__name__}...")
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3,
                                   scoring='f1_weighted', n_jobs=6, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = classifier.set_params(**best_params)
        model.fit(X_train, y_train)
        report = classification_report(y_val, model.predict(X_val), output_dict=True)
        score_val = report["accuracy"]
        val_f1 = report["weighted avg"]["f1-score"]
        val_prec = report["weighted avg"]["precision"]
        val_rec = report["weighted avg"]["recall"]

        del grid_search
        gc.collect()

        return best_params, score_val, val_f1, val_prec, val_rec


class ModelTrainer:
    def __init__(self, file_name, results, output_dir):
        self.file_name = file_name
        self.results = results
        self.output_dir = output_dir

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, name, classifier, param_grid):
        try:
            logging.info(f"\u25b6\ufe0f Iniciando {name} para arquivo {self.file_name}")
            print(f"\u25b6\ufe0f Iniciando {name} para arquivo {self.file_name}")
            start_time = time.time()
            self.y_test = y_test

            best_params, val_acc, val_f1, val_prec, val_rec = HyperparameterSearcher.search(X_train, y_train, X_val, y_val, classifier, param_grid)
            model = classifier.set_params(**best_params)
            model.fit(X_train, y_train)

            predictions_test = model.predict(X_test)
            test_report = classification_report(y_test, predictions_test, output_dict=True)

            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
            mean_cv_score = cv_scores.mean()

            try:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            except:
                roc_auc = None

            mcc = matthews_corrcoef(y_test, predictions_test)
            elapsed_time = time.time() - start_time

            result = {
                "Arquivo Feature": self.file_name,
                "Classifier": name,
                "Melhores Hiperparâmetros": best_params,
                "Val_Accuracy": val_acc,
                "Val_F1": val_f1,
                "Val_Precision": val_prec,
                "Val_Recall": val_rec,
                "Test_Accuracy": test_report["accuracy"],
                "Test_Precision": test_report["weighted avg"]["precision"],
                "Test_Recall": test_report["weighted avg"]["recall"],
                "Test_F1": test_report["weighted avg"]["f1-score"],
                "CV_F1_Score": mean_cv_score,
                "ROC-AUC": roc_auc,
                "MCC": mcc,
                "Tempo Execucao (s)": elapsed_time
            }

            self.save_results(result, model, predictions_test)
            logging.info(f"\u2705 Finalizado {name}")
            print(f"\u2705 Finalizado {name} para arquivo {self.file_name}")
        except Exception as e:
            logging.error(f"\u274c Erro no classificador {name}: {e}")
            print(f"\u274c Erro no classificador {name}: {e}")
            raise e

    def save_results(self, result, model, predictions):
        results_dir = os.path.join(self.output_dir, 'Results/Final Scores')
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame([result])
        output_path = os.path.join(results_dir, f"{self.file_name}_{result['Classifier']}_results.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Resultados salvos em: {output_path}")

        cm_dir = os.path.join(self.output_dir, 'Results/Confusion Matrices', self.file_name)
        os.makedirs(cm_dir, exist_ok=True)
        cm = confusion_matrix(self.y_test, predictions, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title(f"Matriz de Confusão para {result['Classifier']} - {self.file_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f"{self.file_name}_{result['Classifier']}.png"))
        plt.close()

        self.results.append(result)


class DatasetProcessor:
    def __init__(self, feature_file, results, output_dir, val_size):
        self.feature_file = feature_file
        self.results = results
        self.output_dir = output_dir
        self.val_size = val_size

    def process(self):
        data = pd.read_csv(self.feature_file)

        X = data.drop('label', axis=1)
        y = data['label']

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        file_name = os.path.splitext(os.path.basename(self.feature_file))[0]
        trainer = ModelTrainer(file_name, self.results, self.output_dir)

        param_grids = self.get_param_grids()
        classifiers = self.get_classifiers()

        for name, clf in tqdm(classifiers, desc=f"Processando {file_name}"):
            trainer.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, name, clf, param_grids.get(name, {}))

    @staticmethod
    def get_classifiers():
        return [
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
            ('K-Nearest Neighbors', KNeighborsClassifier()),
            ('Random Forest', RandomForestClassifier(random_state=42)),
            ('Decision Tree', DecisionTreeClassifier(random_state=42)),
            ('XGB', XGBClassifier(random_state=42)),
            ('SVM', SVC(probability=True, random_state=42)),
            ('Naive Bayes', GaussianNB()),
            ('AdaBoost', AdaBoostClassifier(random_state=42)),
            ('MLP Classifier', MLPClassifier(random_state=42)),
            ('SGD', SGDClassifier(random_state=42)),
            ('QDA', QuadraticDiscriminantAnalysis()),
            ('LightGBM', LGBMClassifier(random_state=42, verbose=-1))
        ]

    @staticmethod
    def get_param_grids():
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        return config['classification']['param_grids']


class ClassifierPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.output_dir = self.config['dataset']['output_dir']
        self.features_dir = os.path.join(self.output_dir, 'NumpyFeatures')
        self.val_size = self.config['classification'].get('validation_size', 0.2)
        self.results = []

        logging.basicConfig(filename=self.config['paths']['log_file'],
                            level=getattr(logging, self.config['logging']['log_level']),
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def run(self):
        DirectoryManager.create_directories(self.output_dir)

        feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.csv')]

        for feature_file in tqdm(feature_files, desc="Executando Pipeline"):
            feature_path = os.path.join(self.features_dir, feature_file)
            processor = DatasetProcessor(feature_path, self.results, self.output_dir, self.val_size)
            processor.process()

        self.save_final_results()

    def save_final_results(self):
        df = pd.DataFrame(self.results)
        print(self.results)
        results_dir = os.path.join(self.output_dir, 'Results/Final Scores')
        os.makedirs(results_dir, exist_ok=True)

        if not df.empty:
            df.sort_values(by="CV_F1_Score", ascending=False, inplace=True)
            df.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
            logging.info("Resultados finais salvos.")
        else:
            logging.warning("Nenhum resultado coletado.")
            print("❌ Nenhum resultado coletado. Classificação encerrada sem resultados.")
            

if __name__ == "__main__":
    config_path = sys.argv[1]
    pipeline = ClassifierPipeline(config_path)
    pipeline.run()
    print("✅ Classificação concluída.")
