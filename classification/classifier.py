import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ClassificationPipeline:
    def __init__(self, train_folder, test_folder, results_dir='../Results'):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.results_dir = results_dir
        self.create_directories()
        self.results = []

    def create_directories(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/Confusion Matrices', exist_ok=True)
        os.makedirs(f'{self.results_dir}/Final Scores', exist_ok=True)

    def hyperparameter_search(self, X_train, y_train, X_val, y_val, classifier, param_grid):
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        return classifier.set_params(**best_params)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, classifier_name, classifier, param_grid, file_name):
        model = self.hyperparameter_search(X_train, y_train, X_val, y_val, classifier, param_grid)
        model.fit(X_train, y_train)
        predictions_test = model.predict(X_test)
        report = classification_report(y_test, predictions_test, output_dict=True)
        cm = confusion_matrix(y_test, predictions_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{self.results_dir}/Confusion Matrices/{file_name}_{classifier_name}.png")
        
        self.results.append({
            "File": file_name,
            "Classifier": classifier_name,
            "Accuracy": report["accuracy"],
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-Score": report["weighted avg"]["f1-score"]
        })

    def process_dataset(self, train_file, test_file):
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        X_train, X_val, y_train, y_val = train_test_split(train_data.iloc[:, :-1], train_data.iloc[:, -1], test_size=0.2, random_state=42)
        X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

        classifiers = {
            'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), {'C': [0.1, 1, 10], 'solver': ['lbfgs'], 'penalty': ['l2']}),
            'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [1, 5, 10], 'weights': ['uniform', 'distance']}),
            'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [10, 50, 100]}),
            'Decision Tree': (DecisionTreeClassifier(random_state=42), {'max_depth': [5, 10, 20]})
        }

        file_name = os.path.basename(train_file)
        for name, (classifier, param_grid) in classifiers.items():
            self.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, name, classifier, param_grid, file_name)

    def run(self):
        for file in os.listdir(self.train_folder):
            if file.endswith('.csv'):
                train_file = os.path.join(self.train_folder, file)
                test_file = os.path.join(self.test_folder, file)
                self.process_dataset(train_file, test_file)
        pd.DataFrame(self.results).to_csv(f"{self.results_dir}/Final Scores/classification_results.csv", index=False)

if __name__ == "__main__":
    train_folder = "../data/train"
    test_folder = "../data/test"
    pipeline = ClassificationPipeline(train_folder, test_folder)
    pipeline.run()