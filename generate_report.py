import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys


class ReportGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.output_dir = self.config['dataset']['output_dir']
        self.results_dir = os.path.join(self.output_dir, 'Results/Final Scores')
        self.reports_dir = os.path.join(self.output_dir, 'Reports')
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate(self):
        all_results = []

        for file in os.listdir(self.results_dir):
            if file.endswith('_results.csv'):
                df = pd.read_csv(os.path.join(self.results_dir, file))
                all_results.append(df)

        if not all_results:
            print("Nenhum resultado encontrado.")
            return

        final_df = pd.concat(all_results, ignore_index=True)
        ranking = final_df.sort_values(by="Val_F1", ascending=False)

        # Salvar consolidado
        ranking.to_parquet(os.path.join(self.reports_dir, 'final_report.parquet'), index=False)
        ranking.to_csv(os.path.join(self.reports_dir, 'final_report.csv'), index=False)

        print("Relatório consolidado salvo.")

        # ======= GRÁFICOS =======

        # Ranking F1-Score
        ranking = final_df.sort_values(by="Val_F1", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=ranking, x="Classifier", y="Val_F1", hue="Arquivo Feature")
        plt.xticks(rotation=45)
        plt.title("Ranking de Classificadores por F1-Score de Validação")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "ranking_f1_score.png"))
        plt.close()

        # Tempo de execução
        plt.figure(figsize=(12, 6))
        sns.barplot(data=final_df, x="Classifier", y="Tempo Execucao (s)", hue="Arquivo Feature")
        plt.xticks(rotation=45)
        plt.title("Tempo de Execução por Classificador")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "tempo_execucao.png"))
        plt.close()

        # Scatter F1-Score vs ROC-AUC
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=final_df, x="Val_F1", y="ROC-AUC", hue="Classifier")
        plt.title("F1-Score vs ROC-AUC")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "f1_vs_roc_auc.png"))
        plt.close()

        print(f"Relatórios e gráficos salvos em: {self.reports_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1]
    generator = ReportGenerator(config_path)
    generator.generate()
