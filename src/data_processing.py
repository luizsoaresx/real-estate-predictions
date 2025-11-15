import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[DataProcessing] %(message)s')

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class DataProcessor:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        logging.info(f'Carregando dados de {self.input_path}...')
        self.df = pd.read_csv(self.input_path)
        logging.info(f'Dados carregados. Linhas: {len(self.df)}')

    def clean_data(self):
        logging.info('Removendo duplicatas...')
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        logging.info(f'Duplicatas removidas: {before - len(self.df)}')

        numeric = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric] = self.df[numeric].fillna(self.df[numeric].median())

    def create_target_variable(self):
        logging.info('Criando variável alvo e features...')
        
        np.random.seed(42)
        
        if 'pib_per_capita' not in self.df.columns or self.df['pib_per_capita'].isna().all():
            self.df['pib_per_capita'] = np.random.uniform(10000, 80000, len(self.df))
            logging.info('PIB per capita gerado')
        
        if 'populacao' not in self.df.columns or self.df['populacao'].isna().all():
            self.df['populacao'] = np.random.uniform(5000, 500000, len(self.df))
            logging.info('População gerada')
        
        if 'ipca' not in self.df.columns or self.df['ipca'].isna().all():
            self.df['ipca'] = np.random.uniform(4, 12, len(self.df))
            logging.info('IPCA gerado')
        
        if 'selic' not in self.df.columns or self.df['selic'].isna().all():
            self.df['selic'] = np.random.uniform(8, 14, len(self.df))
            logging.info('SELIC gerada')
        
        if 'score_economico' not in self.df.columns:
            pib_norm = (self.df['pib_per_capita'] - self.df['pib_per_capita'].min()) / (self.df['pib_per_capita'].max() - self.df['pib_per_capita'].min())
            pop_norm = (self.df['populacao'] - self.df['populacao'].min()) / (self.df['populacao'].max() - self.df['populacao'].min())
            ipca_norm = (self.df['ipca'] - self.df['ipca'].min()) / (self.df['ipca'].max() - self.df['ipca'].min())
            self.df['score_economico'] = (pib_norm * 0.5 + pop_norm * 0.3 + ipca_norm * 0.2) * 100
            logging.info('Score econômico calculado')
        
        pib_norm = (self.df['pib_per_capita'] - self.df['pib_per_capita'].min()) / (self.df['pib_per_capita'].max() - self.df['pib_per_capita'].min())
        pop_norm = (self.df['populacao'] - self.df['populacao'].min()) / (self.df['populacao'].max() - self.df['populacao'].min())
        selic_inv = 1 - (self.df['selic'] - self.df['selic'].min()) / (self.df['selic'].max() - self.df['selic'].min())
        
        self.df['valorizacao_esperada_12m'] = (
            pib_norm * 15 + 
            pop_norm * 10 + 
            selic_inv * 5 +
            np.random.normal(0, 0.5, len(self.df))
        ) / 30 * 30  
        
        self.df['valorizacao_esperada_12m'] = self.df['valorizacao_esperada_12m'].clip(0, 30)
        
        logging.info(f'Variável alvo criada! Média: {self.df["valorizacao_esperada_12m"].mean():.2f}%')

    def normalize_features(self):
        numeric = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            logging.info('Nenhuma coluna numérica para normalizar.')
            return

        scaler = MinMaxScaler()
        self.df[numeric] = scaler.fit_transform(self.df[numeric])

        logging.info(f'Normalizadas: {list(numeric)}')

    def save_processed(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f'Salvando em {self.output_path}...')
        self.df.to_csv(self.output_path, index=False)

        logging.info('Arquivo salvo com sucesso!')

    def run(self):
        self.load_data()
        self.clean_data()
        self.create_target_variable()
        self.normalize_features()
        self.save_processed()


if __name__ == "__main__":
    processor = DataProcessor(
        input_path=RAW_DIR / "merged_data.csv",
        output_path=PROCESSED_DIR / "dataset_ml.csv"
    )
    processor.run()
