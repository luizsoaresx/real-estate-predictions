import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime


class MLPipeline:

    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.data_path = os.path.join(self.base_path, "data", "processed", "dataset_ml.csv")
        self.models_path = os.path.join(self.base_path, "models")
        os.makedirs(self.models_path, exist_ok=True)
        self.target_col = "valorizacao_esperada_12m"

    def carregar_dados(self):
        print(f"[ML] Carregando dados de {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"[ML] Dados carregados: {df.shape}")
        return df

    def preparar_features_target(self, df):
        print("[ML] Preparando features e target...")

        if self.target_col not in df.columns:
            raise ValueError(f"[ML] A coluna alvo '{self.target_col}' nao existe no dataset.")

        drop_categoricas = df.select_dtypes(include=["object"]).columns.tolist()
        print(f"[ML] Removendo colunas categoricas: {drop_categoricas}")
        df = df.drop(columns=drop_categoricas)

        cols_all_nan = df.columns[df.isna().all()].tolist()
        if len(cols_all_nan) > 0:
            print(f"[ML] Removendo colunas totalmente NaN: {cols_all_nan}")
            df = df.drop(columns=cols_all_nan)

        df = df.dropna(subset=[self.target_col])

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if X.isna().sum().sum() > 0:
            print("[ML] Preenchendo NaN nas features com media...")
            X = X.fillna(X.mean())

        cols_zero_var = X.columns[X.var() == 0].tolist()
        if len(cols_zero_var) > 0:
            print(f"[ML] Removendo features com variancia zero: {cols_zero_var}")
            X = X.drop(columns=cols_zero_var)

        if X.shape[1] == 0:
            print("[ML] AVISO: Nenhuma feature disponivel! Criando feature dummy...")
            X = pd.DataFrame({'feature_dummy': np.random.randn(len(X))})

        print(f"[ML] Features finais: {X.shape}, Target shape: {y.shape}")
        print(f"[ML] Colunas utilizadas: {list(X.columns)}")
        return X, y, df

    def normalizar(self, X_train, X_test):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

        cols_all_nan_train = X_train.columns[X_train.isna().all()].tolist()
        if len(cols_all_nan_train) > 0:
            print(f"[ML] Removendo colunas problematicas antes de normalizar: {cols_all_nan_train}")
            X_train = X_train.drop(columns=cols_all_nan_train)
            X_test = X_test.drop(columns=cols_all_nan_train, errors="ignore")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler_path = os.path.join(self.models_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"[ML] Scaler salvo em {scaler_path}")

        return X_train_scaled, X_test_scaled

    def criar_modelo(self, input_dim):
        print("[ML] Criando modelo neural...")
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def treinar(self, model, X_train, y_train, X_test, y_test):
        print("[ML] Treinando modelo...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=40,
            batch_size=32,
            verbose=1
        )
        model_path = os.path.join(self.models_path, "modelo_regressao.keras")
        model.save(model_path)
        print(f"[ML] Modelo salvo em {model_path}")
        return history

    def avaliar(self, model, X_test, y_test):
        print("[ML] Avaliando modelo...")
        preds = model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        print("\n[ML] RESULTADOS FINAIS")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        return preds

    def pipeline_completo(self):
        print("\n[ML PIPELINE] INICIANDO PIPELINE DE MACHINE LEARNING\n")

        df = self.carregar_dados()
        X, y, _ = self.preparar_features_target(df)

        print("[ML PIPELINE] Separando treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled, X_test_scaled = self.normalizar(X_train, X_test)

        model = self.criar_modelo(X_train_scaled.shape[1])

        self.treinar(model, X_train_scaled, y_train, X_test_scaled, y_test)

        self.avaliar(model, X_test_scaled, y_test)

        print("\n[ML PIPELINE] Pipeline concluido com sucesso!\n")


def main():
    pipeline = MLPipeline()
    pipeline.pipeline_completo()


if __name__ == "__main__":
    main()
