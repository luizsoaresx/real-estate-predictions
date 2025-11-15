import pandas as pd
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "merged_data.csv"


def limpar(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    return " ".join(texto.split())


def carregar_csv(caminho, nome):
    print(f"Lendo {nome}: {caminho}")
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho)


def main():
    print("MERGE DOS DATASETS BRUTOS")

    path_municipios = RAW_DIR / "municipios.csv"
    path_pib = RAW_DIR / "pib_municipal.csv"
    path_indicadores = RAW_DIR / "indicadores_economicos.csv"

    df_mun = carregar_csv(path_municipios, "municipios.csv")
    df_pib = carregar_csv(path_pib, "pib_municipal.csv")
    df_ind = carregar_csv(path_indicadores, "indicadores_economicos.csv")

    for df in [df_mun, df_pib, df_ind]:
        if "nome_municipio" in df.columns:
            df["nome_municipio"] = df["nome_municipio"].apply(limpar)
        if "municipio" in df.columns:
            df["municipio"] = df["municipio"].apply(limpar)
        if "uf" in df.columns:
            df["uf"] = df["uf"].apply(limpar)

    def create_key(row):
        nome = row.get("nome_municipio") or row.get("municipio") or ""
        uf = row.get("uf", "")
        return f"{nome}__{uf}"

    df_mun["chave"] = df_mun.apply(create_key, axis=1)
    df_pib["chave"] = df_pib.apply(create_key, axis=1)
    df_ind["chave"] = df_ind.apply(create_key, axis=1)

    df_merged = df_mun.merge(df_pib, on="chave", how="left")
    df_merged = df_merged.merge(df_ind, on="chave", how="left")

    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Merge concluído: {OUTPUT_FILE}")
    print(f"Linhas: {df_merged.shape[0]}")
    print(f"Colunas: {df_merged.shape[1]}")


if __name__ == "__main__":
    main()