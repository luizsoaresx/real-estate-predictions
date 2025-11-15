from pathlib import Path
import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR = os.path.dirname(BASE_DIR)                   
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataCollector")


ROOT = Path(__file__).resolve().parents[1] 
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class DataCollector:
    def __init__(self, request_timeout: int = 30, max_retries: int = 3, sleep_base: float = 1.2):
        self.base_urls = {
            "bcb": "https://api.bcb.gov.br/dados/serie/bcdata.sgs",
            "ibge_localidades": "https://servicodados.ibge.gov.br/api/v1/localidades",
            "ibge_agregados": "https://servicodados.ibge.gov.br/api/v3/agregados",
            "brasil_api": "https://brasilapi.com.br/api"
        }

        self.bcb_series = {
            "ipca": 433,
            "selic": 432,
            "pib": 4380,
            "cambio": 1,
            "desemprego": 24369
        }

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.sleep_base = sleep_base

    def _get(self, url: str, params: Dict[str, Any] = None, timeout: int = None) -> Any:
        timeout = timeout or self.request_timeout
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                last_exc = exc
                wait = self.sleep_base * attempt
                logger.warning(f"Request failed ({attempt}/{self.max_retries}) for {url}: {exc} — sleeping {wait:.1f}s")
                time.sleep(wait)
        logger.error(f"All retries failed for {url}: {last_exc}")
        raise last_exc

    def coletar_dados_bacen(self, serie: str, data_inicio: str, data_fim: str) -> pd.DataFrame:
        try:
            codigo = self.bcb_series.get(serie)
            if not codigo:
                raise ValueError(f"Série Bacen '{serie}' não mapeada.")

            url = f"{self.base_urls['bcb']}.{codigo}/dados"
            params = {"formato": "json", "dataInicial": data_inicio, "dataFinal": data_fim}

            logger.info(f"Coletando {serie.upper()} do Bacen: {url} ({data_inicio} -> {data_fim})")
            resp = self._get(url, params=params)
            dados = resp.json()

            if not isinstance(dados, list) or len(dados) == 0:
                logger.warning(f"Nenhum dado retornado para {serie}")
                return pd.DataFrame()

            df = pd.DataFrame(dados)

            if "data" in df.columns:
                df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
            else:
                df["data"] = pd.NaT

            if "valor" in df.columns:
                df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
            else:
                df["valor"] = np.nan

            df["serie"] = serie
            logger.info(f"{len(df)} registros coletados de {serie.upper()}")
            return df[["data", "valor", "serie"]]

        except Exception as e:
            logger.error(f"Erro ao coletar Bacen ({serie}): {e}")
            return pd.DataFrame()

    def coletar_municipios_brasil(self) -> pd.DataFrame:
        try:
            url = f"{self.base_urls['ibge_localidades']}/municipios"
            logger.info("Coletando municípios do IBGE...")
            resp = self._get(url, timeout=60)
            municipios = resp.json()

            dados = []
            for m in municipios:
                micror = m.get("microrregiao") or {}
                mesor = micror.get("mesorregiao") or {}
                uf = mesor.get("UF") or {}
                regiao = uf.get("regiao") or {}

                dados.append({
                    "codigo_municipio": m.get("id"),
                    "nome_municipio": m.get("nome"),
                    "codigo_uf": uf.get("id"),
                    "uf": uf.get("sigla"),
                    "regiao": regiao.get("nome")
                })

            df = pd.DataFrame(dados)
            logger.info(f"{len(df)} municípios coletados (IBGE)")
            return df

        except Exception as e:
            logger.error(f"Erro ao coletar municípios IBGE: {e}")
            return pd.DataFrame()

    def coletar_pib_municipal(self, ano: int = 2021) -> pd.DataFrame:
        try:
            url = f"{self.base_urls['ibge_agregados']}/5938/periodos/{ano}/variaveis/37?localidades=N6[all]"
            logger.info(f"Coletando PIB municipal {ano} (IBGE agregados): {url}")
            resp = self._get(url, timeout=90)
            dados = resp.json()

            pib_data = []
            resultados = dados[0].get("resultados") if isinstance(dados, list) and len(dados) > 0 else None
            if not resultados:
                logger.warning("Estrutura inesperada no retorno do PIB municipal")
                return pd.DataFrame()

            for resultado in resultados:
                for serie in resultado.get("series", []) or []:
                    local = serie.get("localidade") or {}
                    mun_id = local.get("id")
                    serie_dict = serie.get("serie") or {}
                    valor = serie_dict.get(str(ano)) or serie_dict.get(ano)
                    if mun_id and valor and valor != "...":
                        try:
                            pib_data.append({
                                "codigo_municipio": mun_id,
                                "ano": int(ano),
                                "pib_municipal": float(str(valor).replace(",", "."))
                            })
                        except Exception:
                            logger.debug(f"Falha conversão PIB municipio {mun_id}: {valor}")

            df = pd.DataFrame(pib_data)
            logger.info(f"PIB municipal coletado: {len(df)} municípios")
            return df

        except Exception as e:
            logger.error(f"Erro ao coletar PIB municipal: {e}")
            return pd.DataFrame()
        
    def coletar_populacao_municipal(self, ano: int = 2022) -> pd.DataFrame:
        try:
            url = f"{self.base_urls['ibge_agregados']}/6579/periodos/{ano}/variaveis/9324?localidades=N6[all]"
            logger.info(f"Coletando população municipal {ano} (IBGE agregados): {url}")
            resp = self._get(url, timeout=90)
            dados = resp.json()

            pop_data = []
            resultados = dados[0].get("resultados") if isinstance(dados, list) and len(dados) > 0 else None
            if not resultados:
                logger.warning("Estrutura inesperada no retorno da população municipal")
                return pd.DataFrame()

            for resultado in resultados:
                for serie in resultado.get("series", []) or []:
                    local = serie.get("localidade") or {}
                    mun_id = local.get("id")
                    serie_dict = serie.get("serie") or {}
                    valor = serie_dict.get(str(ano)) or serie_dict.get(ano)
                    if mun_id and valor and valor != "...":
                        try:
                            pop_data.append({
                                "codigo_municipio": mun_id,
                                "ano": int(ano),
                                "populacao": int(float(str(valor).replace(",", ".")))
                            })
                        except Exception:
                            logger.debug(f"Falha conversão população municipio {mun_id}: {valor}")

            df = pd.DataFrame(pop_data)
            logger.info(f"População municipal coletada: {len(df)} municípios")
            return df

        except Exception as e:
            logger.error(f"Erro ao coletar população municipal: {e}")
            return pd.DataFrame()

    def coletar_todos_indicadores_economicos(self, anos_historico: int = 5) -> pd.DataFrame:
        data_fim = datetime.now()
        data_inicio = data_fim - timedelta(days=anos_historico * 365)
        di = data_inicio.strftime("%d/%m/%Y")
        df_str = data_fim.strftime("%d/%m/%Y")

        logger.info(f"Coletando indicadores econômicos ({di} até {df_str})")

        dfs = []
        for s in ["ipca", "selic", "pib", "cambio", "desemprego"]:
            df = self.coletar_dados_bacen(s, di, df_str)
            if not df.empty:
                dfs.append(df)
            time.sleep(0.8)

        if not dfs:
            logger.warning("Nenhum indicador econômico coletado")
            return pd.DataFrame()

        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["data", "serie"])
        try:
            df_pivot = df_all.pivot_table(index="data", columns="serie", values="valor", aggfunc="first").reset_index()
            logger.info(f"Indicadores pivotados: {len(df_pivot)} períodos")
            return df_pivot
        except Exception as e:
            logger.error(f"Erro ao pivotear indicadores: {e}")
            return pd.DataFrame()
        
    def coletar_dataset_completo(self) -> Dict[str, pd.DataFrame]:
        logger.info("Iniciando coleta completa de dados...")
        datasets: Dict[str, pd.DataFrame] = {}

        datasets["indicadores_economicos"] = self.coletar_todos_indicadores_economicos(anos_historico=5)
        time.sleep(1)

        datasets["municipios"] = self.coletar_municipios_brasil()
        time.sleep(1)

        datasets["pib_municipal"] = self.coletar_pib_municipal(ano=2021)
        time.sleep(1)

        datasets["populacao_municipal"] = self.coletar_populacao_municipal(ano=2022)

        logger.info("Coleta concluída. Resumo:")
        for k, v in datasets.items():
            logger.info(f" - {k}: {len(v)} registros")
        return datasets

    def salvar_dados_brutos(self, datasets: Dict[str, pd.DataFrame], diretorio: Path = RAW_DIR):
        if isinstance(diretorio, str):
            diretorio = Path(diretorio)
        diretorio.mkdir(parents=True, exist_ok=True)

        logger.info(f"Salvando dados brutos em {diretorio} ...")
        for nome, df in datasets.items():
            try:
                if df is None or df.empty:
                    logger.info(f"Pulando {nome} (vazio)")
                    continue
                filepath = diretorio / f"{nome}.csv"
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
                logger.info(f"{nome}.csv salvo ({len(df)} linhas) -> {filepath}")
            except Exception as e:
                logger.error(f"Erro ao salvar {nome}: {e}")

        logger.info("Salvamento concluído")

def main():
    collector = DataCollector()
    datasets = collector.coletar_dataset_completo()
    collector.salvar_dados_brutos(datasets)
    logger.info(f"Coleta finalizada. Arquivos gravados em: {RAW_DIR}")


if __name__ == "__main__":
    main()
