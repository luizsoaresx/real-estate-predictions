# 📈 Predição de Valorização Imobiliária  
### Sistema de análise, predição e visualização de valorização municipal com Machine Learning

Este projeto integra **coleta de dados**, **processamento**, **modelagem preditiva** e um **dashboard interativo em Streamlit** para análise de valorização imobiliária em municípios do Brasil.  

---

## 🧠 Principais Funcionalidades

### ✔️ Coleta automatizada de dados  
- IPCA, SELIC e taxa de câmbio — Banco Central do Brasil  
- PIB Municipal, População e Densidade — IBGE  
- Indicadores complementares — Brasil API  
- Normalização e enriquecimento dos dados  

### ✔️ Modelagem preditiva  
- Pipeline completo de ML (scikit-learn)  
- Teste de múltiplos modelos:
  - Random Forest  
  - Gradient Boosting  
  - XGBoost (opcional)  
  - Ensemble (vencedor)  
- Métricas: R², RMSE, MAPE  
- Geração automática de relatório com desempenho dos modelos  

### ✔️ Dashboard interativo (Streamlit)  
Inclui:  
- Visão geral dos indicadores  
- Mapa interativo de valorização municipal  
- Ranking das regiões mais promissoras  
- Simulador de cenários macroeconômicos  
- Correlação das variáveis  
- PLN para gerar palavras-chave imobiliárias por município  

### ✔️ Relatórios
- Estatísticas gerais  
- Top municípios  
- Principais correlações  
- Recomendações de curto e médio prazo
---

## 📁 Estrutura do Projeto
```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── relatorio/
│   └── report_preditivo.md
│
├── models/
│
├── src/
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── ml_pipeline.py
│   ├── dashboard.py
│   ├── nlp_analysis.py
│   └── data_merging.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar

### 1️⃣ Criar ambiente virtual
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

### 2️⃣ Instalar dependências
```
pip install -r requirements.txt
```

### 3️⃣ Coletar e processar dados
```
python src/data_collection.py
python src/data_merging.py
python src/data_processing.py
python src/ml_pipeline.py
python src/nlp_analysis.py
```

### 4️⃣ Rodar o dashboard
```
streamlit run dashboard.py
```
---

## 🗺️ Tecnologias Utilizadas ##

- **Python 3.10+**
- **Pandas / NumPy** — limpeza, tratamento e manipulação de dados.
- **Scikit-learn** — modelos de regressão, pipelines e validação cruzada.
- **Plotly** — gráficos interativos.
- **Streamlit** — interface de dashboard para visualização dos resultados.
- **ReportLab** — geração automatizada com análises e gráficos.
- **Requests / APIs públicas** — coleta de dados do IBGE, Banco Central e Brasil API.
- **Folium / Geopandas (opcional)** — visualizações geoespaciais.

---

## 📊 Principais Insights Obtidos ##

- **Score econômico** e **PIB per capita** foram os maiores *drivers* de valorização imobiliária.
- **Municípios com maior dinamismo econômico** apresentam tendência mais forte de valorização no horizonte de 12 meses.
- O **modelo ensemble** apresentou o melhor desempenho geral, com o maior R² entre todos os modelos testados.
- A distribuição de valorização apresenta **alta concentração próxima à média nacional**, indicando estabilidade macroeconômica.
- A análise de **PLN (Processamento de Linguagem Natural)** permitiu identificar automaticamente características imobiliárias regionais a partir de descrições textuais.
