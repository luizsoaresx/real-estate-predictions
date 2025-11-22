import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
from collections import Counter
import re
warnings.filterwarnings('ignore')

from nlp_analysis import ImobiliarioPLN

st.set_page_config(
    page_title="Preditor Imobiliário Regional",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def carregar_dados():
    import os
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        df_ml = pd.read_csv(os.path.join(base_dir, 'data/processed/dataset_ml.csv'))
        
        indicadores_path = os.path.join(base_dir, 'data/processed/indicadores_processados.csv')
        if os.path.exists(indicadores_path):
            df_indicadores = pd.read_csv(indicadores_path)
        else:
            df_indicadores = pd.DataFrame()
        
        metrics = {}
        metrics_path = os.path.join(base_dir, 'data/models/metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return df_ml, df_indicadores, metrics
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}


def criar_mapa_valorizacao(df):

    if 'valorizacao_esperada_12m' not in df.columns:
        st.warning("Coluna 'valorizacao_esperada_12m' não encontrada.")
        return None

    coords_uf = {
        "AC": (-9.0238, -70.8120),
        "AL": (-9.5713, -36.7820),
        "AM": (-3.4168, -65.8561),
        "AP": (1.3260, -51.9160),
        "BA": (-12.5797, -41.7007),
        "CE": (-5.4984, -39.3206),
        "DF": (-15.8267, -47.9218),
        "ES": (-19.1834, -40.3089),
        "GO": (-15.8270, -49.8362),
        "MA": (-4.9609, -45.2744),
        "MG": (-18.5122, -44.5550),
        "MS": (-20.7722, -54.7852),
        "MT": (-12.6819, -55.4250),
        "PA": (-3.4168, -52.9088),
        "PB": (-7.2399, -36.7819),
        "PE": (-8.8137, -36.9541),
        "PI": (-6.5990, -42.2800),
        "PR": (-24.4843, -51.6460),
        "RJ": (-22.9068, -43.1729),
        "RN": (-5.4026, -36.9541),
        "RO": (-11.5057, -63.5806),
        "RR": (2.7376, -62.0751),
        "RS": (-30.0346, -51.2177),
        "SC": (-27.2423, -50.2187),
        "SE": (-10.5741, -37.3857),
        "SP": (-22.5180, -48.6350),
        "TO": (-10.1753, -48.2982)
    }

    df_map = df.copy()

    if "uf" not in df_map.columns:
        st.error("Dataset precisa conter a coluna 'uf' para construir o mapa.")
        return None

    df_map["lat"] = df_map["uf"].apply(lambda x: coords_uf.get(x, (None, None))[0])
    df_map["lon"] = df_map["uf"].apply(lambda x: coords_uf.get(x, (None, None))[1])

    df_map = df_map.dropna(subset=["lat", "lon"])

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="valorizacao_esperada_12m",
        size="populacao" if "populacao" in df_map.columns else None,
        hover_name="nome_municipio" if "nome_municipio" in df_map.columns else "uf",
        mapbox_style="carto-positron",
        color_continuous_scale="RdYlGn",
        size_max=22,
        zoom=3.4,
        title="Mapa de Valorização Imobiliária (Coordenadas Reais por UF)",
    )

    fig.update_layout(height=600)
    return fig



def criar_ranking_regioes(df, top_n=12):
    if 'valorizacao_esperada_12m' not in df.columns:
        st.warning("Coluna 'valorizacao_esperada_12m' não encontrada.")
        return None, pd.DataFrame()
    
    colunas_obrigatorias = ['valorizacao_esperada_12m']
    colunas_opcionais = ['nome_municipio', 'uf', 'regiao', 'pib_per_capita', 'populacao', 'score_economico']
    colunas_ranking = colunas_obrigatorias + [c for c in colunas_opcionais if c in df.columns]
    
    df_ranking = df.nlargest(top_n, 'valorizacao_esperada_12m')[colunas_ranking].copy()
    
    if len(df_ranking) == 0:
        st.warning("Nenhum dado disponível para ranking.")
        return None, pd.DataFrame()
    
    df_ranking['posicao'] = range(1, len(df_ranking) + 1)
    
    fig = px.bar(
        df_ranking,
        x='valorizacao_esperada_12m',
        y='nome_municipio' if 'nome_municipio' in df_ranking.columns else None,
        orientation='h',
        color='valorizacao_esperada_12m',
        color_continuous_scale='Greens',
        title=f'Top {top_n} Municípios com Maior Valorização Esperada',
        labels={
            'valorizacao_esperada_12m': 'Valorização Esperada (%)',
            'nome_municipio': 'Município'
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'} if 'nome_municipio' in df_ranking.columns else {}
    )
    
    return fig, df_ranking


def criar_analise_correlacao(df):
    vars_correlacao = [
        'valorizacao_esperada_12m', 'pib_per_capita', 'populacao',
        'score_economico', 'ipca', 'selic'
    ]
    
    vars_disponiveis = [v for v in vars_correlacao if v in df.columns]
    
    df_corr = df[vars_disponiveis].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=df_corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlação")
    ))
    
    fig.update_layout(
        title='Matriz de Correlação - Variáveis Chave',
        height=500
    )
    
    return fig


def criar_simulador_cenarios(df):
    st.subheader("Simulador de Cenários Econômicos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ipca_sim = st.slider(
            "IPCA Anual (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Inflação anual esperada"
        )
    
    with col2:
        selic_sim = st.slider(
            "Taxa SELIC (%)",
            min_value=2.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help="Taxa básica de juros"
        )
    
    with col3:
        pib_sim = st.slider(
            "Crescimento PIB (%)",
            min_value=-5.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
            help="Crescimento do PIB"
        )
    
    impacto_ipca = (5.0 - ipca_sim) * 0.3
    impacto_selic = (10.0 - selic_sim) * 0.5
    impacto_pib = pib_sim * 0.8
    
    impacto_total = impacto_ipca + impacto_selic + impacto_pib
    
    st.markdown("### Impacto Previsto no Mercado Imobiliário")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Impacto IPCA",
            f"{impacto_ipca:+.2f}%",
            delta=f"{impacto_ipca:.2f}%"
        )
    
    with col2:
        st.metric(
            "Impacto SELIC",
            f"{impacto_selic:+.2f}%",
            delta=f"{impacto_selic:.2f}%"
        )
    
    with col3:
        st.metric(
            "Impacto PIB",
            f"{impacto_pib:+.2f}%",
            delta=f"{impacto_pib:.2f}%"
        )
    
    with col4:
        delta_color = "normal" if impacto_total > 0 else "inverse"
        st.metric(
            "Impacto Total",
            f"{impacto_total:+.2f}%",
            delta=f"{'Valorização' if impacto_total > 0 else 'Desvalorização'}",
            delta_color=delta_color
        )
    
    if impacto_total > 5:
        st.success("🚀 **Cenário Muito Favorável**: Forte potencial de valorização imobiliária")
    elif impacto_total > 2:
        st.info("📈 **Cenário Favorável**: Moderado potencial de valorização")
    elif impacto_total > -2:
        st.warning("⚖️ **Cenário Neutro**: Mercado estável")
    else:
        st.error("📉 **Cenário Desfavorável**: Possível desvalorização")


def mostrar_metricas_modelos(metrics):
    st.subheader("🤖 Performance dos Modelos de Machine Learning")
    
    df_metrics = pd.DataFrame(metrics).T
    df_metrics = df_metrics[['test_rmse', 'test_mae', 'test_mape', 'test_r2', 'directional_accuracy']]
    df_metrics.columns = ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'Acurácia Direcional (%)']
    df_metrics = df_metrics.round(4)
    
    best_model_idx = df_metrics['R²'].idxmax()
    
    st.dataframe(
        df_metrics.style.highlight_max(axis=0, subset=['R²', 'Acurácia Direcional (%)'], color='lightgreen')
                        .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen'),
        use_container_width=True
    )
    
    st.success(f"🏆 **Melhor Modelo**: {best_model_idx.upper()} com R² de {df_metrics.loc[best_model_idx, 'R²']:.4f}")
    
    fig = go.Figure()
    
    for metric in ['RMSE', 'MAE', 'MAPE (%)']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics.index,
            y=df_metrics[metric],
            text=df_metrics[metric].round(2),
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Comparação de Métricas dos Modelos',
        xaxis_title='Modelo',
        yaxis_title='Valor',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def criar_analise_temporal(df_indicadores):
    if df_indicadores.empty or 'data' not in df_indicadores.columns:
        st.warning("Dados de séries temporais não disponíveis")
        return
    
    df_indicadores['data'] = pd.to_datetime(df_indicadores['data'])
    df_indicadores = df_indicadores.sort_values('data')
    
    df_temp = df_indicadores.tail(36)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('IPCA (%)', 'SELIC (%)', 'Taxa de Câmbio', 'Desemprego (%)'),
        vertical_spacing=0.15
    )
    
    indicators = [
        ('ipca', 1, 1),
        ('selic', 1, 2),
        ('cambio', 2, 1),
        ('desemprego', 2, 2)
    ]
    
    for indicator, row, col in indicators:
        if indicator in df_temp.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_temp['data'],
                    y=df_temp[indicator],
                    mode='lines+markers',
                    name=indicator.upper(),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title_text="Evolução dos Indicadores Econômicos (36 meses)")
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Valor")
    
    st.plotly_chart(fig, use_container_width=True)


def gerar_palavras_chave_regiao(regiao_nome, df_ml):
    palavras_chave_imobiliarias = {
        'localizacao': ['centro', 'bem localizado', 'acessível', 'perto', 'região central', 
                        'bairro nobre', 'zona premium', 'proximidade', 'acesso fácil'],
        'preco': ['preço', 'valor', 'caro', 'barato', 'investimento', 'retorno', 
                  'custo-benefício', 'acessível', 'especulação'],
        'condicao': ['novo', 'reformado', 'moderno', 'conservado', 'bem conservado', 
                     'oportunidade', 'excelente estado', 'precisa reforma', 'reabilitação'],
        'infraestrutura': ['metrô', 'transporte', 'comercio', 'escolas', 'hospital', 
                          'supermercado', 'shopping', 'infraestrutura', 'amenidades', 'lazer'],
        'investimento': ['valorização', 'crescimento', 'potencial', 'oportunidade', 
                        'rentabilidade', 'bom investimento', 'lucro', 'apreciação'],
        'demanda': ['procura', 'busca', 'interesse', 'aquecido', 'demanda alta', 
                   'competitivo', 'acirrado', 'venda rápida', 'disputado']
    }
    
    df_regiao = pd.DataFrame()
    
    if 'regiao' in df_ml.columns:
        regiao_upper = regiao_nome.upper().strip()
        regioes_unicas = df_ml['regiao'].dropna().unique()
        
        for regiao_unica in regioes_unicas:
            if regiao_upper == str(regiao_unica).upper():
                df_regiao = df_ml[df_ml['regiao'].str.upper() == regiao_upper]
                break
        
        if df_regiao.empty:
            df_regiao = df_ml[df_ml['regiao'].str.contains(regiao_nome, case=False, na=False)]
    
    if df_regiao.empty and 'nome_municipio' in df_ml.columns:
        df_regiao = df_ml[df_ml['nome_municipio'].str.contains(regiao_nome, case=False, na=False)]
    
    if df_regiao.empty:
        return None
    
    stats = {
        'num_municipios': len(df_regiao),
        'valorizacao_media': df_regiao['valorizacao_esperada_12m'].mean() if 'valorizacao_esperada_12m' in df_regiao.columns else 0,
        'pib_media': df_regiao['pib_per_capita'].mean() if 'pib_per_capita' in df_regiao.columns else 0,
        'populacao_total': df_regiao['populacao'].sum() if 'populacao' in df_regiao.columns else 0,
        'score_economico_medio': df_regiao['score_economico'].mean() if 'score_economico' in df_regiao.columns else 0
    }
    
    palavras_geradas = []
    
    palavras_geradas.append('bem localizado')
    palavras_geradas.append('região atrativa')
    
    if stats['valorizacao_media'] > 20:
        palavras_geradas.extend(['potencial de valorização', 'investimento promissor', 'oportunidade de crescimento'])
    elif stats['valorizacao_media'] > 10:
        palavras_geradas.extend(['boa oportunidade', 'crescimento esperado', 'mercado aquecido'])
    else:
        palavras_geradas.extend(['mercado estável', 'seguro', 'consolidado'])
    
    if stats['score_economico_medio'] > 0.6:
        palavras_geradas.extend(['infraestrutura moderna', 'economia dinâmica', 'desenvolvimento'])
    else:
        palavras_geradas.extend(['em desenvolvimento', 'potencial de crescimento'])
    
    if stats['populacao_total'] > 1000000:
        palavras_geradas.extend(['alta demanda', 'mercado aquecido', 'grande interesse'])
    elif stats['populacao_total'] > 500000:
        palavras_geradas.extend(['demanda consistente', 'público alvo amplo'])
    
    region_display = regiao_nome.upper()
    
    return {
        'region': region_display,
        'statistics': stats,
        'palavras_chave': palavras_geradas,
        'palavras_por_categoria': palavras_chave_imobiliarias
    }


def main():
    st.markdown("### Análise Preditiva com Machine Learning")
    
    with st.spinner('Carregando dados...'):
        df_ml, df_indicadores, metrics = carregar_dados()
    
    if df_ml.empty:
        st.error("Não foi possível carregar os dados. Execute os scripts de coleta e processamento primeiro.")
        st.info("Execute os seguintes comandos:\n```\npython src/data_collection.py\npython src/data_processing.py\npython src/ml_pipeline.py\n```")
        return
    
    st.sidebar.title("📊 Navegação")
    pagina = st.sidebar.radio(
        "Escolha a análise:",
        [
            "🏠 Visão Geral",
            "🗺️ Mapa de Valorização",
            "🏆 Ranking de Regiões",
            "🎮 Simulador de Cenários",
            "🔗 Correlações",
            "💬 Análise PLN"
        ]
    )
    
    if pagina == "🏠 Visão Geral":
        st.header("Visão Geral do Projeto")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Municípios Analisados",
                f"{len(df_ml):,}",
                help="Total de municípios no dataset"
            )
        
        with col2:
            if 'valorizacao_esperada_12m' in df_ml.columns:
                val_media = df_ml['valorizacao_esperada_12m'].mean()
                st.metric(
                    "Valorização Média",
                    f"{val_media:.2f}%",
                    delta=f"{val_media:.2f}%",
                    help="Valorização média esperada em 12 meses"
                )
            else:
                st.metric("Valorização Média", "N/A")
        

        
        with col4:
            if metrics:
                melhor_r2 = max([m.get('test_r2', 0) for m in metrics.values()])
                st.metric(
                    "R² Melhor Modelo",
                    f"{melhor_r2:.4f}",
                    help="Coeficiente de determinação do melhor modelo"
                )
        
        st.markdown("---")
        
        st.subheader("Estatísticas por Região")
        
        agg_dict = {}
        if 'valorizacao_esperada_12m' in df_ml.columns:
            agg_dict['valorizacao_esperada_12m'] = 'mean'
        if 'pib_per_capita' in df_ml.columns:
            agg_dict['pib_per_capita'] = 'mean'
        if 'populacao' in df_ml.columns:
            agg_dict['populacao'] = 'sum'
        
        col_contagem = None
        for col in ['codigo_municipio', 'codigo_municipio_x', 'nome_municipio']:
            if col in df_ml.columns:
                col_contagem = col
                break
        
        if col_contagem:
            agg_dict[col_contagem] = 'count'
        
        if 'regiao' in df_ml.columns and len(agg_dict) > 0:
            df_regiao = df_ml.groupby('regiao').agg(agg_dict).round(2)
            
            rename_dict = {}
            if 'valorizacao_esperada_12m' in df_regiao.columns:
                rename_dict['valorizacao_esperada_12m'] = 'Valorização Média (%)'
            if 'pib_per_capita' in df_regiao.columns:
                rename_dict['pib_per_capita'] = 'PIB per Capita (R$)'
            if 'populacao' in df_regiao.columns:
                rename_dict['populacao'] = 'População Total'
            if col_contagem in df_regiao.columns:
                rename_dict[col_contagem] = 'Nº Municípios'
            
            df_regiao = df_regiao.rename(columns=rename_dict)
            st.dataframe(df_regiao, use_container_width=True)
        else:
            st.info("Dados por região não disponíveis")
        
        fig = px.histogram(
            df_ml,
            x='valorizacao_esperada_12m',
            nbins=50,
            title='Distribuição da Valorização Esperada',
            labels={'valorizacao_esperada_12m': 'Valorização Esperada (%)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=15, line_dash="dash", line_color="green", 
                     annotation_text="Meta: 15%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif pagina == "🗺️ Mapa de Valorização":
        st.header("Mapa Interativo de Valorização")
        fig_mapa = criar_mapa_valorizacao(df_ml)
        st.plotly_chart(fig_mapa, use_container_width=True)
        
        
    elif pagina == "🏆 Ranking de Regiões":
        st.header("Ranking das Regiões Mais Promissoras")
        
        top_n = st.slider("Número de municípios no ranking:", 5, 50, 12)
        fig_ranking, df_ranking = criar_ranking_regioes(df_ml, top_n)
        
        if fig_ranking is not None:
            st.plotly_chart(fig_ranking, use_container_width=True)
            
            st.subheader("Detalhes do Ranking")
            
            colunas_ranking = ['posicao']
            for col in ['nome_municipio', 'uf', 'regiao', 'valorizacao_esperada_12m', 'pib_per_capita', 'populacao']:
                if col in df_ranking.columns:
                    colunas_ranking.append(col)
            
            if len(colunas_ranking) > 1:
                st.dataframe(
                    df_ranking[colunas_ranking].round(2),
                    use_container_width=True
                )
            else:
                st.info("Nenhuma coluna disponível para exibição")
    
    elif pagina == "🎮 Simulador de Cenários":
        st.header("Simulador de Cenários Econômicos")
        criar_simulador_cenarios(df_ml)
    
    
    elif pagina == "🔗 Correlações":
        st.header("Análise de Correlações")
        fig_corr = criar_analise_correlacao(df_ml)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        ### 📊 Interpretação:
        - **Azul**: Correlação positiva (quando uma variável aumenta, a outra também)
        - **Vermelho**: Correlação negativa (quando uma variável aumenta, a outra diminui)
        - **Branco**: Correlação fraca ou inexistente
        """)
    
    elif pagina == "💬 Análise PLN":
        st.header("Análise de Linguagem Natural - Palavras-chave Imobiliárias")
        
        st.markdown("""
        Digite o nome de uma região ou município para extrair as principais palavras-chave 
        imobiliárias associadas àquela localidade, baseado na análise de dados econômicos e 
        indicadores do mercado imobiliário.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            regiao_input = st.text_input(
                "Digite o nome da região ou município:",
                placeholder="Ex: Santos, Sao Paulo, Rio de Janeiro...",
                label_visibility="collapsed"
            )
        
        with col2:
            buscar = st.button("Buscar", use_container_width=True)
        
        if buscar and regiao_input:
            with st.spinner(f"Analisando PLN para {regiao_input}..."):
                resultado_pln = gerar_palavras_chave_regiao(regiao_input, df_ml)
            
            if resultado_pln is None:
                st.warning(f"❌ Nenhuma região encontrada com o nome '{regiao_input}'. "
                          f"Tente outro nome ou verifique a grafia.")
            else:
                stats = resultado_pln['statistics']
                palavras = resultado_pln['palavras_chave']
                
                st.markdown(f"### Análise PLN: {resultado_pln['region'].upper()}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Valorização Média",
                        f"{stats['valorizacao_media']:.2f}%",
                        delta=f"{stats['valorizacao_media']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Municípios",
                        f"{stats['num_municipios']}",
                        delta=f"{stats['num_municipios']} analisados"
                    )
                
                with col3:
                    st.metric(
                        "Score Econômico",
                        f"{stats['score_economico_medio']:.2f}",
                        delta=f"{stats['score_economico_medio']:.2f}"
                    )
                
                st.markdown("### Palavras-chave Imobiliárias Extraídas")
                
                col_count = 3
                cols = st.columns(col_count)
                
                for idx, palavra in enumerate(palavras):
                    with cols[idx % col_count]:
                        if stats['valorizacao_media'] > 15:
                            badge_color = "🟢"
                        elif stats['valorizacao_media'] > 5:
                            badge_color = "🟡"
                        else:
                            badge_color = "🔵"
                        
                        st.write(f"{badge_color} **{palavra}**")
    
    
    st.markdown("---")


if __name__ == "__main__":
    main()
