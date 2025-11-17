import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class ImobiliarioPLN:
    
    def __init__(self):
        self.palavras_positivas = {
            'valorização', 'crescimento', 'expansão', 'desenvolvimento',
            'investimento', 'alta', 'aumento', 'melhoria', 'modernização',
            'potencial', 'oportunidade', 'promissor', 'favorável', 'boom',
            'infraestrutura', 'revitalização', 'progresso', 'inovação'
        }
        
        self.palavras_negativas = {
            'desvalorização', 'queda', 'redução', 'crise', 'recessão',
            'estagnação', 'baixa', 'declínio', 'problemas', 'déficit',
            'retração', 'colapso', 'dificuldade', 'deterioração',
            'abandono', 'depreciação', 'desaceleração'
        }
        
        self.indicadores_economicos = {
            'ipca': ['inflação', 'ipca', 'índice de preços'],
            'selic': ['selic', 'juros', 'taxa básica'],
            'pib': ['pib', 'produto interno', 'economia', 'crescimento econômico'],
            'desemprego': ['desemprego', 'emprego', 'mercado de trabalho'],
            'renda': ['renda', 'salário', 'poder de compra'],
            'credito': ['crédito', 'financiamento', 'empréstimo']
        }
    
    def preprocessar_texto(self, texto: str) -> str:
        if pd.isna(texto):
            return ""
        texto = str(texto).lower()
        texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    def classificar_sentimento(self, texto: str) -> dict:
        texto_proc = self.preprocessar_texto(texto)
        palavras = set(texto_proc.split())
        positivas = len(palavras.intersection(self.palavras_positivas))
        negativas = len(palavras.intersection(self.palavras_negativas))
        total = positivas + negativas
        
        if total == 0:
            sentimento = 'neutro'
            confianca = 0.5
        elif positivas > negativas:
            sentimento = 'positivo'
            confianca = positivas / total
        elif negatives > positives:
            sentimento = 'negativo'
            confianca = negativas / total
        else:
            sentimento = 'neutro'
            confianca = 0.5
        
        return {
            'sentimento': sentimento,
            'confianca': confianca,
            'palavras_positivas': positivas,
            'palavras_negativas': negativas,
            'score': (positivas - negativas) / max(total, 1)
        }
    
    def extrair_indicadores(self, texto: str) -> list:
        texto_proc = self.preprocessar_texto(texto)
        indicadores_encontrados = []
        
        for indicador, palavras_chave in self.indicadores_economicos.items():
            for palavra in palavras_chave:
                if palavra in texto_proc:
                    indicadores_encontrados.append(indicador)
                    break
        
        return list(set(indicadores_encontrados))
    
    def extrair_entidades_valores(self, texto: str) -> dict:
        percentuais = re.findall(r'(\d+(?:,\d+)?)\s*%', texto)
        percentuais = [float(p.replace(',', '.')) for p in percentuais]
        
        valores_r = re.findall(r'R\$\s*(\d+(?:\.\d{3})*(?:,\d{2})?)', texto)
        valores_r = [v.replace('.', '').replace(',', '.') for v in valores_r]
        valores_r = [float(v) for v in valores_r if v]
        
        anos = re.findall(r'\b(20\d{2}|19\d{2})\b', texto)
        anos = [int(ano) for ano in anos]
        
        return {
            'percentuais': percentuais,
            'valores_monetarios': valores_r,
            'anos': anos
        }
    
    def extrair_localidades(self, texto: str, df_municipios: pd.DataFrame = None) -> list:
        texto_proc = self.preprocessar_texto(texto)
        localidades = []
        
        estados = {
            'acre', 'alagoas', 'amapá', 'amazonas', 'bahia', 'ceará',
            'distrito federal', 'espírito santo', 'goiás', 'maranhão',
            'mato grosso', 'mato grosso do sul', 'minas gerais', 'pará',
            'paraíba', 'paraná', 'pernambuco', 'piauí', 'rio de janeiro',
            'rio grande do norte', 'rio grande do sul', 'rondônia',
            'roraima', 'santa catarina', 'são paulo', 'sergipe', 'tocantins'
        }
        
        siglas_estados = {
            'ac', 'al', 'ap', 'am', 'ba', 'ce', 'df', 'es', 'go', 'ma',
            'mt', 'ms', 'mg', 'pa', 'pb', 'pr', 'pe', 'pi', 'rj', 'rn',
            'rs', 'ro', 'rr', 'sc', 'sp', 'se', 'to'
        }
        
        regioes = {'norte', 'nordeste', 'centro-oeste', 'sudeste', 'sul'}
        
        for estado in estados:
            if estado in texto_proc:
                localidades.append(estado.title())
        
        palavras = texto_proc.split()
        for palavra in palavras:
            if palavra in siglas_estados:
                localidades.append(palavra.upper())
        
        for regiao in regioes:
            if regiao in texto_proc:
                localidades.append(regiao.title())
        
        if df_municipios is not None and 'nome_municipio' in df_municipios.columns:
            municipios = set(df_municipios['nome_municipio'].str.lower())
            for municipio in municipios:
                if municipio in texto_proc:
                    localidades.append(municipio.title())
        
        return list(set(localidades))
    
    def gerar_resumo_analise(self, texto: str, df_municipios: pd.DataFrame = None) -> dict:
        sentimento = self.classificar_sentimento(texto)
        indicadores = self.extrair_indicadores(texto)
        entidades = self.extrair_entidades_valores(texto)
        localidades = self.extrair_localidades(texto, df_municipios)
        
        if 'pib' in indicadores or 'credito' in indicadores:
            tematica = 'Desenvolvimento Econômico'
        elif 'selic' in indicadores or 'ipca' in indicadores:
            tematica = 'Política Monetária'
        elif 'desemprego' in indicadores or 'renda' in indicadores:
            tematica = 'Mercado de Trabalho'
        elif localidades:
            tematica = 'Análise Regional'
        else:
            tematica = 'Mercado Imobiliário Geral'
        
        return {
            'tematica': tematica,
            'sentimento': sentimento['sentimento'],
            'confianca_sentimento': sentimento['confianca'],
            'score_sentimento': sentimento['score'],
            'indicadores_chave': indicadores,
            'localidades_mencionadas': localidades,
            'percentuais_extraidos': entidades['percentuais'],
            'valores_monetarios': entidades['valores_monetarios'],
            'anos_mencionados': entidades['anos'],
            'texto_tamanho': len(texto.split())
        }
    
    def analisar_dataset_textual(self, textos: list, df_municipios: pd.DataFrame = None) -> pd.DataFrame:
        resultados = []
        
        for texto in textos:
            analise = self.gerar_resumo_analise(texto, df_municipios)
            resultados.append(analise)
        
        return pd.DataFrame(resultados)
    
    def gerar_relatorio_pln(self, df_analise: pd.DataFrame) -> str:
        relatorio = []
        relatorio.append("=" * 70)
        relatorio.append("RELATÓRIO DE ANÁLISE DE PROCESSAMENTO DE LINGUAGEM NATURAL")
        relatorio.append("Mercado Imobiliário Brasileiro")
        relatorio.append("=" * 70)
        relatorio.append("")
        
        relatorio.append("📊 ESTATÍSTICAS GERAIS")
        relatorio.append("-" * 70)
        relatorio.append(f"Total de textos analisados: {len(df_analise)}")
        relatorio.append("")
        
        relatorio.append("😊 ANÁLISE DE SENTIMENTO")
        relatorio.append("-" * 70)
        sent_dist = df_analise['sentimento'].value_counts()
        for sent, count in sent_dist.items():
            pct = (count / len(df_analise)) * 100
            relatorio.append(f"  {sent.capitalize()}: {count} ({pct:.1f}%)")
        
        score_medio = df_analise['score_sentimento'].mean()
        relatorio.append(f"\nScore médio de sentimento: {score_medio:.3f}")
        
        if score_medio > 0.2:
            interpretacao = "predominantemente POSITIVO"
        elif score_medio < -0.2:
            interpretacao = "predominantemente NEGATIVO"
            interpretacao = "NEUTRO"
        relatorio.append(f"Interpretação: Mercado {interpretacao}")
        relatorio.append("")
        
        relatorio.append("🎯 TEMÁTICAS IDENTIFICADAS")
        relatorio.append("-" * 70)
        tematicas = df_analise['tematica'].value_counts()
        for tema, count in tematicas.items():
            pct = (count / len(df_analise)) * 100
            relatorio.append(f"  {tema}: {count} ({pct:.1f}%)")
        relatorio.append("")
        
        relatorio.append("📈 INDICADORES ECONÔMICOS MAIS MENCIONADOS")
        relatorio.append("-" * 70)
        todos_indicadores = []
        for ind_list in df_analise['indicadores_chave']:
            todos_indicadores.extend(ind_list)
        
        if todos_indicadores:
            indicadores_freq = Counter(todos_indicadores)
            for indicador, freq in indicadores_freq.most_common(5):
                relatorio.append(f"  {indicador.upper()}: {freq} menções")
        else:
            relatorio.append("  Nenhum indicador específico identificado")
        relatorio.append("")
        
        relatorio.append("🗺️ LOCALIDADES MAIS MENCIONADAS")
        relatorio.append("-" * 70)
        todas_localidades = []
        for loc_list in df_analise['localidades_mencionadas']:
            todas_localidades.extend(loc_list)
        
        if todas_localidades:
            localidades_freq = Counter(todas_localidades)
            for localidade, freq in localidades_freq.most_common(10):
                relatorio.append(f"  {localidade}: {freq} menções")
        else:
            relatorio.append("  Nenhuma localidade específica identificada")
        relatorio.append("")
        
        relatorio.append("💡 PRINCIPAIS INSIGHTS")
        relatorio.append("-" * 70)
        
        if score_medio > 0:
            relatorio.append("1. O sentimento geral em relação ao mercado imobiliário é POSITIVO,")
            relatorio.append("   indicando expectativas favoráveis de valorização.")
        else:
            relatorio.append("1. O sentimento geral em relação ao mercado imobiliário é CAUTELOSO,")
            relatorio.append("   sugerindo incertezas ou preocupações no setor.")
        
        tema_principal = tematicas.index[0]
        relatorio.append(f"\n2. A temática predominante nas análises é '{tema_principal}',")
        relatorio.append("   evidenciando o foco principal das discussões do mercado.")
        
        if todos_indicadores:
            ind_top = Counter(todos_indicadores).most_common(1)[0][0]
            relatorio.append(f"\n3. O indicador econômico '{ind_top.upper()}' é o mais mencionado,")
            relatorio.append("   demonstrando sua relevância nas análises do setor imobiliário.")
        
        relatorio.append("")
        relatorio.append("=" * 70)
        
        return "\n".join(relatorio)


def exemplo_uso():
    pln = ImobiliarioPLN()
    
    textos_exemplo = [
        """
        O mercado imobiliário de São Paulo apresenta forte crescimento em 2025,
        com valorização média de 15% nos principais bairros. A queda da taxa SELIC
        para 10% ao ano estimula o crédito imobiliário e aumenta a demanda.
        """,
        """
        Região Nordeste registra desaceleração no setor imobiliário devido ao
        aumento do desemprego e à inflação elevada. IPCA acumulado de 8% ao ano
        pressiona o poder de compra das famílias.
        """,
        """
        Investimentos em infraestrutura no Rio de Janeiro impulsionam valorização
        imobiliária. Novos projetos de mobilidade urbana e revitalização de áreas
        centrais atraem investidores e promovem desenvolvimento econômico.
        """,
        """
        Mercado imobiliário brasileiro mostra resiliência apesar dos desafios
        econômicos. PIB cresce 2.5% e renda média aumenta, criando oportunidades
        para o setor em diversas regiões do país.
        """
    ]
    
    df_analise = pln.analisar_dataset_textual(textos_exemplo)
    relatorio = pln.gerar_relatorio_pln(df_analise)
    print("\n" + relatorio)
    
    analise_detalhada = pln.gerar_resumo_analise(textos_exemplo[0])
    for chave, valor in analise_detalhada.items():
        print(f"{chave}: {valor}")


def main():
    exemplo_uso()


if __name__ == "__main__":
    main()
