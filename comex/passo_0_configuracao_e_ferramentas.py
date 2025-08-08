# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
import gc
from dotenv import load_dotenv
import json

from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.nvidia import NVIDIAEmbedding

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
nvidia_key = os.getenv("NVIDIA_API_KEY")

llm_groq = Groq(model="llama-3.1-8b-instant", api_key=groq_key)
Settings.llm = llm_groq
Settings.embed_model = NVIDIAEmbedding(model="nv-embed-qa-e4", api_key=nvidia_key, truncate="END")

gc.set_threshold(700, 10, 10)

df_comex = None

CACHE_DIR = "cache_comex"
os.makedirs(CACHE_DIR, exist_ok=True)
def obter_dados_comex_por_ano(ano: str, pergunta_original: str, usar_cache: bool = True) -> str:
    """
    Baixa e carrega dados de um ANO INTEIRO, com cache. Infere EXP/IMP pela pergunta.
    """
    global df_comex
    print("\n--- obter_dados_comex_por_ano ---")

    pergunta_lower = pergunta_original.lower()
    palavras_exportacao = ['exportado', 'exportação', 'vendeu', 'enviou', 'exportadores']
    palavras_importacao = ['importado', 'importação', 'comprou', 'recebeu', 'importadores']
    tipo_operacao = "EXP" if any(p in pergunta_lower for p in palavras_exportacao) else "IMP" if any(p in pergunta_lower for p in palavras_importacao) else None
    
    if not tipo_operacao: return "Não consegui identificar se a pergunta se refere a exportação ou importação."
    print(f"Operação inferida: {tipo_operacao}")

    nome_arquivo_cache = f"{tipo_operacao}_{ano}.parquet"
    caminho_arquivo_cache = os.path.join(CACHE_DIR, nome_arquivo_cache)

    if usar_cache and os.path.exists(caminho_arquivo_cache):
        print(f"CACHE: Carregando de '{caminho_arquivo_cache}'")
        df_comex = pd.read_parquet(caminho_arquivo_cache)
        return f"Dados anuais de {tipo_operacao} para {ano} carregados do cache."

    print(f"DOWNLOAD: Baixando dados de {tipo_operacao} para {ano}")
    url = f"https://balanca.economia.gov.br/balanca/bd/comexstat-bd/mun/{tipo_operacao}_{ano}_MUN.csv"
    try:
        df_comex = pd.read_csv(url, sep=';', encoding='iso-8859-1')
        print(f"CACHE: Salvando em '{caminho_arquivo_cache}'")
        df_comex.to_parquet(caminho_arquivo_cache)
        return f"Dados anuais de {tipo_operacao} para {ano} baixados e salvos no cache."
    except Exception as e:
        return f"Erro no download: {e}"

def analisar_principais_entidades(coluna_entidade: str, coluna_metrica: str = 'VL_FOB', top_n: int = 5, mes: str = None) -> str:
    """
    Analisa os dados anuais carregados. Pode filtrar por um mês específico.
    """
    global df_comex
    if df_comex is None: return "Erro: Nenhum dado carregado."

    df_para_analise = df_comex
    if mes:
        meses = {'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12}
        mes_num = meses.get(mes.lower())
        if not mes_num: return f"Erro: Mês '{mes}' inválido."
        df_para_analise = df_comex[df_comex['CO_MES'] == mes_num]
    
    if df_para_analise.empty: return f"Nenhum dado para o mês de {mes}."

    analise_df = df_para_analise.groupby(coluna_entidade)[coluna_metrica].sum().sort_values(ascending=False).head(top_n)
    return json.dumps(analise_df.reset_index().to_dict(orient='records'), ensure_ascii=False)

def obter_estatisticas_gerais(coluna_metrica: str = 'VL_FOB', mes: str = None) -> str:
    """
    Calcula estatísticas dos dados anuais. Pode filtrar por mês.
    """
    global df_comex
    if df_comex is None: return "Erro: Nenhum dado carregado."
    
    df_para_analise = df_comex
    if mes:
        meses = {'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12}
        mes_num = meses.get(mes.lower())
        if not mes_num: return f"Erro: Mês '{mes}' inválido."
        df_para_analise = df_comex[df_comex['CO_MES'] == mes_num]

    resultado = {
        f"total_{coluna_metrica}": df_para_analise[coluna_metrica].sum(),
        f"media_{coluna_metrica}": df_para_analise[coluna_metrica].mean(),
        "numero_operacoes": len(df_para_analise)
    }
    return json.dumps(resultado)

def limpar_dados_comex() -> str:
    """Libera os dados da memória."""
    global df_comex
    df_comex = None
    gc.collect()
    return "Os dados foram removidos da memória."
