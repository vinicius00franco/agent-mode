# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
import gc
from dotenv import load_dotenv

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
def obter_dados_comex(ano: str, mes: str, tipo_operacao: str) -> str:
    """
    Baixa e carrega os dados de comércio exterior (EXP ou IMP) para um
    determinado ano e mês.
    """
    global df_comex
    tipo_operacao = tipo_operacao.upper()
    meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    mes_num = meses.get(mes.lower())

    if not mes_num:
        return "Mês inválido."
    if tipo_operacao not in ["EXP", "IMP"]:
        return "Tipo de operação inválido. Use 'EXP' ou 'IMP'."

    url = f"https://balanca.economia.gov.br/balanca/bd/comexstat-bd/mun/{tipo_operacao}_{ano}_MUN.csv"
    try:
        df_anual = pd.read_csv(url, sep=';', encoding='iso-8859-1')
        df_comex = df_anual[df_anual['CO_MES'] == mes_num].copy()
        if df_comex.empty:
            return f"Nenhum dado de {tipo_operacao} encontrado para {mes}/{ano}."
        return f"Dados de {tipo_operacao} para {mes}/{ano} carregados com sucesso."
    except Exception as e:
        return f"Ocorreu um erro ao processar os dados: {e}"

def resumo_dados_comex(consulta: str) -> str:
    """
    Executa uma consulta específica (e rígida) nos dados carregados.
    """
    global df_comex
    if df_comex is None:
        return "Nenhum dado carregado. Use 'obter_dados_comex' primeiro."
    
    consulta_lower = consulta.lower()
    if "média do peso líquido" in consulta_lower:
        media = df_comex['KG_LIQUIDO'].mean()
        return f"A média do peso líquido é de {media:.2f} kg."
    elif "principais estados" in consulta_lower:
        # Nota: Esta coluna estava incorreta no código original, causando erros.
        top_estados = df_comex['SG_UF_NCM'].value_counts().head(5)
        return f"Os 5 principais estados são:\n{top_estados.to_string()}"
    else:
        return "Consulta não reconhecida."

def limpar_dados_comex() -> str:
    """Libera os dados da memória."""
    global df_comex
    df_comex = None
    gc.collect()
    return "Os dados foram removidos da memória."
