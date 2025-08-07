# passo_0_configuracao_e_ferramentas.py
# -*- coding: utf-8 -*-
"""
Arquivo de configuração central para a solução de comércio exterior.
Este módulo contém as configurações, imports, chaves de API, e
definições de funções (ferramentas) para os outros scripts.
"""

# ===============================================================================
# IMPORTS E CONFIGURAÇÕES
# ===============================================================================
import os
import requests
import pandas as pd
import gc
import zipfile
import io
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# ===============================================================================
# CARREGAMENTO DE CHAVES DE API
# ===============================================================================
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
nvidia_key = os.getenv("NVIDIA_API_KEY")

if not all([groq_key, nvidia_key]):
    raise ValueError("Chaves de API 'GROQ_API_KEY' e 'NVIDIA_API_KEY' não encontradas no arquivo .env")

# ===============================================================================
# CONFIGURAÇÃO GLOBAL DE MODELOS
# ===============================================================================
llm_groq = Groq(model="llama-3.1-8b-instant", api_key=groq_key)
Settings.llm = llm_groq
Settings.embed_model = NVIDIAEmbedding(
    model="nv-embed-qa-e4", api_key=nvidia_key, truncate="END"
)
gc.set_threshold(700, 10, 10)

# Variável global para armazenar o DataFrame dos dados
df_comex = None

# ===============================================================================
# DEFINIÇÕES DE FUNÇÕES (FERRAMENTAS)
# ===============================================================================
def obter_dados_comex(ano: str, mes: str, tipo_operacao: str) -> str:
    """
    Baixa e carrega os dados de comércio exterior (EXPORTACAO ou IMPORTACAO) para um
    determinado ano e mês.

    Args:
        ano (str): O ano dos dados, por exemplo, '2024'.
        mes (str): O mês dos dados, por exemplo, 'janeiro'.
        tipo_operacao (str): 'EXPORTACAO' ou 'IMPORTACAO'.

    Returns:
        str: Uma mensagem de sucesso ou erro.
    """
    global df_comex
    tipo_operacao = tipo_operacao.upper()
    mes_num = {
        'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
        'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
        'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'
    }.get(mes.lower())

    if not mes_num:
        return "Mês inválido. Por favor, use o nome completo do mês em português."
    
    if tipo_operacao not in ["EXPORTACAO", "IMPORTACAO"]:
        return "Tipo de operação inválido. Use 'EXPORTACAO' ou 'IMPORTACAO'."

    url = f"https://www.gov.br/produtividade/servicos/comex-stat/niveis/uf-mun/dados-abertos/{ano}/{ano}{mes_num}_{tipo_operacao}.zip"
    
    try:
        print(f"Baixando dados de {tipo_operacao} para {mes}/{ano}...")
        response = requests.get(url, timeout=60)  # Aumentei o timeout para 60 segundos
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_file_name = [f for f in zip_file.namelist() if f.endswith('.csv')][0]
            with zip_file.open(csv_file_name) as csv_file:
                df_comex = pd.read_csv(csv_file, sep=';', encoding='iso-8859-1')
        
        print(f"Dados de {tipo_operacao} carregados. Total de linhas: {len(df_comex)}")
        return f"Dados de {tipo_operacao} para {mes}/{ano} carregados com sucesso. Agora você pode fazer perguntas sobre eles."

    except requests.exceptions.HTTPError as e:
        return f"Erro HTTP ao baixar os dados: {e}. Verifique se a URL e o período estão corretos. URL tentada: {url}"
    except Exception as e:
        return f"Ocorreu um erro ao processar os dados: {e}"

def resumo_dados_comex(consulta: str) -> str:
    """
    Executa uma consulta específica nos dados de comércio exterior carregados.

    Args:
        consulta (str): Uma pergunta em linguagem natural sobre os dados.
    
    Returns:
        str: A resposta à consulta ou uma mensagem de erro.
    """
    global df_comex
    if df_comex is None:
        return "Nenhum dado de comércio exterior foi carregado. Por favor, use a ferramenta 'obter_dados_comex' primeiro."
    
    consulta_lower = consulta.lower()
    
    if "média do peso líquido" in consulta_lower:
        if 'KG_LIQUIDO' in df_comex.columns:
            media = df_comex['KG_LIQUIDO'].mean()
            return f"A média do peso líquido dos dados carregados é de {media:.2f} kg."
        else:
            return "A coluna 'KG_LIQUIDO' não foi encontrada nos dados."
    
    elif "principais estados" in consulta_lower:
        if 'SG_UF' in df_comex.columns:
            top_estados = df_comex['SG_UF'].value_counts().head(5)
            return f"Os 5 principais estados por número de operações são:\n{top_estados.to_string()}"
        else:
            return "A coluna 'SG_UF' não foi encontrada nos dados."
            
    else:
        return "Não foi possível processar a sua consulta. Tente perguntas sobre 'média do peso líquido' ou 'principais estados'."

def limpar_dados_comex() -> str:
    """
    Libera os dados de comércio exterior da memória para economizar recursos computacionais.
    """
    global df_comex
    if df_comex is None:
        return "Não há dados para limpar na memória."
    
    df_comex = None
    gc.collect()
    print("Dados de comércio exterior foram limpos da memória.")
    return "Os dados foram removidos da memória com sucesso."

print("Módulo de configuração de Comércio Exterior carregado.")
