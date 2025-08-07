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
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
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
    print("\n================ INÍCIO obter_dados_comex ================")
    print(f"Parâmetros recebidos: ano={ano}, mes={mes}, tipo_operacao={tipo_operacao}")
    tipo_operacao = tipo_operacao.upper()
    meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    mes_num = meses.get(mes.lower())

    if not mes_num:
        print("[ERRO] Mês inválido!")
        print("================ FIM obter_dados_comex ================\n")
        return "Mês inválido. Por favor, use o nome completo do mês em português."
    if tipo_operacao not in ["EXPORTACAO", "IMPORTACAO"]:
        print("[ERRO] Tipo de operação inválido!")
        print("================ FIM obter_dados_comex ================\n")
        return "Tipo de operação inválido. Use 'EXPORTACAO' ou 'IMPORTACAO'."

    url = f"https://balanca.economia.gov.br/balanca/bd/comexstat-bd/mun/{tipo_operacao}_{ano}_MUN.csv"
    try:
        print(f"[DOWNLOAD] Baixando dados anuais de {tipo_operacao} para {ano}...")
        df_anual = pd.read_csv(url, sep=';', encoding='iso-8859-1')
        print(f"[OK] Dados anuais de {tipo_operacao} carregados. Total de linhas: {len(df_anual)}")
        df_comex = df_anual[df_anual['CO_MES'] == mes_num].copy()
        if df_comex.empty:
            print(f"[AVISO] Nenhum dado encontrado para o mês de {mes}/{ano}!")
            print("================ FIM obter_dados_comex ================\n")
            return f"Nenhum dado de {tipo_operacao} encontrado para o mês de {mes}/{ano}. Verifique se a combinação de mês e ano possui dados."
        print(f"[OK] Dados filtrados para o mês de {mes}/{ano}. Total de linhas: {len(df_comex)}")
        print("================ FIM obter_dados_comex ================\n")
        return f"Dados de {tipo_operacao} para {mes}/{ano} carregados com sucesso. Agora você pode fazer perguntas sobre eles."
    except requests.exceptions.HTTPError as e:
        print(f"[ERRO HTTP] {e}")
        if e.response is not None and e.response.status_code == 404:
            print("================ FIM obter_dados_comex ================\n")
            return f"Erro: O arquivo de dados para {tipo_operacao} em {ano} não foi encontrado. A URL tentada foi: {url}. Por favor, verifique se os dados para este período estão disponíveis."
        print("================ FIM obter_dados_comex ================\n")
        return f"Erro HTTP ao baixar os dados: {e}. URL tentada: {url}"
    except Exception as e:
        print(f"[ERRO EXCEÇÃO] {e}")
        print("================ FIM obter_dados_comex ================\n")
        return f"Ocorreu um erro ao processar os dados: {e}"

def resumo_dados_comex(consulta: str) -> str:
    """
    Executa uma consulta específica nos dados de comércio exterior carregados.

    Args:
        consulta (str): Uma pergunta em linguagem natural sobre os dados.
    
    Returns:
        str: A resposta à consulta ou uma mensagem de erro.
    """
    print("\n================ INÍCIO resumo_dados_comex ================")
    global df_comex
    if df_comex is None:
        print("[ERRO] Nenhum dado carregado!")
        print("================ FIM resumo_dados_comex ================\n")
        return "Nenhum dado de comércio exterior foi carregado. Por favor, use a ferramenta 'obter_dados_comex' primeiro."
    consulta_lower = consulta.lower()
    print(f"Consulta recebida: {consulta}")
    if "média do peso líquido" in consulta_lower:
        if 'KG_LIQUIDO' in df_comex.columns:
            media = df_comex['KG_LIQUIDO'].mean()
            print(f"[OK] Média do peso líquido: {media:.2f} kg")
            print("================ FIM resumo_dados_comex ================\n")
            return f"A média do peso líquido dos dados carregados é de {media:.2f} kg."
        else:
            print("[ERRO] Coluna 'KG_LIQUIDO' não encontrada!")
            print("================ FIM resumo_dados_comex ================\n")
            return "A coluna 'KG_LIQUIDO' não foi encontrada nos dados."
    elif "principais estados" in consulta_lower:
        if 'SG_UF_NCM' in df_comex.columns:
            top_estados = df_comex['SG_UF_NCM'].value_counts().head(5)
            print(f"[OK] Top 5 estados:\n{top_estados}")
            print("================ FIM resumo_dados_comex ================\n")
            return f"Os 5 principais estados por número de operações são:\n{top_estados.to_string()}"
        else:
            print("[ERRO] Coluna 'SG_UF_NCM' não encontrada!")
            print("================ FIM resumo_dados_comex ================\n")
            return "A coluna 'SG_UF_NCM' não foi encontrada nos dados."
    else:
        print("[ERRO] Consulta não reconhecida!")
        print("================ FIM resumo_dados_comex ================\n")
        return "Não foi possível processar a sua consulta. Tente perguntas sobre 'média do peso líquido' ou 'principais estados'."

def limpar_dados_comex() -> str:
    """
    Libera os dados de comércio exterior da memória para economizar recursos computacionais.
    """
    print("\n================ INÍCIO limpar_dados_comex ================")
    global df_comex
    if df_comex is None:
        print("[AVISO] Não há dados para limpar!")
        print("================ FIM limpar_dados_comex ================\n")
        return "Não há dados para limpar na memória."
    df_comex = None
    gc.collect()
    print("[OK] Dados de comércio exterior foram limpos da memória.")
    print("================ FIM limpar_dados_comex ================\n")
    return "Os dados foram removidos da memória com sucesso."

print("Módulo de configuração de Comércio Exterior carregado.")
