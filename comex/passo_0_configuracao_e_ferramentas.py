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
import json

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
def obter_dados_comex_inteligente(ano: str, mes: str, pergunta_original: str) -> str:
    """
    Baixa e carrega os dados de comércio exterior para um ano e mês.
    Esta função infere se a operação é de Exportação (EXP) ou Importação (IMP)
    com base nas palavras-chave da pergunta do usuário.

    Args:
        ano (str): O ano dos dados, por exemplo, '2024'.
        mes (str): O mês dos dados, por exemplo, 'janeiro'.
        pergunta_original (str): A pergunta completa feita pelo usuário.

    Returns:
        str: Uma mensagem de sucesso ou erro.
    """
    global df_comex
    print("\n================ INÍCIO obter_dados_comex_inteligente ================")
    print(f"Parâmetros recebidos: ano={ano}, mes={mes}")
    print(f"Analisando pergunta: '{pergunta_original}'")

    pergunta_lower = pergunta_original.lower()

    # Lógica para inferir o tipo de operação
    palavras_exportacao = ['exportado', 'exportação', 'vendeu', 'enviou', 'exportadores']
    palavras_importacao = ['importado', 'importação', 'comprou', 'recebeu', 'importadores']

    tipo_operacao = None
    if any(palavra in pergunta_lower for palavra in palavras_exportacao):
        tipo_operacao = "EXP"
    elif any(palavra in pergunta_lower for palavra in palavras_importacao):
        tipo_operacao = "IMP"

    if not tipo_operacao:
        print("[ERRO] Não foi possível determinar o tipo de operação (Exportação/Importação).")
        print("================ FIM obter_dados_comex_inteligente ================\n")
        return "Não consegui identificar se a pergunta se refere a exportação ou importação. Por favor, seja mais específico."

    print(f"[INFO] Operação inferida: {tipo_operacao}")

    # O resto da função continua o mesmo, usando a variável `tipo_operacao` inferida
    meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    mes_num = meses.get(mes.lower())

    if not mes_num:
        return "Mês inválido."

    url = f"https://balanca.economia.gov.br/balanca/bd/comexstat-bd/mun/{tipo_operacao}_{ano}_MUN.csv"
    try:
        print(f"[DOWNLOAD] Baixando dados de {tipo_operacao} para {ano}...")
        df_anual = pd.read_csv(url, sep=';', encoding='iso-8859-1')
        df_comex = df_anual[df_anual['CO_MES'] == mes_num].copy()
        if df_comex.empty:
            return f"Nenhum dado de {tipo_operacao} encontrado para {mes}/{ano}."

        print(f"[OK] Dados de {tipo_operacao} para {mes}/{ano} carregados. Linhas: {len(df_comex)}")
        print("================ FIM obter_dados_comex_inteligente ================\n")
        return f"Dados de {tipo_operacao} para {mes}/{ano} carregados com sucesso. Agora posso analisá-los para gerar insights."
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

def analisar_principais_entidades(
    coluna_entidade: str, 
    coluna_metrica: str = 'VL_FOB', 
    top_n: int = 5
) -> str:
    """
    Analisa os dados carregados para encontrar as principais entidades (estados, municípios, produtos)
    com base em uma métrica (valor ou peso).

    Args:
        coluna_entidade (str): A coluna a ser agrupada (ex: 'SG_UF_MUN', 'NO_NCM_POR').
        coluna_metrica (str): A métrica para somar (ex: 'VL_FOB' para valor, 'KG_LIQUIDO' para peso).
        top_n (int): O número de principais resultados a serem retornados.

    Returns:
        str: Uma string JSON contendo os dados da análise ou uma mensagem de erro.
    """
    print("\n================ INÍCIO analisar_principais_entidades ================")
    global df_comex
    if df_comex is None:
        return "Erro: Nenhum dado carregado. Use 'obter_dados_comex_inteligente' primeiro."
    if coluna_entidade not in df_comex.columns or coluna_metrica not in df_comex.columns:
        return f"Erro: Uma das colunas '{coluna_entidade}' ou '{coluna_metrica}' não foi encontrada."

    print(f"Analisando top {top_n} de '{coluna_entidade}' pela métrica '{coluna_metrica}'")

    # Agrupar, somar, ordenar e pegar o top N
    analise_df = df_comex.groupby(coluna_entidade)[coluna_metrica].sum().sort_values(ascending=False).head(top_n)

    # Formatar como um dicionário para converter para JSON
    resultado = analise_df.reset_index().to_dict(orient='records')

    print(f"[OK] Análise concluída: {resultado}")
    print("================ FIM analisar_principais_entidades ================\n")
    return json.dumps(resultado, ensure_ascii=False)

def obter_estatisticas_gerais(coluna_metrica: str = 'VL_FOB') -> str:
    """
    Calcula estatísticas descritivas gerais (total, média, máximo) para uma métrica.

    Args:
        coluna_metrica (str): A métrica a ser analisada (ex: 'VL_FOB', 'KG_LIQUIDO').

    Returns:
        str: Uma string JSON contendo as estatísticas ou uma mensagem de erro.
    """
    print("\n================ INÍCIO obter_estatisticas_gerais ================")
    global df_comex
    if df_comex is None:
        return "Erro: Nenhum dado carregado."
    if coluna_metrica not in df_comex.columns:
        return f"Erro: A coluna '{coluna_metrica}' não foi encontrada."

    total = df_comex[coluna_metrica].sum()
    media = df_comex[coluna_metrica].mean()
    maximo = df_comex[coluna_metrica].max()

    resultado = {
        f"total_{coluna_metrica}": total,
        f"media_{coluna_metrica}": media,
        f"maximo_{coluna_metrica}": maximo,
        "numero_operacoes": len(df_comex)
    }

    print(f"[OK] Estatísticas calculadas: {resultado}")
    print("================ FIM obter_estatisticas_gerais ================\n")
    return json.dumps(resultado)

print("Módulo de configuração de Comércio Exterior carregado.")
