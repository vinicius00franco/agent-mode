# passo_0_configuracao_e_ferramentas.py
# -*- coding: utf-8 -*-
"""
Arquivo de configuração central. NÃO EXECUTE DIRETAMENTE.
Este módulo contém todas as configurações, imports, chaves de API,
inicialização de modelos e definições de função para serem
reutilizados pelos outros scripts de passo a passo.
"""

# ==============================================================================
# IMPORTS E CONFIGURAÇÕES
# ==============================================================================
import os
import requests
import arxiv
import gc
import json  # Usaremos para salvar e carregar resultados
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.tools.tavily_research import TavilyToolSpec

# CrewAI Imports (serão importados quando necessário nos passos específicos)
# from crewai import Agent, Task, Crew, Process
# from crewai_tools import LlamaIndexTool

# ==============================================================================
# CARREGAMENTO DE CHAVES DE API
# ==============================================================================
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
nvidia_key = os.getenv("NVIDIA_API_KEY")

if not all([groq_key, tavily_key, nvidia_key]):
    raise ValueError("Chaves de API não encontradas no arquivo .env")

# ==============================================================================
# CONFIGURAÇÃO GLOBAL DE MODELOS
# ==============================================================================
# LLM para LlamaIndex (rápido para agentes)
llm_groq = Groq(model="llama-3.1-8b-instant", api_key=groq_key)

# LLM para CrewAI será configurado quando necessário nos outros passos

# Configurações globais do LlamaIndex
Settings.llm = llm_groq
Settings.embed_model = NVIDIAEmbedding(
    model="nv-embed-qa-e4", api_key=nvidia_key, truncate="END"
)

# Configuração otimizada do garbage collector
gc.set_threshold(700, 10, 10)


# ==============================================================================
# DEFINIÇÕES DE FUNÇÕES (FERRAMENTAS)
# ==============================================================================
def calcular_engajamento(
    curtidas: int, comentarios: int, compartilhamentos: int, seguidores: int
) -> str:
    if seguidores == 0:
        return "O número de seguidores não pode ser zero."
    engajamento_total = curtidas + comentarios + compartilhamentos
    taxa_engajamento = (engajamento_total / seguidores) * 100
    return f"O engajamento total é {engajamento_total} e a taxa de engajamento é {taxa_engajamento:.2f}%."


def consulta_artigos(titulo: str) -> str:
    try:
        busca = arxiv.Search(
            query=titulo, max_results=5, sort_by=arxiv.SortCriterion.Relevance
        )
        resultados = []
        links = []
        for resultado in busca.results():
            resultados.append(
                f"Título: {resultado.title}\nResumo: {resultado.summary}\nCategoria: {resultado.primary_category}\nLink: {resultado.entry_id}\n"
            )
            links.append(resultado.entry_id)

        # Salva o resultado para ser usado por outros passos
        with open("resultado_pesquisa_arxiv.json", "w", encoding="utf-8") as f:
            json.dump(links, f)
        print("Resultado da pesquisa Arxiv salvo em 'resultado_pesquisa_arxiv.json'")
        return "\n\n".join(resultados) if resultados else "Nenhum artigo encontrado."
    except Exception as e:
        return f"Ocorreu um erro ao buscar no arXiv: {e}"


def baixar_pdf_arxiv(link: str) -> str:
    try:
        if "arxiv.org" not in link:
            return "O link fornecido não é um link válido do arXiv."
        artigo_id = link.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{artigo_id}.pdf"
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            os.makedirs("downloads", exist_ok=True)
            nome_arquivo = f"downloads/artigo_{artigo_id}.pdf"
            with open(nome_arquivo, "wb") as f:
                f.write(response.content)
            return f"PDF salvo como {nome_arquivo}"
        else:
            return f"Erro ao baixar o PDF. Código de status: {response.status_code}"
    except Exception as e:
        return f"Ocorreu um erro: {e}"


print("Módulo de configuração carregado.")
