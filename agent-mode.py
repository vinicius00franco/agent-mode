# -*- coding: utf-8 -*-
"""
Consolidated Python script from the LlamaIndex and CrewAI course.
This version is adapted for a local environment (non-Colab).

This script combines all the code from the lessons into a single file.
It covers:
- Setting up LlamaIndex with different LLMs (Groq, NVIDIA).
- Creating function tools for custom Python functions.
- Using LlamaIndex agents (FunctionCallingAgent, ReActAgent).
- Integrating external tools like Arxiv and Tavily for research.
- Building and querying vector databases from documents (PDFs).
- Setting up basic, sequential, and hierarchical crews with CrewAI.

Instructions for local execution:
1. Run the `pip install` command specified below in your terminal.
2. Create a file named `.env` in the same directory as this script.
3. Add your API keys to the `.env` file in the format:
   GROQ_API_KEY="your_key"
   TAVILY_API_KEY="your_key"
   NVIDIA_API_KEY="your_key"
4. The script will create 'data' and 'storage' folders. If you have your own PDFs,
   place them in the 'data' folder before running.
"""

# ==============================================================================
# 1. INSTALL DEPENDENCIES
# ==============================================================================
# !pip install python-dotenv llama-index llama_index.embeddings.huggingface llama-index-readers-file llama-index-llms-groq arxiv llama-index-tools-tavily-research crewai crewai-tools requests

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
import os
import requests
import arxiv
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    AgentRunner,
    ReActAgent
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research import TavilyToolSpec

# CrewAI Imports
from crewai import Agent, Task, Crew, Process
from crewai.llms import LLM as CrewAI_LLM # Renamed to avoid conflict with LlamaIndex's llm variable
from crewai_tools import LlamaIndexTool

# ==============================================================================
# 3. API KEY AND SETTINGS CONFIGURATION
# ==============================================================================
# Load environment variables from a .env file
load_dotenv()

groq_key = os.getenv('GROQ_API_KEY')
tavily_key = os.getenv('TAVILY_API_KEY')
nvidia_key = os.getenv('NVIDIA_API_KEY')

# Validate that keys were loaded
if not all([groq_key, tavily_key, nvidia_key]):
    raise ValueError("API keys not found. Please create a .env file and add your GROQ, TAVILY, and NVIDIA API keys.")


# --- LlamaIndex LLM and Embedding Model Configuration ---
# LLM for LlamaIndex Agents
llm_groq = Groq(model="llama-3.1-70b-versatile", api_key=groq_key)

# Global settings for LlamaIndex
Settings.llm = llm_groq
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

# --- CrewAI LLM Configuration ---
# Note: The original code used "nvidia_nim/meta/llama-3.3-70b-instruct".
# This model name might change. Check NVIDIA's documentation for available models.
# As of late 2024, a common model is "meta/llama3-70b-instruct".
llm_crewai = CrewAI_LLM(
    model="meta/llama3-70b-instruct",
    api_key=nvidia_key,
    base_url="https://integrate.api.nvidia.com/v1" # Required for NVIDIA NIM
)


# ==============================================================================
# 4. FUNCTION DEFINITIONS
# ==============================================================================

def calcular_engajamento(curtidas: int, comentarios: int, compartilhamentos: int, seguidores: int) -> str:
    """
    Calcula o engajamento total e a taxa de engajamento de uma postagem.
    """
    if seguidores == 0:
        return "O número de seguidores não pode ser zero."
    engajamento_total = curtidas + comentarios + compartilhamentos
    taxa_engajamento = (engajamento_total / seguidores) * 100
    resultado = f"O engajamento total é {engajamento_total} e a taxa de engajamento é {taxa_engajamento:.2f}%."
    return resultado

def consulta_artigos(titulo: str) -> str:
    """
    Consulta os artigos na base de dados arXiv e retorna resultados formatados.
    """
    try:
        busca = arxiv.Search(
            query=titulo,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        resultados = []
        for resultado in busca.results():
            resultados.append(f"Título: {resultado.title}\n"
                              f"Resumo: {resultado.summary}\n"
                              f"Categoria: {resultado.primary_category}\n"
                              f"Link: {resultado.entry_id}\n")
        return "\n\n".join(resultados) if resultados else "Nenhum artigo encontrado."
    except Exception as e:
        return f"Ocorreu um erro ao buscar no arXiv: {e}"

def baixar_pdf_arxiv(link: str) -> str:
    """
    Baixa o PDF de um artigo do arXiv dado o link do artigo.
    """
    try:
        if "arxiv.org" not in link:
            return "O link fornecido não é um link válido do arXiv."
        artigo_id = link.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{artigo_id}.pdf"
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            nome_arquivo = f"artigo_{artigo_id}.pdf"
            with open(nome_arquivo, "wb") as f:
                f.write(response.content)
            return f"PDF salvo como {nome_arquivo}"
        else:
            return f"Erro ao baixar o PDF. Código de status: {response.status_code}"
    except Exception as e:
        return f"Ocorreu um erro: {e}"


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # AULA 1: CRIANDO UM AGENTE DE FUNÇÕES
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("INICIANDO AULA 1: CRIANDO UM AGENTE DE FUNÇÕES")
    print("="*50 + "\n")

    # Vídeo 1.3: Transformando a função em ferramenta
    ferramenta_calculo = FunctionTool.from_defaults(
        fn=calcular_engajamento,
        name="Calcular_Engajamento",
        description=(
            "Calcula o engajamento total e a taxa de engajamento de uma postagem. "
            "Argumentos: curtidas (int), comentarios (int), compartilhamentos (int), seguidores (int)."
        )
    )

    # Vídeo 1.4: Consultando artigos
    ferramenta_consulta_arxiv = FunctionTool.from_defaults(fn=consulta_artigos, name="Consultar_Artigos_Arxiv")

    agent_worker_aula1 = FunctionCallingAgentWorker.from_tools(
        tools=[ferramenta_calculo, ferramenta_consulta_arxiv],
        verbose=True,
        allow_parallel_tool_calls=False, # Set to False for simpler, sequential execution
        llm=llm_groq,
    )
    agent_aula1 = AgentRunner(agent_worker_aula1)

    print("\n--- Teste 1.1: Calculando engajamento ---")
    response1 = agent_aula1.chat(
        "Qual é o engajamento de uma postagem que teve 150 curtidas, "
        "35 comentários, 20 compartilhamentos, e o perfil tem 2000 seguidores?"
    )
    print("Resposta do Agente:", response1)

    print("\n--- Teste 1.2: Conhecimento geral (sem ferramenta) ---")
    response2 = agent_aula1.chat("Quem é Albert Einstein?")
    print("Resposta do Agente:", response2)

    print("\n--- Teste 1.3: Consultando artigos no Arxiv ---")
    response3 = agent_aula1.chat("Me retorne artigos sobre o uso da inteligência artificial nas redes sociais")
    print("Resposta do Agente:", response3)


    # --------------------------------------------------------------------------
    # AULA 2: APROFUNDANDO NAS PESQUISAS
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("INICIANDO AULA 2: APROFUNDANDO NAS PESQUISAS")
    print("="*50 + "\n")

    # Vídeo 2.1: Adotando ferramentas prontas (Tavily)
    tavily_tool_spec = TavilyToolSpec(api_key=tavily_key)
    tavily_tools = tavily_tool_spec.to_tool_list()

    agent_worker_tavily = FunctionCallingAgentWorker.from_tools(tavily_tools, llm=llm_groq, verbose=True)
    agent_tavily = AgentRunner(agent_worker_tavily)

    print("\n--- Teste 2.1: Pesquisando na web com Tavily ---")
    response4 = agent_tavily.chat(
        "Me retorne artigos científicos sobre o uso da inteligência artificial nas redes sociais que você encontrar na web"
    )
    print("Resposta do Agente:", response4)

    # Vídeo 2.2 & 2.3: Base de dados vetorial e engines de busca
    print("\n--- Criando e carregando base de dados vetorial ---")
    # O código criará as pastas 'data' e 'storage' no diretório atual.
    # Coloque seus PDFs em 'data' para usá-los.
    os.makedirs("data", exist_ok=True)
    os.makedirs("storage/artigo", exist_ok=True)
    os.makedirs("storage/livro", exist_ok=True)

    # Para o código rodar, vamos criar arquivos dummy. Substitua por seus PDFs reais.
    try:
        with open("data/artigo1.pdf", "w", encoding="utf-8") as f:
            f.write("Este é um texto sobre algoritmos de IA em redes sociais.")
        with open("data/livro1.pdf", "w", encoding="utf-8") as f:
            f.write("Este é um livro sobre tendências em inteligência artificial.")

        artigo_docs = SimpleDirectoryReader(input_files=["data/artigo1.pdf"]).load_data()
        livro_docs = SimpleDirectoryReader(input_files=["data/livro1.pdf"]).load_data()

        # Criando e persistindo os índices
        artigo_index = VectorStoreIndex.from_documents(artigo_docs)
        artigo_index.storage_context.persist(persist_dir="storage/artigo")

        livro_index = VectorStoreIndex.from_documents(livro_docs)
        livro_index.storage_context.persist(persist_dir="storage/livro")

        # Carregando os índices
        artigo_storage = StorageContext.from_defaults(persist_dir="storage/artigo")
        loaded_artigo_index = load_index_from_storage(artigo_storage)

        livro_storage = StorageContext.from_defaults(persist_dir="storage/livro")
        loaded_livro_index = load_index_from_storage(livro_storage)

        artigo_engine = loaded_artigo_index.as_query_engine(similarity_top_k=3)
        livro_engine = loaded_livro_index.as_query_engine(similarity_top_k=3)

        query_engine_tools = [
            QueryEngineTool(
                query_engine=artigo_engine,
                metadata=ToolMetadata(
                    name="artigo_engine",
                    description="Fornece informações sobre algoritmos de IA em redes sociais a partir de um artigo específico."
                ),
            ),
            QueryEngineTool(
                query_engine=livro_engine,
                metadata=ToolMetadata(
                    name="livro_engine",
                    description="Fornece informações sobre tendências de IA a partir de um livro específico."
                ),
            ),
        ]
        print("Bases de dados vetoriais criadas e carregadas com sucesso.")

    except Exception as e:
        print(f"Erro ao processar documentos locais: {e}")
        print("Pulando testes com documentos locais.")
        query_engine_tools = []


    # --------------------------------------------------------------------------
    # AULA 3: VERIFICANDO O FUNCIONAMENTO DO AGENTE
    # --------------------------------------------------------------------------
    if query_engine_tools:
        print("\n" + "="*50)
        print("INICIANDO AULA 3: VERIFICANDO O FUNCIONAMENTO DO AGENTE")
        print("="*50 + "\n")

        # Vídeo 3.1: Consultando textos com FunctionCallingAgent
        agent_worker_docs = FunctionCallingAgentWorker.from_tools(query_engine_tools, llm=llm_groq, verbose=True)
        agent_documentos = AgentRunner(agent_worker_docs)

        print("\n--- Teste 3.1: Consultando artigo com FunctionCallingAgent ---")
        response5 = agent_documentos.chat("Quais os principais algoritmos de IA usados nas redes sociais?")
        print("Resposta do Agente:", response5)

        print("\n--- Teste 3.2: Consultando livro com FunctionCallingAgent ---")
        response6 = agent_documentos.chat("Quais as principais tendências de IA que eu deveria estudar?")
        print("Resposta do Agente:", response6)

        # Vídeo 3.2: Usando um agente ReAct
        agent_react = ReActAgent.from_tools(query_engine_tools, llm=llm_groq, verbose=True)

        print("\n--- Teste 3.3: Consultando artigo com ReActAgent ---")
        response7 = agent_react.chat("Quais os principais algoritmos de IA usados nas redes sociais?")
        print("Resposta do Agente:", response7)
        
    # Vídeo 3.3: Configurando o CrewAI
    print("\n--- Configurando CrewAI para pesquisa no Arxiv ---")
    ferramenta_arxiv_crewai = FunctionTool.from_defaults(fn=consulta_artigos, name="consulta_artigos_arxiv")
    tool_arxiv_crewai = LlamaIndexTool.from_tool(ferramenta_arxiv_crewai)

    pesquisador_arxiv_agent = Agent(
        role='Pesquisador Científico do Arxiv',
        goal='Encontrar artigos científicos relevantes no repositório arXiv sobre um determinado tópico.',
        backstory='Você é um especialista em navegar e extrair informações do arXiv. Sua missão é fornecer os artigos mais pertinentes para qualquer consulta.',
        tools=[tool_arxiv_crewai],
        llm=llm_crewai,
        verbose=True
    )
    
    task_pesquisa_arxiv = Task(
        description="Busque artigos científicos no arXiv sobre o uso de inteligência artificial em redes sociais.",
        expected_output="Uma lista formatada de 5 artigos, cada um com título, resumo, categoria e link.",
        agent=pesquisador_arxiv_agent
    )
    
    crew_arxiv = Crew(
        agents=[pesquisador_arxiv_agent],
        tasks=[task_pesquisa_arxiv],
        verbose=2
    )
    
    result_crew1 = crew_arxiv.kickoff()
    print("\n###################### RESULTADO CREW 1 ######################")
    print(result_crew1)
    print("############################################################\n")


    # --------------------------------------------------------------------------
    # AULA 4: MÚLTIPLAS TAREFAS E AGENTES
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("INICIANDO AULA 4: MÚLTIPLAS TAREFAS E AGENTES")
    print("="*50 + "\n")

    # Vídeo 4.1: Adicionando uma nova tarefa (Download)
    print("\n--- Configurando CrewAI com tarefas sequenciais (Pesquisa e Download) ---")
    ferramenta_baixar_crewai = FunctionTool.from_defaults(fn=baixar_pdf_arxiv, name="baixar_pdf_arxiv")
    tool_baixar_crewai = LlamaIndexTool.from_tool(ferramenta_baixar_crewai)

    pesquisador_downloader_agent = Agent(
        role='Agente de Pesquisa e Download',
        goal='Encontrar e baixar artigos científicos do arXiv.',
        backstory='Você é um agente eficiente que primeiro localiza artigos no arXiv e depois baixa seus PDFs usando os links encontrados.',
        tools=[tool_arxiv_crewai, tool_baixar_crewai],
        llm=llm_crewai,
        verbose=True
    )

    task_download_pdf = Task(
        description="Usando os links da tarefa anterior, baixe o PDF do primeiro artigo encontrado. Use a ferramenta de download para isso.",
        expected_output="A confirmação de que o PDF foi salvo, com o nome do arquivo.",
        agent=pesquisador_downloader_agent,
        context=[task_pesquisa_arxiv] # Garante que esta tarefa use o resultado da anterior
    )
    
    crew_download = Crew(
        agents=[pesquisador_downloader_agent],
        tasks=[task_pesquisa_arxiv, task_download_pdf],
        verbose=2
    )
    
    result_crew2 = crew_download.kickoff()
    print("\n###################### RESULTADO CREW 2 ######################")
    print(result_crew2)
    print("############################################################\n")
    
    # Vídeo 4.2: Combinando agentes (Pesquisa na Web e Verificação)
    print("\n--- Configurando CrewAI com múltiplos agentes (Pesquisador Web e Verificador) ---")
    tavily_crewai_tools = [LlamaIndexTool.from_tool(t) for t in tavily_tools]

    pesquisador_web_agent = Agent(
        role='Pesquisador Web Especialista',
        goal='Encontrar artigos científicos na web sobre um tópico específico.',
        backstory='Você é um mestre da pesquisa online, capaz de encontrar artigos em fontes confiáveis como Springer, SciELO e ResearchGate.',
        tools=tavily_crewai_tools,
        llm=llm_crewai,
        verbose=True
    )

    verificador_agent = Agent(
        role='Verificador de Artigos Científicos',
        goal='Garantir que os links encontrados são de artigos científicos autênticos e não de posts de blog ou outras fontes não acadêmicas.',
        backstory='Com um olhar crítico, você analisa os resultados da pesquisa para filtrar e aprovar apenas artigos científicos genuínos.',
        tools=tavily_crewai_tools,
        llm=llm_crewai,
        verbose=True
    )
    
    task_pesquisa_web = Task(
        description="Busque na web por artigos científicos sobre o impacto da IA na privacidade do usuário em redes sociais.",
        expected_output="Uma lista de 5 links para artigos encontrados.",
        agent=pesquisador_web_agent
    )

    task_verificacao = Task(
        description="Verifique a lista de links fornecida pela tarefa anterior. Confirme se cada link leva a um artigo científico e retorne a lista final validada.",
        expected_output="Uma lista final de links confirmados como artigos científicos.",
        agent=verificador_agent,
        context=[task_pesquisa_web]
    )

    crew_verificacao = Crew(
        agents=[pesquisador_web_agent, verificador_agent],
        tasks=[task_pesquisa_web, task_verificacao],
        verbose=2
    )

    result_crew3 = crew_verificacao.kickoff()
    print("\n###################### RESULTADO CREW 3 ######################")
    print(result_crew3)
    print("############################################################\n")
    
    # Vídeo 4.3: Trabalhando com Hierarquia
    print("\n--- Configurando CrewAI com processo hierárquico ---")
    
    gerente_agent = Agent(
        role="Gerente de Pesquisa",
        goal="Coordenar a equipe de pesquisa para produzir uma lista validada de artigos científicos sobre um tópico.",
        backstory="Você é um gerente de projeto experiente. Sua função é delegar a busca e a verificação de artigos para os agentes especialistas, garantindo um resultado final de alta qualidade.",
        allow_delegation=True,
        llm=llm_crewai,
        verbose=True
    )

    # Reutilizando agentes e tarefas da seção anterior
    crew_hierarquica = Crew(
        agents=[pesquisador_web_agent, verificador_agent],
        tasks=[task_pesquisa_web, task_verificacao],
        manager_llm=llm_crewai, # É comum usar o mesmo LLM, ou um mais poderoso para o gerente
        process=Process.hierarchical,
        verbose=2
    )
    
    result_crew4 = crew_hierarquica.kickoff()
    print("\n###################### RESULTADO CREW 4 (Hierárquico) ######################")
    print(result_crew4)
    print("#########################################################################\n")