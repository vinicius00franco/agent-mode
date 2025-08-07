# passo_4_crew_pesquisa_e_download.py
import passo_0_configuracao_e_ferramentas as config
import json

if __name__ == '__main__':
    print("\n" + "="*50)
    print("PASSO 4: CREWAI COM TAREFAS SEQUENCIAIS (AULA 4)")
    print("="*50 + "\n")

    # Ferramentas para o CrewAI
    tool_arxiv = config.LlamaIndexTool.from_tool(config.FunctionTool.from_defaults(fn=config.consulta_artigos))
    tool_baixar = config.LlamaIndexTool.from_tool(config.FunctionTool.from_defaults(fn=config.baixar_pdf_arxiv))

    # Agente
    pesquisador_downloader_agent = config.Agent(
        role='Agente de Pesquisa e Download',
        goal='Encontrar e baixar artigos científicos do arXiv.',
        backstory='Você é um agente eficiente que primeiro localiza artigos e depois baixa seus PDFs.',
        tools=[tool_arxiv, tool_baixar],
        llm=config.llm_crewai,
        verbose=True
    )

    # Tarefas
    task_pesquisa = config.Task(
        description="Busque artigos no arXiv sobre 'Large Language Models'.",
        expected_output="Uma lista formatada dos artigos e seus links. O resultado também deve ser salvo em disco.",
        agent=pesquisador_downloader_agent
    )

    task_download = config.Task(
        description=(
            "Leia o arquivo 'resultado_pesquisa_arxiv.json' para obter os links da pesquisa anterior. "
            "Baixe o PDF do PRIMEIRO artigo da lista."
        ),
        expected_output="A confirmação de que o PDF foi salvo, com o nome do arquivo.",
        agent=pesquisador_downloader_agent,
        context=[task_pesquisa] # Garante que a pesquisa ocorra primeiro
    )

    # Crew
    crew = config.Crew(
        agents=[pesquisador_downloader_agent],
        tasks=[task_pesquisa, task_download],
        verbose=2
    )

    result = crew.kickoff()
    
    print("\n###################### RESULTADO CREW SEQUENCIAL ######################")
    print(result)
    print("#####################################################################\n")
    
    print("\n" + "="*50)
    print("PASSO 4 CONCLUÍDO")
    print("="*50 + "\n")