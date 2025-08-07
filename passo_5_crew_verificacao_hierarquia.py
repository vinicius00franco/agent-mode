# passo_5_crew_verificacao_hierarquia.py
import passo_0_configuracao_e_ferramentas as config

if __name__ == '__main__':
    print("\n" + "="*50)
    print("PASSO 5: CREWAI COM VERIFICAÇÃO E HIERARQUIA (AULA 4)")
    print("="*50 + "\n")

    # Ferramenta de pesquisa web (Tavily)
    tavily_tool_spec = config.TavilyToolSpec(api_key=config.tavily_key)
    tavily_tools = [config.LlamaIndexTool.from_tool(t) for t in tavily_tool_spec.to_tool_list()]

    # Agentes
    pesquisador_web = config.Agent(
        role='Pesquisador Web Especialista',
        goal='Encontrar artigos científicos na web sobre IA na privacidade.',
        backstory='Você é mestre da pesquisa online, focado em fontes confiáveis.',
        tools=tavily_tools, llm=config.llm_crewai, verbose=True
    )
    verificador = config.Agent(
        role='Verificador de Artigos',
        goal='Garantir que os links encontrados são de artigos científicos autênticos.',
        backstory='Você tem um olhar crítico para filtrar apenas artigos genuínos.',
        tools=tavily_tools, llm=config.llm_crewai, verbose=True
    )

    # Tarefas para o Crew de verificação
    task_pesquisa_web = config.Task(
        description="Busque na web por artigos sobre o impacto da IA na privacidade do usuário em redes sociais.",
        expected_output="Uma lista de 5 links para artigos encontrados.",
        agent=pesquisador_web
    )
    task_verificacao = config.Task(
        description="Verifique a lista de links da tarefa anterior. Retorne a lista final validada.",
        expected_output="Uma lista final de links confirmados como artigos científicos.",
        agent=verificador, context=[task_pesquisa_web]
    )

    # Crew com processo sequencial de verificação
    print("\n--- Executando Crew de Verificação (Sequencial) ---")
    crew_verificacao = config.Crew(
        agents=[pesquisador_web, verificador],
        tasks=[task_pesquisa_web, task_verificacao],
        verbose=2
    )
    result_verificacao = crew_verificacao.kickoff()
    print("\n###################### RESULTADO CREW VERIFICAÇÃO ######################")
    print(result_verificacao)
    print("######################################################################\n")

    # Crew com processo hierárquico
    print("\n--- Executando Crew com Gerente (Hierárquico) ---")
    gerente = config.Agent(
        role="Gerente de Pesquisa",
        goal="Coordenar a equipe para produzir uma lista validada de artigos sobre IA na privacidade.",
        backstory="Você delega a busca e a verificação para garantir um resultado de alta qualidade.",
        allow_delegation=True, llm=config.llm_crewai, verbose=True
    )
    crew_hierarquica = config.Crew(
        agents=[pesquisador_web, verificador],
        tasks=[task_pesquisa_web, task_verificacao],
        manager_llm=config.llm_crewai,
        process=config.Process.hierarchical,
        verbose=2
    )
    result_hierarquia = crew_hierarquica.kickoff()
    print("\n###################### RESULTADO CREW HIERÁRQUICO ######################")
    print(result_hierarquia)
    print("######################################################################\n")
    
    print("\n" + "="*50)
    print("PASSO 5 CONCLUÍDO")
    print("="*50 + "\n")