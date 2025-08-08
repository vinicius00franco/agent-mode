import asyncio
import passo_0_configuracao_e_ferramentas as config

async def main():
    print("--- CHECKPOINT 2: AGENTE ANALÍTICO COM CACHE ---")

    ferramenta_obter_dados = config.FunctionTool.from_defaults(
        fn=config.obter_dados_comex_por_ano,
        name="obter_dados_comex_por_ano",
        description="Passo inicial. Carrega dados de um ANO INTEIRO. Use 'usar_cache=False' para forçar o recarregamento.",
    )
    ferramenta_analise = config.FunctionTool.from_defaults(
        fn=config.analisar_principais_entidades,
        name="analisar_principais_entidades",
        description="Analisa os dados anuais carregados. Pode filtrar por mês (ex: mes='abril').",
    )
    ferramenta_estatisticas = config.FunctionTool.from_defaults(
        fn=config.obter_estatisticas_gerais,
        name="obter_estatisticas_gerais",
        description="Calcula estatísticas gerais dos dados anuais. Pode filtrar por mês.",
    )
    ferramenta_limpar_dados = config.FunctionTool.from_defaults(fn=config.limpar_dados_comex)

    agent = config.ReActAgent.from_new(
        tools=[ferramenta_obter_dados, ferramenta_analise, ferramenta_estatisticas, ferramenta_limpar_dados],
        llm=config.llm_groq,
        verbose=True
    )

    await agent.achat("Carregue os dados de exportação de 2024.")
    response = await agent.achat("Agora, com base nesses dados, me dê um insight sobre os 5 principais estados exportadores em maio.")
    print("Resposta do Agente:", response)

if __name__ == "__main__":
    asyncio.run(main())
