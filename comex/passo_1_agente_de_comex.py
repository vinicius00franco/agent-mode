# passo_1_agente_de_comex.py
import asyncio
import passo_0_configuracao_e_ferramentas as config

async def main():
    print("\n" + "=" * 50)
    print("PASSO 1: AGENTE DE ANÁLISE DE COMÉRCIO EXTERIOR (V2 - COM INSIGHTS)")
    print("=" * 50 + "\n")

    # Criando ferramentas a partir das NOVAS funções
    ferramenta_obter_dados = config.FunctionTool.from_defaults(
        fn=config.obter_dados_comex_inteligente,
        name="obter_dados_comex_inteligente",
        description="""
        Primeiro passo obrigatório. Baixa os dados de um mês/ano específico. 
        Esta ferramenta descobre se é exportação ou importação pela pergunta do usuário.
        Usa os parâmetros 'ano', 'mes' e 'pergunta_original'.
        """,
    )

    ferramenta_analise = config.FunctionTool.from_defaults(
        fn=config.analisar_principais_entidades,
        name="analisar_principais_entidades",
        description="""
        Use esta ferramenta APÓS carregar os dados.
        Retorna um JSON com os top N de uma categoria (estados, produtos) por valor ('VL_FOB') ou peso ('KG_LIQUIDO').
        Use 'SG_UF_MUN' para estados e 'NO_NCM_POR' para nome do produto.
        """,
    )

    ferramenta_estatisticas = config.FunctionTool.from_defaults(
        fn=config.obter_estatisticas_gerais,
        name="obter_estatisticas_gerais",
        description="""
        Use esta ferramenta APÓS carregar os dados.
        Retorna um JSON com estatísticas gerais (soma, média, etc.) sobre uma coluna numérica.
        """,
    )

    ferramenta_limpar_dados = config.FunctionTool.from_defaults(
        fn=config.limpar_dados_comex,
        name="limpar_dados_comex",
        description="""
        Use esta ferramenta ao final de toda a análise para limpar os dados da memória.
        """,
    )

    agent = config.ReActAgent.from_new(
        tools=[ferramenta_obter_dados, ferramenta_analise, ferramenta_estatisticas, ferramenta_limpar_dados],
        llm=config.llm_groq,
        verbose=True,
        # Aumentar os passos pois a análise pode exigir mais interações
        max_steps=10  
    )

    print("\n--- Teste 2.1: Pergunta sobre Exportadores com Geração de Insight ---")
    pergunta1 = "Quais foram os principais estados exportadores em abril de 2024? Me dê um insight sobre os dados."
    response1 = await agent.achat(pergunta1)
    print("Resposta do Agente:", response1)

    # Limpando a memória manualmente para o próximo teste independente
    config.limpar_dados_comex() 

    print("\n--- Teste 2.2: Pergunta sobre Produtos Importados e Estatísticas ---")
    pergunta2 = "Analise as importações de maio de 2024. Quais os 3 produtos mais comprados pelo Brasil em termos de valor? Some o valor total importado no mês."
    response2 = await agent.achat(pergunta2)
    print("Resposta do Agente:", response2)

    config.limpar_dados_comex()

    print("\n" + "=" * 50)
    print("PASSO 1 (V2) CONCLUÍDO")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
