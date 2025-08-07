# passo_1_agente_de_comex.py
import asyncio
import passo_0_configuracao_e_ferramentas as config

async def main():
    print("\n" + "=" * 50)
    print("PASSO 1: AGENTE DE ANÁLISE DE COMÉRCIO EXTERIOR")
    print("=" * 50 + "\n")

    # Criando ferramentas a partir das funções importadas
    ferramenta_obter_dados = config.FunctionTool.from_defaults(
        fn=config.obter_dados_comex,
        name="obter_dados_comex",
        description="""
        Esta ferramenta baixa e carrega os dados brutos de exportação ou importação
        do governo brasileiro para um ano e mês específicos.
        Usa os parâmetros 'ano', 'mes' e 'tipo_operacao' ('EXPORTACAO' ou 'IMPORTACAO').
        """,
    )
    
    ferramenta_resumo_dados = config.FunctionTool.from_defaults(
        fn=config.resumo_dados_comex,
        name="resumo_dados_comex",
        description="""
        Esta ferramenta executa consultas específicas sobre os dados de comércio exterior
        já carregados na memória, como 'média do peso líquido' ou 'principais estados'.
        Usa o parâmetro 'consulta'.
        """,
    )

    ferramenta_limpar_dados = config.FunctionTool.from_defaults(
        fn=config.limpar_dados_comex,
        name="limpar_dados_comex",
        description="""
        Esta ferramenta remove os dados carregados da memória para liberar recursos.
        Use-a quando a análise estiver completa.
        """,
    )

    agent = config.ReActAgent(
        tools=[ferramenta_obter_dados, ferramenta_resumo_dados, ferramenta_limpar_dados],
        llm=config.llm_groq,
        verbose=True,
    )

    print("\n--- Teste 1.1: Cenário para Agente de Cargas ---")
    response1 = await agent.run(
        "Baixe os dados de importação de maio de 2024. Depois, me diga qual a média do peso líquido das cargas. Por fim, limpe os dados da memória."
    )
    print("Resposta do Agente:", response1)
    
    print("\n--- Teste 1.2: Cenário para Exportadores ---")
    response2 = await agent.run(
        "Para o ano de 2024, no mês de abril, quais foram os 5 principais estados exportadores? Baixe os dados necessários e me dê essa informação. Depois, limpe a memória."
    )
    print("Resposta do Agente:", response2)

    print("\n--- Teste 1.3: Tentativa de análise sem baixar os dados ---")
    response3 = await agent.run(
        "Qual a média do peso líquido das exportações? Não sei o mês ou ano, e não baixei os dados."
    )
    print("Resposta do Agente:", response3)

    print("\n" + "=" * 50)
    print("PASSO 1 CONCLUÍDO")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
