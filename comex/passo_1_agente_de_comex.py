import asyncio
import passo_0_configuracao_e_ferramentas as config

async def main():
    print("--- CHECKPOINT 1: AGENTE INICIAL ---")

    ferramenta_obter_dados = config.FunctionTool.from_defaults(fn=config.obter_dados_comex)
    ferramenta_resumo_dados = config.FunctionTool.from_defaults(fn=config.resumo_dados_comex)
    ferramenta_limpar_dados = config.FunctionTool.from_defaults(fn=config.limpar_dados_comex)

    agent = config.ReActAgent.from_new(
        tools=[ferramenta_obter_dados, ferramenta_resumo_dados, ferramenta_limpar_dados],
        llm=config.llm_groq,
        verbose=True
    )

    response = await agent.achat("Baixe os dados de exportação de abril de 2024. Depois, me diga a média do peso líquido.")
    print("Resposta do Agente:", response)

if __name__ == "__main__":
    asyncio.run(main())
