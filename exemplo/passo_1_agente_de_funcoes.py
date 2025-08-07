# passo_1_agente_de_funcoes.py
import asyncio
import passo_0_configuracao_e_ferramentas as config


async def main():
    print("\n" + "=" * 50)
    print("PASSO 1: AGENTE DE FUNÇÕES SIMPLES (AULA 1)")
    print("=" * 50 + "\n")

    # Criando ferramentas a partir das funções importadas
    ferramenta_calculo = config.FunctionTool.from_defaults(
        fn=config.calcular_engajamento
    )
    ferramenta_consulta_arxiv = config.FunctionTool.from_defaults(
        fn=config.consulta_artigos
    )

    agent = config.ReActAgent(
        tools=[ferramenta_calculo, ferramenta_consulta_arxiv],
        llm=config.llm_groq,
        verbose=True,
    )

    print("\n--- Teste 1.1: Calculando engajamento ---")
    response1 = await agent.run(
        "Qual é o engajamento de uma postagem com 150 curtidas, 35 comentários, 20 compartilhamentos e 2000 seguidores?"
    )
    print("Resposta do Agente:", response1)

    print("\n--- Teste 1.2: Consultando artigos no Arxiv ---")
    response2 = await agent.run(
        "Me retorne artigos sobre o uso da inteligência artificial nas redes sociais"
    )
    print("Resposta do Agente:", response2)

    print("\n" + "=" * 50)
    print("PASSO 1 CONCLUÍDO")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
