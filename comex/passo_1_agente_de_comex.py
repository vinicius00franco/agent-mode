import asyncio
import nest_asyncio  # NOVO: Importa a biblioteca

# NOVO: Aplica o patch que permite loops aninhados do asyncio.
nest_asyncio.apply()

# ALTERADO: A importação agora aponta para a classe correta e moderna do agente ReAct.
# Não há mais a classe AgentRunner separada, o próprio ReActAgent é o workflow executável.
from llama_index.core.agent.workflow import ReActAgent
import passo_0_configuracao_e_ferramentas as config


async def main():
    print("--- CHECKPOINT 2.2: AGENTE ANALÍTICO COM WORKFLOW CORRETO ---")

    # A definição das suas ferramentas continua exatamente a mesma
    ferramenta_obter_dados = config.FunctionTool.from_defaults(
        fn=config.obter_dados_comex_por_ano,
        name="obter_dados_comex_por_ano",
        description="Passo inicial. Carrega dados de um ANO INTEIRO. A pergunta do usuário deve conter 'exportação' ou 'importação' para inferir o tipo. Use 'usar_cache=False' para forçar o recarregamento.",
    )
    ferramenta_analise = config.FunctionTool.from_defaults(
        fn=config.analisar_principais_entidades,
        name="analisar_principais_entidades",
        description="Analisa os dados anuais carregados. Pode filtrar por mês (ex: mes='abril'). Colunas comuns para 'coluna_entidade' são 'SG_UF_NCM' (estado), 'NO_MUN_MIN' (município), 'NO_NCM_POR' (produto).",
    )
    ferramenta_estatisticas = config.FunctionTool.from_defaults(
        fn=config.obter_estatisticas_gerais,
        name="obter_estatisticas_gerais",
        description="Calcula estatísticas gerais dos dados anuais. Pode filtrar por mês.",
    )
    ferramenta_limpar_dados = config.FunctionTool.from_defaults(
        fn=config.limpar_dados_comex
    )

    # A criação do agente com .from_tools() continua válida e é a forma recomendada
    agent = ReActAgent(
        tools=[
            ferramenta_obter_dados,
            ferramenta_analise,
            ferramenta_estatisticas,
            ferramenta_limpar_dados,
        ],
        llm=config.llm_groq,
        verbose=True,
    )

    print("\n>>> INICIANDO TAREFA 1: Carregar dados de 2024...")
    response1 = await asyncio.to_thread(
        agent.run, message="Carregue os dados de exportação de 2024."
    )
    print("\n--- FIM DA TAREFA 1 ---")
    print("Resposta do Agente:", response1)

    print("\n>>> INICIANDO TAREFA 2: Analisar dados de Maio...")
    response2 = await asyncio.to_thread(
        agent.run,
        message="Agora, com base nesses dados, me dê um insight sobre os 5 principais estados (SG_UF_NCM) exportadores em maio.",
    )
    print("\n--- FIM DA TAREFA 2 ---")
    print("Resposta Final do Agente:", response2)


if __name__ == "__main__":
    # Lembrete: Certifique-se de que o 'pyarrow' está instalado para o cache funcionar!
    # pip install pyarrow
    asyncio.run(main())
