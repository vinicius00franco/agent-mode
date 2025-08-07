# passo_3_consulta_base_vetorial.py
import passo_0_configuracao_e_ferramentas as config
import os

if __name__ == '__main__':
    print("\n" + "="*50)
    print("PASSO 3: CONSULTANDO A BASE VETORIAL DO DISCO (AULA 3)")
    print("="*50 + "\n")

    if not os.path.exists("storage/artigo") or not os.path.exists("storage/livro"):
        print("Erro: A base de dados vetorial não foi encontrada.")
        print("Por favor, execute 'passo_2_criacao_base_vetorial.py' primeiro.")
    else:
        print("Carregando índices da pasta 'storage'...")
        artigo_storage = config.StorageContext.from_defaults(persist_dir="storage/artigo")
        loaded_artigo_index = config.load_index_from_storage(artigo_storage)

        livro_storage = config.StorageContext.from_defaults(persist_dir="storage/livro")
        loaded_livro_index = config.load_index_from_storage(livro_storage)

        artigo_engine = loaded_artigo_index.as_query_engine(similarity_top_k=3)
        livro_engine = loaded_livro_index.as_query_engine(similarity_top_k=3)
        
        query_engine_tools = [
            config.QueryEngineTool(query_engine=artigo_engine, metadata=config.ToolMetadata(name="artigo_engine", description="Fornece informações sobre algoritmos de IA em redes sociais a partir de um artigo.")),
            config.QueryEngineTool(query_engine=livro_engine, metadata=config.ToolMetadata(name="livro_engine", description="Fornece informações sobre tendências de IA a partir de um livro.")),
        ]
        print("Motores de consulta prontos.")

        agent_documentos = config.AgentRunner(
            config.FunctionCallingAgentWorker.from_tools(query_engine_tools, llm=config.llm_groq, verbose=True)
        )

        print("\n--- Teste 3.1: Consultando artigo ---")
        response = agent_documentos.chat("Quais os principais algoritmos de IA usados nas redes sociais?")
        print("Resposta do Agente:", response)
        
        print("\n--- Teste 3.2: Consultando livro ---")
        response = agent_documentos.chat("Quais as principais tendências de IA que eu deveria estudar?")
        print("Resposta do Agente:", response)
        
    print("\n" + "="*50)
    print("PASSO 3 CONCLUÍDO")
    print("="*50 + "\n")