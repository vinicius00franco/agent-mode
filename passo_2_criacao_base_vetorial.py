# passo_2_criacao_base_vetorial.py
import passo_0_configuracao_e_ferramentas as config
import os

if __name__ == '__main__':
    print("\n" + "="*50)
    print("PASSO 2: CRIAÇÃO E PERSISTÊNCIA DA BASE VETORIAL (AULA 2)")
    print("="*50 + "\n")

    # Garante que as pastas de dados e armazenamento existam
    os.makedirs("data", exist_ok=True)
    os.makedirs("storage/artigo", exist_ok=True)
    os.makedirs("storage/livro", exist_ok=True)

    # Cria arquivos de exemplo se não existirem
    if not os.path.exists("data/artigo1.txt"):
        with open("data/artigo1.txt", "w", encoding="utf-8") as f:
            f.write("Este é um texto sobre algoritmos de IA em redes sociais. O algoritmo de recomendação é crucial.")
    if not os.path.exists("data/livro1.txt"):
        with open("data/livro1.txt", "w", encoding="utf-8") as f:
            f.write("Este é um livro sobre tendências em inteligência artificial. As principais tendências para estudar são IA generativa e ética em IA.")

    try:
        print("Lendo documentos da pasta 'data'...")
        artigo_docs = config.SimpleDirectoryReader(input_files=["data/artigo1.txt"]).load_data()
        livro_docs = config.SimpleDirectoryReader(input_files=["data/livro1.txt"]).load_data()
        
        print("Gerando e persistindo índice 'artigo'...")
        artigo_index = config.VectorStoreIndex.from_documents(artigo_docs)
        artigo_index.storage_context.persist(persist_dir="storage/artigo")
        
        print("Gerando e persistindo índice 'livro'...")
        livro_index = config.VectorStoreIndex.from_documents(livro_docs)
        livro_index.storage_context.persist(persist_dir="storage/livro")

        # Limpeza de memória
        del artigo_docs, livro_docs, artigo_index, livro_index
        config.gc.collect()

        print("\nÍndices vetoriais criados e salvos com sucesso na pasta 'storage'.")

    except Exception as e:
        print(f"Ocorreu um erro ao criar a base vetorial: {e}")

    print("\n" + "="*50)
    print("PASSO 2 CONCLUÍDO")
    print("="*50 + "\n")