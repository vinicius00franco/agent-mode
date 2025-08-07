# Instruções de Execução - Projeto LlamaIndex e CrewAI

## Estrutura dos Arquivos

O projeto foi dividido em 6 arquivos para otimizar o uso de memória e facilitar a execução modular:

1. **passo_0_configuracao_e_ferramentas.py** - Arquivo central de configuração (NÃO EXECUTE DIRETAMENTE)
2. **passo_1_agente_de_funcoes.py** - Agente simples com ferramentas (Aula 1)
3. **passo_2_criacao_base_vetorial.py** - Criação da base de dados vetorial (Aula 2)
4. **passo_3_consulta_base_vetorial.py** - Consulta da base vetorial (Aula 3)
5. **passo_4_crew_pesquisa_e_download.py** - CrewAI com tarefas sequenciais (Aula 4)
6. **passo_5_crew_verificacao_hierarquia.py** - CrewAI com verificação e hierarquia (Aula 4)

## Como Executar

### Pré-requisitos
1. Certifique-se de que o arquivo `.env` existe com suas chaves de API:
   ```
   GROQ_API_KEY="sua_chave_groq"
   TAVILY_API_KEY="sua_chave_tavily"
   NVIDIA_API_KEY="sua_chave_nvidia"
   ```

### Ordem de Execução

Execute os scripts na seguinte ordem:

```bash
# 1. Primeiro, o agente simples
python passo_1_agente_de_funcoes.py

# 2. Depois, a criação da base de dados (o passo mais pesado)
python passo_2_criacao_base_vetorial.py

# 3. Agora, consulte a base de dados que foi salva em disco
python passo_3_consulta_base_vetorial.py

# 4. Execute a equipe de pesquisa e download
python passo_4_crew_pesquisa_e_download.py

# 5. Por fim, execute as equipes avançadas
python passo_5_crew_verificacao_hierarquia.py
```

## Vantagens desta Estrutura

- **Otimização de Memória**: Cada script executa independentemente, liberando memória ao final
- **Modularidade**: Você pode executar apenas as partes que precisa
- **Persistência**: A base vetorial é salva em disco e reutilizada
- **Debugging**: Mais fácil identificar problemas em partes específicas
- **Resultados Intermediários**: Arquivos são salvos para inspeção (storage/, downloads/, resultado_pesquisa_arxiv.json)

## Arquivos Gerados

Durante a execução, os seguintes arquivos/pastas serão criados:
- `data/` - Documentos de exemplo
- `storage/` - Índices vetoriais persistidos
- `downloads/` - PDFs baixados
- `resultado_pesquisa_arxiv.json` - Links dos artigos encontrados

## Observações

- O **passo_0_configuracao_e_ferramentas.py** nunca deve ser executado diretamente
- O **passo_2** é o mais pesado computacionalmente (criação dos embeddings)
- Os **passos_3** em diante dependem da execução do **passo_2**
- Cada script pode ser executado independentemente após suas dependências