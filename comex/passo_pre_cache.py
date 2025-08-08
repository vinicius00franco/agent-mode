# -*- coding: utf-8 -*-
"""
Script de pré-aquecimento do cache. Baixa os dados de comércio exterior
para anos e tipos especificados e os armazena no cache local.
"""
import asyncio
import time
from passo_0_configuracao_e_ferramentas import obter_dados_comex_por_ano

# --- CONFIGURAÇÃO ---
ANOS_PARA_CACHE = ['2023', '2024']
TIPOS_OPERACAO = ['EXP', 'IMP']
# --------------------

async def main():
    print("="*50)
    print("INICIANDO SCRIPT DE PRÉ-AQUECIMENTO DO CACHE")
    print("="*50)
    inicio = time.time()
    
    tasks = []
    for ano in ANOS_PARA_CACHE:
        for tipo in TIPOS_OPERACAO:
            pergunta_falsa = f"Dados de {tipo} para o ano {ano}"
            task = asyncio.to_thread(
                obter_dados_comex_por_ano,
                ano=ano,
                pergunta_original=pergunta_falsa,
                usar_cache=True
            )
            tasks.append(task)
            
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    fim = time.time()
    print("\n" + "="*50)
    print(f"PRÉ-AQUECIMENTO CONCLUÍDO em {fim - inicio:.2f} segundos.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
