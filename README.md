# UniAssist — Classificador de mensagens

**Alunos:** Marcelo Negrão, Maria Eduarda Aguiar e Mariana Martins  
**Professor:** Neuton Melo  
**Disciplina:** Inteligência Artificial

Compara regras por palavras-chave com aprendizado de máquina usando TF-IDF + LinearSVC. Projeto baseado no material fornecido para a atividade prática de Inteligência Artificial.

## Como executar no Windows / VS Code

1. Extraia o ZIP em uma pasta. Não execute diretamente de dentro do arquivo compactado.
2. No VS Code, escolha **Arquivo → Abrir Pasta** e abra `UniAssist_Completo`.
3. Abra **Terminal → Novo Terminal**. Os comandos abaixo devem ser executados nessa pasta.
4. Instale a dependência:

```powershell
python -m pip install -r requirements.txt
```

5. Execute:

```powershell
python uniaassist_pln.py
```

Se o Windows reconhecer apenas `py`, substitua `python` por `py` nos dois comandos. Se nenhum comando for encontrado, instale Python e habilite sua inclusão no PATH. A extensão Python do VS Code ajuda a selecionar o interpretador; a biblioteca precisa estar instalada no mesmo interpretador usado para executar.

Instale a dependência antes de uma aula sem internet. Após a instalação, o programa funciona localmente, sem APIs ou downloads de modelos. Testado com Python 3.12.14 e scikit-learn 1.8.0. `requirements.txt` mantém a faixa original do material. Para reproduzir a versão da biblioteca usada nesta execução:

```powershell
python -m pip install scikit-learn==1.8.0
```

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `uniaassist_pln.py` | Código completo, comentado e executável |
| `parecer.md` | Métricas, matrizes, análise dos oito desafios e conclusão |
| `mensagens.csv` | Dataset original de treino e teste |
| `desafios.csv` | Oito mensagens originais sem rótulo oficial |
| `requirements.txt` | Dependência do projeto |
| `resultados_execucao.txt` | Saída real da execução de referência |
| `LEIA_ME.md` | Orientações originais do professor |

## Entendendo o programa

1. Lê mensagens e setores do CSV.
2. Separa 60 mensagens para treino e 20 para teste, mantendo 15/5 de cada setor.
3. Treina TF-IDF (transforma palavras em números) + LinearSVC (aprende a separar setores).
4. Compara o modelo e as regras nas mesmas mensagens de teste.
5. Mostra os dois resultados para cada desafio.
6. Repete o treinamento com palavras isoladas para o experimento solicitado.

`fit` aprende com o treino; `predict` classifica textos. O teste serve para medir o comportamento em exemplos que o modelo não viu. `random_state=42` fixa a aleatoriedade e `stratify` preserva a proporção dos setores. Nenhum dado de teste é usado para ajustar o vocabulário TF-IDF.

## Resultado de referência

- Regras: 16/20 acertos (80%); quatro mensagens para revisão humana.
- Modelo principal: 14/20 acertos (70%).
- Experimento com unigramas: 14/20 acertos (70%).

Não há gabarito oficial para os desafios; os encaminhamentos sugeridos no parecer exigem avaliação contextual. O protótipo não possui interface gráfica nem política automática completa de triagem híbrida.

## Entrega no GitHub

Crie um repositório e envie os arquivos extraídos, usando **Add file → Upload files**, e confirme em **Commit changes**. O ZIP sozinho não permite visualizar o código facilmente no repositório. Envie o link do repositório na atividade.

O código e `parecer.md` são os dois entregáveis centrais. Mantenha também os CSVs e `requirements.txt` no repositório para que outra pessoa consiga executar; README e saída de referência documentam o uso. O nome do script foi preservado exatamente como no material: `uniaassist_pln.py`.

Os integrantes e o professor estão identificados neste README e no parecer. O material original menciona restrição a IA generativa durante a aula: confirme as regras aplicáveis ao uso deste apoio na sua entrega.
