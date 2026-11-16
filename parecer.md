# Parecer técnico — Classificador de Mensagens da UniAssist

**Alunos:** Marcelo Negrão, Maria Eduarda Aguiar e Mariana Martins  
**Professor:** Neuton Melo  
**Disciplina:** Inteligência Artificial

## 1. Procedimento e reprodutibilidade

Foram utilizados os 80 exemplos sintéticos originais, sem alterações nos textos ou rótulos: 20 mensagens para cada setor. A divisão estratificada usou `test_size=0.25`, `random_state=42` e `stratify=rotulos`, resultando em 60 exemplos de treino (15 por setor) e 20 de teste (5 por setor). Os dois métodos foram avaliados no mesmo teste.

As regras originais foram mantidas. O texto passa por `casefold()` e cada palavra-chave distinta encontrada como substring soma um ponto para seu setor. Empate na maior pontuação ou pontuação zero resulta em `Revisão humana`. Essa implementação simples pode contar `matrícula` dentro de `rematrícula`, favorecendo o mesmo setor, e não interpreta negação nem intenção.

O pipeline contém `TfidfVectorizer(ngram_range=(1, 2), lowercase=True)` e `LinearSVC(random_state=42)`. O vocabulário e os pesos TF-IDF são ajustados somente no treino, evitando vazamento do teste. Os desafios não participam do treinamento. Execução registrada com Python 3.12.14 e scikit-learn 1.8.0; o arquivo `resultados_execucao.txt` contém a saída integral.

## 2. Resultados

| Método | Acurácia no teste | F1 macro dos quatro setores | Resultado |
|---|---:|---:|---|
| Regras | 80% | 0,8819 | 16 acertos, 4 revisões humanas |
| TF-IDF (1, 2) + LinearSVC | 70% | 0,6861 | 14 acertos, 6 erros |
| TF-IDF (1, 1) + LinearSVC — experimento | 70% | 0,7000 | 14 acertos, 6 erros |

Revisão humana conta como ausência de acerto automatizado: a acurácia das regras é 16/20, não 16/16. A cobertura automática foi de 80%, e o acerto entre as 16 mensagens encaminhadas automaticamente foi de 100% neste teste. Isso não garante o mesmo desempenho em mensagens futuras. O modelo principal atribuiu um setor a todas as mensagens: cobertura de 100%, com seis encaminhamentos incorretos.

### Relatório por setor — modelo principal

| Setor | Precisão | Revocação (recall) | F1 | Exemplos reais |
|---|---:|---:|---:|---:|
| Estágio | 0,7143 | 1,0000 | 0,8333 | 5 |
| Financeiro | 0,5000 | 0,4000 | 0,4444 | 5 |
| Matrícula | 0,7500 | 0,6000 | 0,6667 | 5 |
| Suporte | 0,8000 | 0,8000 | 0,8000 | 5 |

### Relatório por setor — regras

| Setor | Precisão | Revocação (recall) | F1 | Exemplos reais |
|---|---:|---:|---:|---:|
| Estágio | 1,0000 | 0,8000 | 0,8889 | 5 |
| Financeiro | 1,0000 | 1,0000 | 1,0000 | 5 |
| Matrícula | 1,0000 | 0,8000 | 0,8889 | 5 |
| Suporte | 1,0000 | 0,6000 | 0,7500 | 5 |

Precisão mede a proporção correta entre previsões de um setor; recall mede quantas mensagens reais desse setor foram recuperadas; F1 combina os dois. A média macro dá o mesmo peso a cada setor. O F1 macro de comparação inclui somente os quatro setores reais. A saída completa das regras também mostra a classe artificial de revisão, com suporte zero; a média macro desse relatório ampliado não deve ser confundida com o F1 macro dos quatro setores.

## 3. Matrizes de confusão

Linhas representam a classe real; colunas representam a previsão.

### TF-IDF (1, 2) + LinearSVC

| Real / Previsto | Estágio | Financeiro | Matrícula | Suporte |
|---|---:|---:|---:|---:|
| Estágio | 5 | 0 | 0 | 0 |
| Financeiro | 1 | 2 | 1 | 1 |
| Matrícula | 0 | 2 | 3 | 0 |
| Suporte | 1 | 0 | 0 | 4 |

### Regras

| Real / Previsto | Estágio | Financeiro | Matrícula | Suporte | Revisão humana |
|---|---:|---:|---:|---:|---:|
| Estágio | 4 | 0 | 0 | 0 | 1 |
| Financeiro | 0 | 5 | 0 | 0 | 0 |
| Matrícula | 0 | 0 | 4 | 0 | 1 |
| Suporte | 0 | 0 | 0 | 3 | 2 |

A matriz impressa pelo programa inclui ainda uma quinta linha de zeros para revisão humana, pois essa categoria é uma abstenção e não existe nos rótulos reais. A coluna de revisão é essencial para não ocultar quatro resultados não automatizados.

## 4. Falhas observadas no teste

- “Meu comprovante de pagamento foi rejeitado”: as regras acertaram Financeiro; o modelo indicou Estágio.
- “Quero solicitar trancamento do semestre”: as regras acertaram Matrícula; o modelo indicou Financeiro.
- “Não consigo recuperar a conta”: as regras solicitaram revisão por ausência de termos; o modelo acertou Suporte.
- “Minha ficha de acompanhamento não foi validada”: as regras solicitaram revisão; o modelo acertou Estágio.
- “O botão de enviar não funciona”: as regras solicitaram revisão; o modelo errou ao indicar Estágio em vez de Suporte.

Esses exemplos mostram que o modelo recupera alguns casos sem palavras-chave explícitas, mas também faz associações inadequadas. TF-IDF + SVM aprende padrões estatísticos; não demonstra compreensão confiável da intenção. As regras foram superiores neste recorte, porém dependem de manutenção manual do vocabulário.

## 5. Oito mensagens de desafio

Os desafios não possuem rótulos oficiais. As políticas abaixo são propostas de atendimento, não um gabarito nem uma avaliação quantitativa de acurácia.

| ID e mensagem | Regras | Modelo principal | O que dificultou? | Política proposta |
|---|---|---|---|---|
| C01 — Não posso acessar o boleto porque esqueci minha senha | Revisão humana | Suporte | “Boleto” e “senha” indicam setores diferentes; há empate nas regras. | Recuperar o acesso com Suporte primeiro; encaminhar ao Financeiro se a dificuldade com o boleto persistir. |
| C02 — Preciso quitar a parcela do semestre | Revisão humana | Financeiro | “Quitar” e “parcela” não estão no vocabulário das regras. | Sugestão de Financeiro, com confirmação do aluno ou revisão na triagem inicial. |
| C03 — Quero trocar de turma e revisar o vínculo | Matrícula | Matrícula | Duas solicitações relacionadas ao vínculo acadêmico. | Encaminhar para Matrícula e registrar as duas demandas. |
| C04 — Meu termo de compromisso precisa de assinatura | Estágio | Estágio | O texto não usa “estágio”, mas menciona seu documento típico. | Encaminhar ao Estágio e confirmar qual assinatura falta. |
| C05 — O sistema caiu ao enviar o relatório de estágio | Estágio | Estágio | Os dois métodos priorizam o assunto do documento; a dificuldade imediata é uma falha técnica. | Suporte investiga o envio; Estágio acompanha o relatório e eventual prazo. |
| C06 — Paguei a taxa mas a rematrícula continua pendente | Matrícula | Matrícula | Combina pagamento e situação acadêmica; as regras não reconhecem “paguei” como “pagamento”. | Triagem humana verifica compensação com Financeiro e liberação com Matrícula. |
| C07 — O aplicativo apresenta falha ao autenticar | Suporte | Suporte | “Autenticar” varia em relação ao vocabulário de autenticação, mas “aplicativo” ajuda. | Encaminhar para Suporte. |
| C08 — Preciso saber onde registrar as horas da empresa | Revisão humana | Matrícula | O contexto de estágio é possível, mas não explícito; as regras não encontram termos. | Perguntar se são horas de estágio; se confirmado, encaminhar ao Estágio. |

Concordância entre os métodos não garante correção: C05 exige atenção ao problema técnico, apesar de ambos indicarem Estágio. No C08, a previsão do modelo é questionável; sem rótulo oficial, deve-se tratá-la como sugestão a validar.

## 6. Experimento com unigramas

Alterou-se apenas `ngram_range` de `(1, 2)` para `(1, 1)`, mantendo a divisão e os demais parâmetros. O primeiro usa palavras isoladas e pares consecutivos; o segundo usa apenas palavras isoladas. Ambos obtiveram 70% de acurácia, mas o F1 macro passou de 0,6861 para 0,7000 e os erros mudaram.

| Real / Previsto — unigramas | Estágio | Financeiro | Matrícula | Suporte |
|---|---:|---:|---:|---:|
| Estágio | 4 | 0 | 1 | 0 |
| Financeiro | 0 | 3 | 1 | 1 |
| Matrícula | 0 | 2 | 3 | 0 |
| Suporte | 1 | 0 | 0 | 4 |

O teste possui apenas 20 mensagens: cada acerto representa cinco pontos percentuais. A pequena diferença de F1 não sustenta uma conclusão geral. Reutilizar o teste para escolher parâmetros introduz viés de seleção. A comparação é descritiva; a escolha deve ocorrer por validação cruzada estratificada nos 60 exemplos de treino, reservando o teste para avaliação final.

## 7. Parecer final — até 15 linhas

1. As regras obtiveram 80% de acurácia, com quatro abstenções, contra 70% do modelo principal.
2. O resultado vale para a divisão fixada de um dataset sintético pequeno.
3. As regras são explicáveis, mas dependem de termos cadastrados e não interpretam intenção.
4. Correspondências por substring podem contar termos sobrepostos e distorcer pontuações.
5. O modelo generalizou em alguns casos sem palavras-chave, mas errou seis encaminhamentos.
6. TF-IDF + LinearSVC foi ajustado somente no treino, sem vazamento do teste.
7. Os desafios não têm rótulos oficiais; não é possível calcular sua acurácia.
8. Mensagens com múltiplos setores devem passar por triagem humana ou confirmação do aluno.
9. Falhas de acesso ou envio podem exigir Suporte antes do setor ligado ao documento.
10. Concordância entre métodos não basta para automatizar mensagens ambíguas.
11. Propõe-se usar os resultados como sugestões, com revisão de divergências e ambiguidades.
12. A política é uma proposta; o protótipo não implementa detecção completa de ambiguidades.
13. LinearSVC não fornece probabilidades; uma margem de decisão não equivale a confiança calibrada.
14. Melhorias exigem mais dados revisados, validação no treino e avaliação com mensagens reais autorizadas.
15. O sistema deve apoiar a triagem, mantendo possibilidade de correção humana e registro de erros.
