"""UniAssist: comparação reproduzível de regras e TF-IDF + LinearSVC.

Alunos: Marcelo Negrão, Maria Eduarda Aguiar e Mariana Martins
Professor: Neuton Melo

Execute: python uniaassist_pln.py
Os CSVs são lidos da pasta deste arquivo, independentemente do diretório atual.
"""
import csv
from collections import Counter
from pathlib import Path
import platform

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PASTA = Path(__file__).resolve().parent
REVISAO = "Revisão humana"


def ler_dataset(arquivo):
    with (PASTA / arquivo).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classificar_regras(texto):
    """Conta palavras-chave distintas presentes como trechos do texto.

    Segue o baseline do material: casefold e correspondência por substring.
    Ex.: 'disciplina' também corresponde a 'disciplinas'. Repetir o mesmo
    termo não aumenta a pontuação. Empates e ausência de termos exigem revisão.
    """
    regras = {
        "Financeiro": ["boleto", "mensalidade", "cobrança", "pagamento", "fatura"],
        "Matrícula": ["matrícula", "rematrícula", "disciplina", "turma", "trancamento"],
        "Estágio": ["estágio", "termo", "supervisor", "concedente", "convênio"],
        "Suporte": ["senha", "login", "portal", "acesso", "aplicativo", "erro"],
    }
    normalizado = texto.casefold()
    pontos = {classe: sum(termo in normalizado for termo in termos)
              for classe, termos in regras.items()}
    maior = max(pontos.values())
    vencedoras = [classe for classe, valor in pontos.items() if valor == maior]
    return vencedoras[0] if maior > 0 and len(vencedoras) == 1 else REVISAO


def criar_pipeline(ngram_range):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, lowercase=True)),
        ("svm", LinearSVC(random_state=42)),
    ])


def avaliar(nome, verdade, previsoes, classes):
    print(f"\n=== {nome} ===")
    print(f"Acurácia: {accuracy_score(verdade, previsoes):.2%}")
    # A quinta coluna mantém visíveis as abstenções do baseline.
    labels = classes + ([REVISAO] if REVISAO in previsoes else [])
    print(classification_report(verdade, previsoes, labels=labels,
                                digits=4, zero_division=0))
    print("Matriz: linhas = classe real; colunas = previsão.")
    print("Ordem:", " | ".join(labels))
    print(confusion_matrix(verdade, previsoes, labels=labels))
    if REVISAO in labels:
        print("A linha de revisão é zero: revisão é uma abstenção, não um setor real.")
        print("A média macro acima inclui a classe artificial de revisão;")
        print("para comparar setores, use o F1 macro das quatro classes abaixo.")
    report = classification_report(verdade, previsoes, labels=classes,
                                   output_dict=True, zero_division=0)
    print(f"F1 macro dos quatro setores: {report['macro avg']['f1-score']:.4f}")
    return report


def main():
    print("Python:", platform.python_version(), "| scikit-learn:", sklearn.__version__)
    registros = ler_dataset("mensagens.csv")
    textos = [r["texto"] for r in registros]
    rotulos = [r["categoria"] for r in registros]
    classes = sorted(set(rotulos))
    print("Dataset:", dict(Counter(rotulos)))
    # A mesma divisão é usada nos três métodos. TF-IDF é ajustado só no treino.
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos)
    print("Treino:", len(x_treino), dict(Counter(y_treino)))
    print("Teste:", len(x_teste), dict(Counter(y_teste)))

    modelo = criar_pipeline((1, 2))
    modelo.fit(x_treino, y_treino)
    previsoes = modelo.predict(x_teste)
    regras = [classificar_regras(t) for t in x_teste]
    avaliar("TF-IDF (1, 2) + LinearSVC", y_teste, previsoes, classes)
    avaliar("Regras", y_teste, regras, classes)
    revisoes = regras.count(REVISAO)
    print(f"Revisão humana: {revisoes}/{len(regras)} ({revisoes / len(regras):.2%})")
    print(f"Cobertura automática: {(len(regras) - revisoes) / len(regras):.2%}")
    automaticas = [(y, p) for y, p in zip(y_teste, regras) if p != REVISAO]
    if automaticas:
        print(f"Acerto entre encaminhamentos automáticos: "
              f"{sum(y == p for y, p in automaticas) / len(automaticas):.2%}")

    print("\n=== Comparação por mensagem do teste ===")
    for texto, real, regra, ml in zip(x_teste, y_teste, regras, previsoes):
        print(f"{texto}\n  Real: {real} | Regras: {regra} | Modelo: {ml}")

    print("\n=== Oito desafios: sem rótulos oficiais ===")
    desafios = ler_dataset("desafios.csv")
    pred_desafios = modelo.predict([r["texto"] for r in desafios])
    for registro, pred in zip(desafios, pred_desafios):
        print(f"{registro['id']}: {registro['texto']}\n"
              f"  Regras: {classificar_regras(registro['texto'])} | Modelo: {pred}")

    # Experimento solicitado; não selecionar hiperparâmetros pelo teste.
    unigramas = criar_pipeline((1, 1))
    unigramas.fit(x_treino, y_treino)
    avaliar("Experimento: TF-IDF (1, 1) + LinearSVC", y_teste,
            unigramas.predict(x_teste), classes)
    print("O teste reutilizado permite comparação descritiva, não seleção imparcial.")
    print("Para escolher parâmetros, usar validação cruzada apenas no treino.")


if __name__ == "__main__":
    main()
