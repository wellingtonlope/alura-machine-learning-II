# coding=utf-8
# adicionando stop-words
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from collections import Counter
# to words
import nltk

classificacoes = pd.read_csv('emails.csv', encoding='utf-8')
textos_puros = classificacoes['email']
frases = textos_puros.str.lower()

# quebrar as palavras nos pontos e espaços
# nltk.download('punkt')
textos_quebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

# retirar as stopwords
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

# pegar a raiz da palavra
# nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

dicionario = set()
for lista in textos_quebrados:
	validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
	dicionario.update(validas)

total_de_palavras = len(dicionario)

tuplas = zip(dicionario, xrange(total_de_palavras))
tradutor = {palavra:indice for palavra, indice in tuplas}

def vetorizar_texto(texto, tradutor):
	vetor = [0] * len(tradutor)

	for palavra in texto:
		if len(palavra) > 0:
			raiz = stemmer.stem(palavra)
			if raiz in tradutor:
				posicao = tradutor[raiz]
				vetor[posicao] += 1

	return vetor

vetores_de_texto = [vetorizar_texto(texto, tradutor) for texto in textos_quebrados]
marcas = classificacoes['classificacao']

X = np.array(vetores_de_texto)
Y = np.array(marcas.tolist())

porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
	k = 10
	scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
	taxa_de_acerto = np.mean(scores)

	msg = 'Taxa de acerto do {}: {}'.format(nome, taxa_de_acerto)
	print(msg)
	return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict('OneVsRest', modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict('OneVsOne', modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomialNB = MultinomialNB()
resultadoMultinomilNB = fit_and_predict('MultinomialNB', modeloMultinomialNB, treino_dados, treino_marcacoes)
resultados[resultadoMultinomilNB] = modeloMultinomialNB

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict('AdaBostClassifier', modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

maximo = max(resultados)
vencedor = resultados[maximo]

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = 'Taxa de acerto do vencedor: {}'.format(taxa_de_acerto)
print(msg)

acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print('Taxa de acerto base: {}'.format(taxa_de_acerto_base))
print('Total de elementos: {}'.format(len(validacao_dados)))