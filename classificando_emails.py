# coding=utf-8

texto1 = 'Se eu comprar cinco anos antecipados, eu ganho algum desconto?'
texto2 = 'O exercício 15 do curso de Java 1 está com a resposta errada. Podem conferir pf.'
texto3 = 'Existe algum curso para cuidar do marketing da minha empresa?'

import pandas as pd

classificacoes = pd.read_csv('emails.csv')
textos_puros = classificacoes['email']
textos_quebrados = textos_puros.str.lower().str.split(' ')

dicionario = set()
for lista in textos_quebrados:
	dicionario.update(lista)

print(dicionario)