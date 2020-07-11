#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:48:58 2020

@author: aarongarcia
"""

# Normalizacion de texto html

import nltk
import re 
from lematizacion import *
from archivos import *

# Funcion que a partir del html genera un string sin las etiquetas html
# y manda a minuscula todos y cada uno de los caracteres
def getTextHTML(file_name):
    from bs4 import BeautifulSoup
    f = open(file_name, encoding = "utf-8")
    text = f.read()
    f.close()
    
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    text = text.lower()
    return text

# Funcion que a partir del text string regresa una lista de tokens 
def getRawTokens(text_string):
    raw_tokens = nltk.Text(nltk.word_tokenize(text_string))
    # Se puede hacer sin el .Text, pero solo es clase list, 
        # de esta manera es clase nltk.text.text, que nos 
        # permite usar varias funciones utiles
    return raw_tokens

# Funcion que limpia el texto de signos de puntuacion
def getCleanTokens(raw_tokens):
    
    clean_tokens = []
    for tok in raw_tokens:
        t =[]
        for c in tok:
            if(re.match('[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]',c)):
                t.append(c)
        letter_token = ''.join(t)
        if letter_token != '':
            clean_tokens.append(letter_token)
    return clean_tokens

# Funcion que elimina los stopwords (palabras de uso comun)
def deleteStopWords(clean_tokens):
    from nltk.corpus import stopwords
    stopwords = stopwords.words("spanish")
    
    tokens_without_stopwords = []
    for tok in clean_tokens:
        if tok not in stopwords:
            tokens_without_stopwords.append(tok)
    
    return tokens_without_stopwords

def deleteStopWordsTags(clean_tokens):
    from nltk.corpus import stopwords
    stopwords = stopwords.words("spanish")
    
    tokens_without_stopwords = []
    for tok in clean_tokens:
        if tok[0] not in stopwords:
            tokens_without_stopwords.append(tok)
    
    return tokens_without_stopwords


def deleteStopWords(token_tags):
    tags = ['n', 'v', 'a', 'r', 's']
    tokens_wout_stopwords = deleteStopWordsTags(token_tags)
    
    resultado = []
    for token_tag in tokens_wout_stopwords:
        if token_tag[1]  in tags:
            resultado.append(token_tag)

    return resultado


# incluye obtener el contendio del html, limpiarlo, pasar a minuscula, y quitar stopwords
def normalizacion(file_name): 
    text_string = getTextHTML(file_name)
    raw_tokens = getRawTokens(text_string)
    clean_tokens = getCleanTokens(raw_tokens)
    tokens_lemmas = getLemmasGenerate(clean_tokens)

    return tokens_lemmas

def normalizacionString(text_string):
    text_string = text_string.lower()
    raw_tokens = getRawTokens(text_string)
    clean_tokens = getCleanTokens(raw_tokens)
    tokens_lemmas = getLemmasGenerate(clean_tokens)

    return tokens_lemmas


def getGroupNormalization(path, filtro):
    file_names = getCorpusNames(path, filtro)
    tokens_final = []

    for fn in file_names:
	    tokens_final += normalizacion(fn)

    return tokens_final; 







    