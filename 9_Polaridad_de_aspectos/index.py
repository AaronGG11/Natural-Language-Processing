import numpy as np
import os 
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pickle import *
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from lematizacion import * 
from tagging import *



def saveFilePKL(variable, fname):
    output = open(fname, 'wb')
    dump(variable, output, -1)
    output.close()
    

def getFilePKL(name_file):   
    from pickle import load
    
    output = open(name_file, "rb")     
    vectors = load(output)
    output.close()
    
    return vectors


# extraccion de oraciones
def getSenteces(PATH, OBJECT):
    sentences = []

    for comment in os.listdir(PATH+OBJECT+"/"):
        path_file = PATH+OBJECT+"/"+comment

        f = open(path_file, encoding = 'ISO-8859-1')
        text = f.read()
        f.close()

        sentences += sent_tokenize(text)

    return sentences


def cleanSentences(SENTENCES): 
    result = []

    for sentence in SENTENCES:
        aux = normalize(sentence)
        if len(aux) > 0:
            result.append(aux)

    return result

# normalizacion 
# Funcion que a partir del text string regresa una lista de tokens 
def getRawTokens(text_string):
    text_string = text_string.lower()
    raw_tokens = nltk.Text(nltk.word_tokenize(text_string))

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


# incluye obtener el contendio del html, limpiarlo, pasar a minuscula, y quitar stopwords
def normalize(TEXT): 
    raw_tokens = getRawTokens(TEXT)
    clean_tokens = getCleanTokens(raw_tokens)
    tokens_without_stopwords = deleteStopWords(clean_tokens)

    return tokens_without_stopwords


def getLemmas(SENTENCES):
    result = []
    lemas_generate = getContentFileLemmasGenerate() 

    for sentence in SENTENCES:
        result.append(get_lemmas_generate(sentence,lemas_generate))

    return result

def getText(SENTENCES):
    result = []

    for sentence in SENTENCES:
        result += sentence
    
    return result


def getFrecuency(TEXT):
    result = []
    vocabulary = set(TEXT)
    
    for word in vocabulary:
        result.append((word,TEXT.count(word)))
    
    result.sort(key=lambda tupla: tupla[1], reverse=True)
    return result


def getSentencesByAspects(SENTENCES, ASPECTS):
    result = {}
    for aspect in ASPECTS:
        result[aspect] = []

    for sentence in SENTENCES:
        for aspect in ASPECTS:
            if aspect + " " + "n" in sentence:
                result[aspect] += sentence
    
    return result


# return a list of lists, where each list has word,tag,polarization
def getDictionary(PATH, LANGUAGE):
    path_file = PATH + "senticon." + LANGUAGE + ".xml"
    word_tag_pol = {}

    stop_words = stopwords.words('spanish')

    f = open(path_file, encoding = 'UTF-8')
    text = f.read()
    f.close()

    soup = BeautifulSoup(text, 'xml')
    for word in soup.find_all("lemma"):
        if word.getText() not in stop_words: 
            word_tag_pol[str((word.getText().replace(" ", "") + ' ' + word["pos"]))] = float(word["pol"])

    return word_tag_pol


def getPolarizationByAspects(SENTENCES_ASPECTS, DICTIONARY):
    result = {}
    for k in SENTENCES_ASPECTS.keys():
        result[k] = 0
    
    for k,v in SENTENCES_ASPECTS.items():
        pol_value = 0
        counter = 0 
        for word in range(0,len(v)):
            if v[word] in DICTIONARY:
                pol_value += DICTIONARY[v[word]]
                counter += 1
        result[k] = pol_value/counter
    
    return result



if __name__ == "__main__":
    # unigram
    DIR = "/Users/aarongarcia/desktop/9_mineria_aspectos/SFU_Spanish_Review_Corpus/"
    OBJETOS = ['hoteles', 'peliculas', 'coches', 'libros', 'ordenadores', 'lavadoras', 'musica', 'moviles']
    aspectos = {}

    aspectos["hoteles"] = ["habitación", "baño", "precio", "servicio", "recepción", "desayuno", "cama"]


    oraciones = getSenteces(DIR, OBJETOS[0] + "/") # obtener todas las oraciones
    oraciones = cleanSentences(oraciones) # limpiar y tokenizar
    oraciones = getLemmas(oraciones) # lematizar

    fname_combined_tagger = 'combined_tagger.pkl' 
    #make_and_save_combined_tagger(fname_combined_tagger)
    oraciones_tag = tagger(fname_combined_tagger, oraciones)

    # ahora vamos a generar todo el texto 
    text_tag = getText(oraciones_tag)
    
    # ahora vamos a generar los aspectos
    aspectos_tag = getFrecuency(text_tag)

    # ahora vamos a obtener las oraciones que incluyen los aspectos
    sentences_aspect = getSentencesByAspects(oraciones_tag, aspectos[OBJETOS[0]])
    
    # ahora vamos a generar el diccionario de polaridad con tags 
    PATH_CORPUS_DIC = "/Users/aarongarcia/desktop/9_mineria_aspectos/ML_SentiCon/"
    #saveFilePKL(getDictionary(PATH_CORPUS_DIC,"es"), "dictionary.pkl")
    word_tag_pol = getFilePKL("dictionary.pkl")
    
    # ahora vamos a generar la polaridad de cada aspecto
    polarizacion_aspectos = getPolarizationByAspects(sentences_aspect, word_tag_pol)


    import matplotlib.pyplot as plt

    aspectos =  list(polarizacion_aspectos.keys())
    polaridd = list(polarizacion_aspectos.values())


    plt.bar(range(7), polaridd, edgecolor='black')

    plt.xticks(range(7), aspectos)
    plt.title("Polariazación en OBJETO: Hoteles")
    #plt.ylim(min(polaridd)-1, max(polaridd)+1)
    plt.show()

    table = PrettyTable()
    table.field_names = aspectos
    table.add_row(polaridd)
    
    print(table)
