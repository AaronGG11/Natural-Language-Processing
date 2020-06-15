import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import nltk
import operator

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


# sencences extraction
# return a dictionary {"yes":[sentences],"no":[sentences]}
def getSentences(PATH, OBJECT):
    sentences = {}
    sentences["YES"] = []
    sentences["NO"] = []

    for comment in os.listdir(PATH+OBJECT+"/"):
        path_file = PATH+OBJECT+"/"+comment

        f = open(path_file, encoding = 'ISO-8859-1')
        text = f.read()
        f.close()

        if comment.split("_")[0] == "yes":
            sentences["YES"] += sent_tokenize(text)
        else:
            sentences["NO"] += sent_tokenize(text)

    return sentences


# normalizacion 

# Funcion que elimina los stopwords (palabras de uso comun)
def deleteStopWords(clean_tokens):
    from nltk.corpus import stopwords
    stopwords = stopwords.words("spanish")
    
    tokens_without_stopwords = []
    for tok in clean_tokens:
        if tok not in stopwords:
            tokens_without_stopwords.append(tok)
    
    return tokens_without_stopwords


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


# Funcion que a partir del text string regresa una lista de tokens 
def getRawTokens(text_string):
    text_string = text_string.lower()
    raw_tokens = nltk.Text(nltk.word_tokenize(text_string))

    return raw_tokens


# incluye obtener el contendio del html, limpiarlo, pasar a minuscula, y quitar stopwords
def normalize(TEXT): 
    raw_tokens = getRawTokens(TEXT)
    clean_tokens = getCleanTokens(raw_tokens)
    tokens_without_stopwords = deleteStopWords(clean_tokens)

    return tokens_without_stopwords


def cleanSentences(SENTENCES): 
    result = []

    for sentence in SENTENCES:
        aux = normalize(sentence)
        if len(aux) > 0:
            result.append(aux)

    return result


def getLemmas(SENTENCES):
    result = []
    lemas_generate = getContentFileLemmasGenerate() 

    for sentence in SENTENCES:
        result.append(get_lemmas_generate(sentence,lemas_generate))

    return result


def getSentencesByAspects(SENTENCES, ASPECTS):
    result = {}
    for aspect in ASPECTS:
        result[aspect] = []

    for sentence in SENTENCES:
        for aspect in ASPECTS:
            if aspect in sentence:
                result[aspect] += sentence
    
    return result


def getDictionary(PATH):
    dictionary = {} #{pal: pol}

    f = open(PATH, encoding = 'ISO-8859-1')
    lines = f.readlines()
    f.close()

    for line in lines:
        aux = line.split("\t")
        dictionary[aux[0]] = aux[-1].replace("\n","")

    return dictionary


def getPolarizationByComparison(ASPECT_TOKENS, DICTIONARY):
    cont_positve = 0
    cont_negative = 0
    for token in ASPECT_TOKENS:
        if token in DICTIONARY:
            if DICTIONARY[token] == "pos":
                cont_positve += 1
            else: 
                cont_negative += 1
    
    if cont_positve > cont_negative: 
        return "positive"
    elif cont_negative > cont_positve:
        return "negative"
    else:
        return "neutral"


# Recibe una lista de aspectos, el diccioanrio de polaridad y la polaridad de eleccion
def getFrequency(ASPECTS_TOKENS, DICTIONARY, POLARITY):
    sub_dictionary = {i:ASPECTS_TOKENS.count(i) for i in ASPECTS_TOKENS if i in DICTIONARY}
    tam = len(sub_dictionary)

    sub_dictionary = {k:v/tam for k,v in sub_dictionary.items() if DICTIONARY[k] == POLARITY}
    
    if len(sub_dictionary) >= 5:
        return sorted(sub_dictionary.items(), key=operator.itemgetter(1), reverse=True)[:5]
    else: 
        return sorted(sub_dictionary.items(), key=operator.itemgetter(1), reverse=True)


def getFormatToTable(LIST):
    cadena = ""
    for word in LIST:
        aux = str(word[0]) + " " + "{:.4f}".format(float(word[1])) + "\n"
        cadena += aux
    return cadena



# Recibe el identifcador de los aspectos, la lista de listas de aspectos y el diccionario 
def printTable(ASPECTS, ASPECT_TOKENS, DICTIONARY):
    table = PrettyTable()
    table.field_names = ["Aspect/Analysis","Polarity YES","Polarity NO","(+) probability YES","(-) probability YES","(+) probability NO","(-) probability NO"]

    print("\nTable: Detailed polarity detection in reviews, OBJECT: HOTELS")
    for ASPECT in ASPECTS: # baño servicio 
        fila = []
        fila.append(ASPECT)
        fila.append(getPolarizationByComparison(ASPECT_TOKENS["YES"][ASPECT], DICTIONARY))
        fila.append(getPolarizationByComparison(ASPECT_TOKENS["NO"][ASPECT], DICTIONARY))
        fila.append(getFormatToTable(getFrequency(ASPECT_TOKENS["YES"][ASPECT],DICTIONARY,"pos")))
        fila.append(getFormatToTable(getFrequency(ASPECT_TOKENS["YES"][ASPECT],DICTIONARY,"neg")))
        fila.append(getFormatToTable(getFrequency(ASPECT_TOKENS["NO"][ASPECT],DICTIONARY,"pos")))
        fila.append(getFormatToTable(getFrequency(ASPECT_TOKENS["NO"][ASPECT],DICTIONARY,"neg")))

        table.add_row(fila)
    print(table)



if __name__ == "__main__":
    # este programa se basa en el anterior donde obtuvimos los aspectos 
    DIR = "/Users/aarongarcia/desktop/10_Polarizacion_objetos_2/SFU_Spanish_Review_Corpus/"
    DIR_DIC = "/Users/aarongarcia/desktop/10_Polarizacion_objetos_2/Spanish_sentiment_lexicon/"
    OBJETOS = ['hoteles', 'peliculas', 'coches', 'libros', 'ordenadores', 'lavadoras', 'musica', 'moviles']

    # obtenemos los aspectos con base al programa anterior 
    aspectos = {}
    aspectos["hoteles"] = ["habitación", "baño", "precio", "servicio", "recepción", "desayuno", "cama"]

    # normalizacion y lemmatizacion
    oraciones = getSentences(DIR, OBJETOS[0]) # obtener todas las oraciones
    oraciones["YES"] = cleanSentences(oraciones["YES"])
    oraciones["NO"] = cleanSentences(oraciones["NO"])
    oraciones["YES"] = getLemmas(oraciones["YES"])
    oraciones["NO"] = getLemmas(oraciones["NO"])

    # ahora vamos a etiquetas por parte de oracion
    #fname_combined_tagger = 'combined_tagger.pkl' 
    #make_and_save_combined_tagger(fname_combined_tagger) ################
    #oraciones["YES"] = tagger(fname_combined_tagger, oraciones["YES"])
    #oraciones["NO"] = tagger(fname_combined_tagger, oraciones["NO"])

    # ahora vamos a obtener las oraciones que incluyen los aspectos
    aspectos_tokens = {"YES": getSentencesByAspects(oraciones["YES"], aspectos[OBJETOS[0]]), "NO": getSentencesByAspects(oraciones["NO"], aspectos[OBJETOS[0]])}

    # ahora vamos a obtener el diccionario
    diccionario = getDictionary(DIR_DIC+"mediumStrengthLexicon.txt")
    diccionario.update(getDictionary(DIR_DIC+"fullStrengthLexicon.txt"))

    printTable(aspectos["hoteles"], aspectos_tokens, diccionario)