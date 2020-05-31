import numpy as np
import os 
import re

from nltk.corpus import stopwords
from nltk import word_tokenize
from pickle import *
from bs4 import BeautifulSoup
from prettytable import PrettyTable


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


# Return a list with list, each of this lsit is a comment
def getText(PATH):
    comments = []

    stop_words = stopwords.words('spanish')

    for i in range(2, 4381):
        comment = []
        path_file = PATH + str(i) + ".review.pos"

        try:
            f = open(path_file, encoding = 'ISO-8859-1')
            lines = [line for line in f]
            f.close()
            tokens = [ word_tokenize(line) for line in lines ]
            comment = [[line[1] + ' ' + line[2][0].lower()] for line in tokens if len(line)>2]
            comments.append(comment)

        except FileNotFoundError:
            pass
    return comments


# Get ranks from each comment
def getRanks(PATH):
    ranks = []
    
    for i in range(2, 4381):
        comment = []
        path_file = PATH + str(i) + ".xml"
        
        try:
            f = open(path_file, encoding = 'ISO-8859-1')
            text = f.read()
            f.close()
            
            soup = BeautifulSoup(text, 'html.parser')
            ranks.append(int(soup.find_all('review')[0]['rank']))
            
        except:
            pass
    
    return ranks


def addPolarization(COMMENTS, DICTIONARY):
    result = []

    for comment in range(0,len(COMMENTS)):
        for word in range(0,len(COMMENTS[comment])):
            if COMMENTS[comment][word][0] in DICTIONARY:
                COMMENTS[comment][word].append(DICTIONARY[COMMENTS[comment][word][0]])
            else:
                COMMENTS[comment][word].append(float(0))

    return COMMENTS


def getPolarizationByComment(COMMENTS):
    resultado = []

    for comment in COMMENTS:
        contador = 0
        suma = 0
        
        for word in comment:
            if word[1] != 0:
                contador += 1
                suma += word[1]

        if contador == 0:
            resultado.append(0)
        else:
            resultado.append(suma/contador)

    return resultado


def getRanksWithPolarization(RANKS, POLARIZATIONS):
    resultado = {}
    ranks_dic = sorted(set(ranks)) 

    for rank in ranks_dic:
        resultado[rank] = []

    for comment in range(0,len(POLARIZATIONS)):
        resultado[RANKS[comment]].append(POLARIZATIONS[comment])

    for k,v in resultado.items():
        resultado[k] = np.array(v)

    return resultado


def getRelationRankPol(RANKPOL):
    resultado = {}

    for k,v in RANKPOL.items():
        resultado[k] = np.sum(v)/len(v)

    return resultado



if __name__ == "__main__":
    PATH_CORPUS_DIC = "/Users/aarongarcia/desktop/8_Polaridad_diccionario/ML_SentiCon/"
    PATH_CORPUS_COM = "/Users/aarongarcia/desktop/8_Polaridad_diccionario/corpusCriticasCine/"

    saveFilePKL(getDictionary(PATH_CORPUS_DIC,"es"), "dictionary.pkl")
    saveFilePKL(getText(PATH_CORPUS_COM), "comments.pkl")
    saveFilePKL(getRanks(PATH_CORPUS_COM), "ranks.pkl")

    word_tag_pol = getFilePKL("dictionary.pkl")
    comentarios = getFilePKL("comments.pkl")
    ranks = getFilePKL("ranks.pkl")

    #add polarization value to every word of every comment
    comentarios = addPolarization(comentarios,word_tag_pol)
    polaridad = getPolarizationByComment(comentarios)
    rank_pol = getRanksWithPolarization(ranks, polaridad)
    relacion = getRelationRankPol(rank_pol)

    table = PrettyTable()
    table.field_names = ['Rank','Polarization']
    for k,v in relacion.items():
        table.add_row([k,v])
    
    print(table)

    