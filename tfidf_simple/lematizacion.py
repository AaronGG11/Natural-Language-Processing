import nltk
from nltk.corpus import cess_esp
from pickle import *


def getContentFileLemmasGenerate():
    f=open("generate.txt",encoding='iso-8859-1')
    lines=f.readlines()

    lines=[line.strip() for line in lines]
    lemmas_and_pos={}
    for line in lines:
        if line != "\n" and line != "":
            words=line.split()
            words=[w.strip() for w in words]
            wordform=words[0]
            wordform=wordform.replace('#','')
            carac=words[-2][0]
            carac=carac.lower()
            lemmas_and_pos[wordform]=words[-1] 
    f.close()

    return lemmas_and_pos


def getLemmasGenerate(tokens):
    lemas_generate = getContentFileLemmasGenerate()  # [pal] : lemma
    resultado = [] # pal

    for token in tokens:
        if token in lemas_generate:
            resultado.append(lemas_generate[token])
        else:
            resultado.append(token)
        

    return resultado

def getTokensWithLemmas(vocabulario):
    lemmas = getContentFileLemmasGenerate()  # [pal tag] : lemma
    resultado = [] # pal, tag

    for tok in vocabulario:
        if tok in lemmas:
            resultado.append(lemmas[tok])
        else:
            resultado.append(tok)

    return resultado



def getvocabularyTags(token_tags_lemas):
    resultado = []
    for tok in token_tags_lemas: 
        resultado.append(tok[0])

    return sorted(set(resultado))




def getConextTags(tokens, ventana):# {[pal tag, pal,tag]}
    import operator
    diccionario = {} 

    for i in range(0, len(tokens)): 
        if i<ventana: # desborde izquierdo
            aux = tokens[0:i] + tokens[i+1:i+ventana+1]
        elif i>(len(tokens)-ventana): # desborde derecho
            aux = tokens[i+1:len(tokens)] + tokens[i-ventana:i]
        else:
            aux = tokens[i-ventana:i] + tokens[i+1:i+ventana+1]
        
        if tokens[i][0] in diccionario: 
            diccionario[tokens[i][0]] +=  [k[0] for k in aux]
        else: 
            diccionario[tokens[i][0]] = [k[0] for k in aux]
            


    return sorted(diccionario.items(), key=operator.itemgetter(0), reverse=False)
    
    
def getvectorFrecuenciaTag(vocabulary, contextos, fname):
    import operator
    vector = {}
    for token in vocabulary:
        vector[token] = []
        for context in contextos:
            vector[token].append(context[1].count(token))

    output = open(fname, 'wb')
    dump(vector, output, -1)
    output.close()
            
def compute_cosines_tags(token, vocabulary, vectores):
    import numpy as np
    import operator 
    cosines = {}

    vec = vectores[token]
    vec = np.array(vec)

    for token in vocabulary:
        vector = vectores[token]
        vector = np.array(vector)
        cosine = np.dot(vec, vector) / ((np.sqrt(np.sum(vec**2)))*(np.sqrt(np.sum(vector**2))))
        cosines[token] = cosine
    
    cosines = dict(sorted(cosines.items(), key=operator.itemgetter(1), reverse=True))
    return cosines


def write_tags_similiar_words_to_file(word, cosines):
    f = open("words_tags_similiar_to_" + word + ".txt", "w")
    for k,v in cosines.items():
        string = str(k) + " " + str(v) + "\n"
        f.write(string)

    f.close()

def getVocabulary(sentences):
    resultado = []
    for sentence in sentences:
        for token in sentence:
            resultado.append(token)

    return sorted(set(resultado))