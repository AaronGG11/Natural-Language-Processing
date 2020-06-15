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


def get_lemmas_generate(vocabulario, lemas_generate):
    resultado = [] 

    for token in vocabulario:
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

