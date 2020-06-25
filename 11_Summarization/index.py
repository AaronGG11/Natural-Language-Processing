### Modulos de uso común
import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import nltk
import operator
import codecs

### Utilidades
from prettytable import PrettyTable
from pickle import *

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


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_feature_matrix(documents, feature_type='frequency'):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


from scipy.sparse.linalg import svds
    
def low_rank_svd(matrix, singular_count=2):
    
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt



### Extracción de opiniones de archivo fuente

# return a dictionary {"yes":[sentences],"no":[sentences]}
def getSentences(PATH, OBJECT):
    result = {}
    result["YES"] = []
    result["NO"] = []

    for comment in os.listdir(PATH+OBJECT+"/"):
        path_file = PATH+OBJECT+"/"+comment

        f = open(path_file, 'r', encoding = 'ISO-8859-1', errors="ignore")
        text = f.read()
        f.close()

        text = re.sub('\n', ' ', text)
        text = text.strip()

        sentences = nltk.sent_tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 12]

        if comment.split("_")[0] == "yes":
            result["YES"] += sentences
        else:
            result["NO"] += sentences

    return result


def getSentencesByAspects(SENTENCES, ASPECTS):
    result = {}
    for aspect in ASPECTS:
        result[aspect] = []

    for sentence in SENTENCES:
        for aspect in ASPECTS:
            if aspect in sentence:
                result[aspect].append(" ".join(sentence))
    
    return result


### Normalizacion
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup

# Funcion que a partir del text string regresa una lista de tokens 
def getRawTokens(text_string):
    text_string = text_string.lower()
    raw_tokens = nltk.Text(nltk.word_tokenize(text_string))

    return raw_tokens


# Funcion que elimina caracteres especiales
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


# Funcion que limpia una sentencia de caracteres especiales, stopwords
# normaliza una sentencia 
def normalize(TEXT): 
    raw_tokens = getRawTokens(TEXT)
    clean_tokens = getCleanTokens(raw_tokens)
    tokens_without_stopwords = deleteStopWords(clean_tokens)

    return tokens_without_stopwords


# funcion que normaliza una lista de oraciones
def cleanSentences(SENTENCES): 
    result = []

    for sentence in SENTENCES:
        aux = normalize(sentence)
        if len(aux) > 0:
            result.append(aux)

    return result


### Etiquetar por parte de oración 

def make_and_save_combined_tagger(fname):
    patterns=[ (r'.*ar$', 'v'),
                (r'.*er$', 'v'),
                (r'.*ir$', 'v'),
                (r'.*ría$', 'v'),
                (r'.*ria$', 'v'),
                (r'.*arla$', 'v'),
                (r'.*ará$', 'v'),
                (r'.*ara$', 'v'),
                (r'.*ir$', 'v'),
                (r'.*van$', 'v'),
                (r'.*ado$', 'v'),
                (r'.*aba$', 'v'),
                (r'.*ina$', 'a'),
                (r'.*vive$', 'v'),
                (r'.*an$', 'v'),
                (r'.*ando$', 'v'),
                (r'.*endo$', 'v'),
                (r'.*iendo$', 'v'),
                (r'.*ano$', 'n'),
                (r'.*eban$', 'v'),
                (r'.*aban$', 'a'),
                (r'.*amos$', 'v'),
                (r'.*itos$', 'n'),
                (r'.*iendo$', 'v'),
                (r'.*eran$', 'v'),
                (r'.*ro$', 'a'),
                (r'.*ro$', 'a'),
                (r'.*ismo$', 'r')
             ]

    default_tagger = nltk.DefaultTagger('s')
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    combined_tagger = nltk.UnigramTagger(cess_esp.tagged_sents(), backoff=regexp_tagger)
    
    #save the trained tagger in a file
    output=open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()


def tagger(fname, sentences):
    result = []
    
    input=open(fname, 'rb')
    default_tagger = load(input)
    input.close()

    sentences_tagged = []
    for sentence in sentences:
        sentences_tagged += [default_tagger.tag(sentence)]

    for sentence in sentences_tagged:
        aux = []
        for word in sentence:
            aux += [list(word)[0] + " " + list(word)[1][0]]
        result.append(aux)

    return result


### Lematización
import spacy


def getLemmasSpacy(SENTENCES):
    nlp = spacy.load('es_core_news_sm')
    result = []

    for sentence in SENTENCES:
        result.append(" ".join([token.lemma_ for token in nlp(" ".join(sentence))]))

    return result


def getContentFileLemmasGenerate():
    f=open("generate.txt",encoding='iso-8859-1')
    lines=f.readlines()

    lines=[line.strip() for line in lines]
    lemmas_and_pos={}
    for line in lines:
        if line != "\n" and line != "":
            words=line.split()
            words=[w.strip() for w in words]
            wordform = words[0]
            wordform = wordform.replace('#','')
            tag = words[-2][0]
            tag = tag.lower()
            lemmas_and_pos[wordform + " " + tag] = words[-1] 
    f.close()

    return lemmas_and_pos


def getLemmasGenerate(sentence, lemmas_generate, lemmas_assistant):
    resultado = [] 

    for token in sentence:
        if token in lemmas_generate:
            resultado.append(lemmas_generate[token])
        elif token.split()[0] in lemmas_assistant:
            resultado.append(lemmas_assistant[token.split()[0]])
        else:
            resultado.append(token.split()[0])

    return resultado


def getAssistantLemmas(LEMMAS_GENERATE):
    result = {}

    for k,v in LEMMAS_GENERATE.items():
        result[ k.split()[0]] = v

    return result


def getLemmas(SENTENCES):
    result = []
    lemmas_generate = getContentFileLemmasGenerate() # word_normal tag: word_lemma
    lemmas_assistant = getAssistantLemmas(lemmas_generate) # word_normal : word_lemma

    for sentence in SENTENCES:
        result.append(getLemmasGenerate(sentence, lemmas_generate, lemmas_assistant))

    return result



### Gensim
from gensim.summarization import summarize, keywords

def addPointToSentences(SENTENCES):
    return [sentence + " ." for sentence in SENTENCES]


def removePoint(SENTENCES):
    return [sentence[:-2] for sentence in SENTENCES]


def zipTokensInSentence(SENTENCES):
    return [" ".join(sentence) for sentence in SENTENCES]


def text_summarization_gensim(text, summary_ratio):
    result = []
    summary = summarize(text, split=True, ratio=summary_ratio)

    for sentence in summary:
        result.append(sentence)

    return result


def getConcordanceBetweenSentences(ORIGINAL_SENTENCES, MODIFY_SENTENCES, SUMMARY_SENTENCES):
    result = ""

    for sentence in SUMMARY_SENTENCES:
        if sentence in MODIFY_SENTENCES:
            result += " " + ORIGINAL_SENTENCES[MODIFY_SENTENCES.index(sentence)]
    
    return result.strip()


def gensim_text_summarizer(documents, sentences, percent_sentences):
    resumen = text_summarization_gensim(" ".join(addPointToSentences(documents)), percent_sentences)
    aux = removePoint(resumen)
    
    return getConcordanceBetweenSentences(sentences, documents, aux)



### Latent Semantic Analysis (LSA)
def lsa_text_summarizer(documents, sentences, num_sentences, num_topics, feature_type, sv_threshold):
    result = ""
    vec, dt_matrix = build_feature_matrix(documents, feature_type=feature_type)

    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)

    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()
    
    for index in top_sentence_indices:
        result += " " + sentences[index]

    return result.strip()


def getOriginalSentencesByAspect(ORIGINAL_SENTENCES, MODIFY_SENTENCES, ASPECT_SENTENCES, ASPECTS, OPINION_TYPE):
    result = {}
    for aspect in ASPECTS:
        result[aspect] = []

    for aspect, sentences in ASPECT_SENTENCES[OPINION_TYPE].items():
        aux = []
        for sentence in sentences:
            aux.append(ORIGINAL_SENTENCES[OPINION_TYPE][zipTokensInSentence(MODIFY_SENTENCES[OPINION_TYPE]).index(sentence)])
        result[aspect] = aux

    return result

### TextRank
import networkx


def textrank_text_summarizer(documents, sentences, num_sentences, feature_type):
    result = ""

    vec, dt_matrix = build_feature_matrix(documents, feature_type)
    similarity_matrix = (dt_matrix * dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)

    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)

    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()

    for index in top_sentence_indices:
        result += " " + sentences[index]
    
    return result.strip()



### Ejecución principal
if __name__ == "__main__":
    DIR = "/Users/aarongarcia/desktop/11_Summarization/SFU_Spanish_Review_Corpus/"
    OBJETOS = ['hoteles', 'peliculas', 'coches', 'libros', 'ordenadores', 'lavadoras', 'musica', 'moviles']

    # obtenemos los aspectos con base al programa anterior
    aspectos = {}
    aspectos["hoteles"] = ["habitación", "baño", "precio", "servicio", "recepción", "desayuno", "cama"]

    # ahora extraemos las oraciones de los archivos fuente y las normalizamos
    oraciones_original = getSentences(DIR, OBJETOS[0]) # obtener todas las oraciones originales
    oraciones_modificadas = {"YES": cleanSentences(oraciones_original["YES"]), "NO": cleanSentences(oraciones_original["NO"])}

    # ahora vamos a etiquetas por parte de oración
    fname_combined_tagger = 'combined_tagger.pkl'
    #make_and_save_combined_tagger(fname_combined_tagger) ################
    oraciones_modificadas = {"YES": tagger(fname_combined_tagger, oraciones_modificadas["YES"]), "NO": tagger(fname_combined_tagger, oraciones_modificadas["NO"])}

    # ahora lematizamos con base al tagging y generate
    oraciones_modificadas = {"YES": getLemmas(oraciones_modificadas["YES"]), "NO": getLemmas(oraciones_modificadas["NO"])}

    # ahora vamos a obtener las oraciones que incluyen los aspectos
    oraciones_aspectos = {"YES": getSentencesByAspects(oraciones_modificadas["YES"], aspectos[OBJETOS[0]]), "NO": getSentencesByAspects(oraciones_modificadas["NO"], aspectos[OBJETOS[0]])}
    oraciones_aspectos_original = {"YES": getOriginalSentencesByAspect(oraciones_original, oraciones_modificadas, oraciones_aspectos, aspectos[OBJETOS[0]], "YES"), "NO": getOriginalSentencesByAspect(oraciones_original, oraciones_modificadas, oraciones_aspectos, aspectos[OBJETOS[0]], "NO")}

    # ahora vamos a generar e imprimir resultados 
    for aspect in aspectos[OBJETOS[0]]:
        for opinion_type in ["YES", "NO"]:
            # obtener resumen GENSIM 
            print("\n",aspect, opinion_type,"gensim")
            print(gensim_text_summarizer(oraciones_aspectos[opinion_type][aspect], oraciones_aspectos_original[opinion_type][aspect], 0.3))

            # obtener resumen LSA
            print("\n",aspect, opinion_type,"lsa")
            print(lsa_text_summarizer(oraciones_aspectos[opinion_type][aspect], oraciones_aspectos_original[opinion_type][aspect], 3, 1, 'frequency', 0.5))

            # obtener resumen TextRank
            print("\n",aspect, opinion_type,"textrank")
            print(textrank_text_summarizer(oraciones_aspectos[opinion_type][aspect], oraciones_aspectos_original[opinion_type][aspect], 3, 'tfidf'))

        print("\n\n\n")