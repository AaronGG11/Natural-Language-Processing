import nltk
import os
import re

from nltk.tag import StanfordPOSTagger
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
    

def getSentences(PATH, OBJECT):
    result = []

    for comment in os.listdir(PATH+OBJECT+"/"):
        path_file = PATH+OBJECT+"/"+comment

        f = open(path_file, 'r', encoding = 'ISO-8859-1', errors="ignore")
        text = f.read()
        f.close()

        text = re.sub('\n', ' ', text)
        text = text.strip()

        sentences = nltk.sent_tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 12]
        sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

        result += sentences

    return result


def filt(x):
    return x.label()=='NP'


def getTrees(sentences, grammar):
    result = []

    cp = nltk.RegexpParser(grammar)

    for sentence in sentences:
        result.append(cp.parse(sentence))

    return result


def writeFileTXT(data, file_name, fil):
    f = open(file_name, "w")

    for sentence in data:
        aux = ""
        for subtree in sentence.subtrees(filter =  fil): # Generate all subtrees
            aux += str(subtree) + "\n"
        f.write(aux)

    f.close()


def filt2(x):
    return x.label() == 'CHUNK'


### Ejecución principal
if __name__ == "__main__":
    DIR = "/Users/aarongarcia/desktop/12_Info_extraction/SFU_Spanish_Review_Corpus/"
    OBJETOS = ['hoteles', 'peliculas', 'coches', 'libros', 'ordenadores', 'lavadoras', 'musica', 'moviles']

    # obtenemos los aspectos con base al programa anterior
    aspectos = {}
    aspectos["hoteles"] = ["habitación", "baño", "precio", "servicio", "recepción", "desayuno", "cama"]

    # ahora extraemos las oraciones de los archivos fuente 
    oraciones = getSentences(DIR, OBJETOS[0]) 

    # ahora vamos a etiquetar por parte de oración 
    tagger = "/Users/aarongarcia/desktop/12_Info_extraction/stanford-tagger-4.0.0/models/spanish-ud.tagger"
    jar = "/Users/aarongarcia/desktop/12_Info_extraction/stanford-tagger-4.0.0/stanford-postagger.jar"
    reference = "https://nlp.stanford.edu/software/spanish-faq.shtml#tagset"
    etiquetador = StanfordPOSTagger(tagger,jar)
    #saveFilePKL([etiquetador.tag(oracion) for oracion in oraciones],"etiquetas.pkl")
    oraciones = getFilePKL("etiquetas.pkl")


    grammar_NP = r"""NP: 
                {<DET><NOUN><ADP><PROPN>}
                {<NOUN><NOUN><NOUN>*}
                {<NOUN><ADP><NOUN>}
                {<DET>?<ADJ><NOUN>}
                {<DET>?<NOUN><ADJ>*}
                {<DET><ADJ>}
                {<ADV><VERB><NOUN>?}
                {<DET><ADV>?<ADJ>}
                {<ADJ><CCONJ><ADV>?<ADJ>}
                {<ADJ>?<NOUN>?}
                {<PROPN><ADJ>*}
                """

    oraciones_tree = getTrees(oraciones, grammar_NP)
    #writeFileTXT(oraciones_tree, "NP.txt", filt)

    grammar_CHUNK = r"""CHUNK: 
                {<DET><NOUN><AUX><ADJ>}
                {<NOUN><ADP><NOUN>}
                {<ADJ><ADP><NOUN><ADJ>}
                {<DET>?<NOUN><VERB><ADJ>}
                {<DET><NOUN><ADJ>}
                {<ADV><VERB><NOUN>}
                {<DET><NOUN><ADV><VERB><NOUN|VERB>*}
                {<NOUN>+<ADP|DET>+<ADJ|NOUN>+}
                {<DET>?<NOUN><ADP><DET>?<NOUN|VERB|ADJ>+}
                {<DET><NOUN><ADP><DET><NOUN>}
                {<DET><NOUN><AUX><DET>?<NOUN>}
                {<PROPN><ADP><PROPN|NOUN>+}
                {<DET><NOUN><VERB><DET>+<NOUN>}
                """

    oraciones_chunk = getTrees(oraciones, grammar_CHUNK)
    writeFileTXT(oraciones_chunk, "CHUNK.txt", filt2)

    
