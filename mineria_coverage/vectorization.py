def frecuency(vocabulary,token_with_out_stopwords):
    resultado = {}

    for token in vocabulary:
        resultado[token[0] + ' ' + token[1]] = token_with_out_stopwords.count(token)

    return resultado


def normalizedFrecuency(frecuency):
    import numpy as np

    resultado = {}
    suma = np.sum(np.array(list(frecuency.values())))
    for k,v in frecuency.items():
        resultado[k] = v/suma

    return resultado

def calcuateVectorTF(k,frecuencias):
    import math
    resultado = {}

    for c,v in frecuencias.items():
        numerador = (k+1)*frecuencias[c]
        denominador = frecuencias[c] + k
        resultado[c] = numerador/denominador
    
    return resultado

def calculateVectorIDF(frecuencias):
    import math
    resultado = {}

    numero_palabras = len(frecuencias)

    for k,v in frecuencias.items():
        resultado[k] = math.log2(numero_palabras/frecuencias[k])

    return resultado


def calculateVectorTFIDF(vector_tf,vector_idf):
    import numpy as np
    import operator

    tf = np.array(list(vector_tf.values()))
    idf = np.array(list(vector_idf.values()))
    llaves = list(vector_tf.keys())

    multiplicacion = list(np.multiply(tf,idf))

    return dict(sorted(dict(zip(llaves,multiplicacion)).items(), key=operator.itemgetter(1), reverse=True))

def getTokenByTag(tokens,tag):
    resultado = {}

    for k,v in tokens.items():
        aux = k.split()
        if aux[1] == tag:
            resultado[k] = v
    
    return resultado