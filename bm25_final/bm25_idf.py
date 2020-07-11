import math

def longitudPromedioContextos(contextos):
    contador = 0

    for k,v in contextos.items():
        contador += len(v)

    return contador/len(contextos)

def sumarContextosVector(contextos_vector):
    import numpy as np

    result = {}
    for k,v in contextos_vector.items():
        result[k] = np.sum(np.array(contextos_vector[k]))
    
    return result

def calculateBM25(k,b,contextos,vector_suma,longitud_promedio_contextos):
    pre_result = {}

    vocabulary = contextos.keys()
    
    for v in vocabulary:
        pre_result[v] = []
        context_word = contextos[v]
        for m in vocabulary:
            numerador = (k+1)*context_word.count(m)
            denominador = context_word.count(m) + k*(1-b+b*(vector_suma[v]/longitud_promedio_contextos))
            aux = numerador/denominador
            pre_result[v].append(aux) 
    return pre_result

def sumaBM25(vector_bm25):
    import numpy as np
    resultado = {}

    for k,v in vector_bm25.items():
        resultado[k] = np.sum(np.array(v))
    
    return resultado




def calculateDocumentFrecuency(contextos): 
    result = {}
    for k in contextos.keys():
        contador = 0
        for v in contextos.values():
            if k in v:
                contador  += 1
        result[k] = contador

    return result


def calculateIDF(m,vector_document_frecuency):
    resultado = []

    for k,v in vector_document_frecuency.items():
        resultado.append(math.log((m+1)/vector_document_frecuency[k]))
    
    return resultado

def calculateVectorBM25_IDF(bm25,IDF,sumaBM25):
    import numpy as np
    
    resultado = {}

    for k in bm25.keys():
        aux = np.array(np.multiply(bm25[k],np.array(IDF)))/sumaBM25[k]
        resultado[k] = aux
    
    return resultado

def imprimirDiccionario(dic):
    for k,v in dic.items():
        print(k,v)