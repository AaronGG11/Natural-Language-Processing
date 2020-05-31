import numpy as np
import pandas as pd
import nltk
import math
import random
from prettytable import PrettyTable

# PREPROCESAMIENTO DEL TEXTO  ----------------------------------------------------------


# leer de archivo .txt
# Entrega una lista de listas, donde cada lista es un mensaje normalizado
# Tambien entrega un vector numerico que indica si el mensaje es spam [1] o es ham [0], lleva el mismo indice que la lista de listas
def getText(path):
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    texto = []
    Y = []

    with open(path, 'r') as f:
        lines = f.readlines()

    tags = ['a', 'n', 'r', 'v'] 
    for i in range(0,len(lines)):
        lines[i] = nltk.word_tokenize(lines[i].lower()) # minuscula y tokenizacion

        if lines[i][-1] == "spam":
            Y.append(1)
        else: 
            Y.append(0)
        lines[i].pop(-1) # eliminar dato objetivo 

        lines[i].pop(-1) # quiar la coma que separa el tipo de mensaje
        lines[i] = nltk.pos_tag(lines[i]) # hacer el POS tag

        row = []
        for j in range(0,len(lines[i])):
            tag = lines[i][j][1][0].lower() # convertir a minuscula el tag[0]
            if tag in tags: # lematizar de acuerdo a tags
                lines[i][j] = nltk.WordNetLemmatizer().lemmatize(lines[i][j][0],tag) 
            else: 
                lines[i][j] = lines[i][j][0]
            # quitamos stopwords
            if lines[i][j] not in stopwords:
                row.append(lines[i][j])
        texto.append(row)

    return texto, Y


# obtiene el vocabulario a partir de cada linea
def getvocabulary(text):
    vocabulary = []

    for line in text:
        vocabulary.extend(line)

    return sorted(set(vocabulary))


# VECTORIZACION TF IDF  ----------------------------------------------------------------------------------------
def getVectorFrecuency(vocabulary, sentences):
    import operator
    vector = []

    for sentence in sentences:
        aux = []
        for token in vocabulary:
            aux.append(sentence.count(token))
        
        vector.append(aux)
    
    return vector


# longitud promedio de cada documento
def averageLengthSentences(sentences):
    contador = 0

    for sentence in sentences:
        contador += len(sentence)

    return contador/len(sentences)


# Numero de documentos donde esta la palabra
def calculateDocumentFrecuency(vocabulary, sentences): 
    resultado = []

    for token in vocabulary:
        contador = 0
        for sentence in sentences:
            if token in sentence:
                contador += 1
        resultado.append(contador)    

    return resultado


def calculateIDF(sentences, vector_document_frecuency):
    import numpy as np
    return np.log((len(sentences)+1)/np.array(vector_document_frecuency))


def calculateNormalization(frecuency):
    import numpy as np
    return np.array(frecuency)/np.sum(np.array(frecuency))


def calculateVectorTF(frecuency,ajuste_k):
    import numpy as np
    resultado = []

    for vector in frecuency:
        normalized_frecuency = calculateNormalization(np.array(vector))
        resultado.append((ajuste_k+1)*normalized_frecuency/(normalized_frecuency+ajuste_k))

    return resultado


def calculateVectorTD_IDF(TF,IDF):
    import numpy as np
    resultado = []
    vec_idf = np.array(IDF)

    for vec in TF:
        resultado.append(np.multiply(vec,IDF))

    return np.array(resultado)


# CLUSTERING -----------------------------------------------------------------------------
# Funcion de cost o distorción
# c[] -> indice de centroid al que x[] fue asignado
def J(c,x,μ):
    centroides = np.array([μ[c[c_i]] for c_i in range(0,len(c))])
    norma  = np.sum(np.sqrt(np.sum((x - centroides)**2, axis=1)))
    norma = np.sum(np.sum((x - centroides)**2, axis=1))

    return norma/len(x)


def distanceVector(x_i,μ_i):
   return  np.sqrt(np.sum((x_i - μ_i)**2))


def distanceCosine(x_i,μ_i):
    return  np.dot(x_i, μ_i)/((np.sqrt(np.sum(x_i**2)))*(np.sqrt(np.sum(μ_i**2))))


def k_means(k,X_train):
    index_of_centroids = [random.randrange(0,len(X_train)) for i in range(k)] 
    μ = np.array([X_train[i] for i in index_of_centroids]) # centroiddes inciales
    data_cluster = dict(zip(np.arange(k), [[]]*k)) # {centroide,indices de datos que pertenecen al cluster}

    contador = 1
    error = 0
    while contador < 100:
        data_cluster = dict(zip(np.arange(k), [[]]*k))
        error = 0

        # Cluster assign
        for example_i in range(0,len(X_train)):
            aux_distances = []
            for centroide_i in range(0,len(μ)):
                #aux_distances.append(distanceCosine(X_train[example_i],μ[centroide_i]))
                aux_distances.append(distanceVector(X_train[example_i],μ[centroide_i]))
            
            indice = aux_distances.index(min(aux_distances))
            data_cluster[indice] = data_cluster[indice] + [example_i]
            error += aux_distances[indice]
        
        for c,v in data_cluster.items():
            aux = np.array([0]*len(X_train[0]))
            for indice in v:
                aux = aux + X_train[indice]
            μ[c] = aux/len(v)
        contador += 1

    print(" Error: ", error/len(X_train))
    return data_cluster


def main(): 
    name_file = "SMS_Spam_Corpus_big.txt"

    text, Y = getText(name_file)
    vocabulary = getvocabulary(text)

    # TF IDF
    vector_frecuency = getVectorFrecuency(vocabulary,text)
    average_length_sentences = averageLengthSentences(text)
    k = 1.2
    b = 0.75

    vector_document_frecuency = calculateDocumentFrecuency(vocabulary,text)
    vector_IDF = calculateIDF(text,vector_document_frecuency)
    matrix_TF = calculateVectorTF(vector_frecuency,k)
    matrix_tf_idf = calculateVectorTD_IDF(matrix_TF,vector_IDF),

    clusters = k_means(2,matrix_tf_idf)
    
    spam = []
    ham = []
    for c,v in clusters.items():
        aux_spam = 0
        aux_ham = 0
        for elem in v:
            if elem == 1:
                aux_spam += 1
            else:
                aux_ham += 1
        spam.append(aux_spam)
        ham.append(aux_ham)

    table = PrettyTable()
    table.field_names = ['  ','#Spam','#Ham']
    table.add_row(['Clúster 1', spam[0], ham[0]])
    table.add_row(['Clúster 2', spam[1], ham[1]])
    print(table)
    table_2 = PrettyTable()
    table_2.field_names = ['  ','#Spam','#Ham']
    table_2.add_row(['Original',Y.count(1), Y.count(0)])
    print(table_2)

if __name__ == "__main__":
    main()