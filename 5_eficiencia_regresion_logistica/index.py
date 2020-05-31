import numpy as np
import random
import math
from tabulate import tabulate
import nltk
import pandas as pd

θs = [] 

# leer de archivo .txt, entrega una lista de lineas 
def getText(path):
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    resultado = []

    with open(path, 'r') as f:
        lines = f.readlines()

    tags = ['a', 'n', 'r', 'v'] 
    for i in range(0,len(lines)):
        lines[i] = nltk.word_tokenize(lines[i].lower()) # minuscula y tokenizacion
        lines[i].pop(-2) # quiar la coma que separa el tipo de mensaje
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
        resultado.append(row)

    return resultado


# obtiene el vocabulario a partir de cada linea
def getvocabulary(text):
    vocabulary = []

    for line in text:
        vocabulary.extend(line)

    return sorted(set(vocabulary))


# funcion que añade la columa de 1's en el vector X en la primera posicion 
def addOnes(vector_x):
    return  np.insert(vector_x, 0, [1], axis=1)


# regresa el vector X y Y
def getData(text,vocabulary):
    np.seterr(divide='ignore', invalid='ignore')
    global θs

    X = []
    Y = []

    for line in text:
        if line[-1] == "spam":
            Y.append(1)
        else: 
            Y.append(0)
        line.pop(-1) # eliminar dato objetivo 
        row = []
        for token in vocabulary:
            row.append(line.count(token))
        X.append(np.array(row))

    X = np.array(X)
    Y = np.array(Y)

    # realizar normalizacion de frecuencia o "probabilidad"
    sumas = np.sum(X, axis=0)

    x = []
    for i in range(0,len(X)):
        x.append(np.divide(X[i],sumas))

    x = np.array(x)
    x = addOnes(x)
    θs = np.array([0]*len(x[0]))


    df = pd.DataFrame(data=x)
    df = df.replace(np.nan, 0)

    x = df.to_numpy()

    return x,Y


# funcion que separa los datos para entrenamiento y prueba
def separateSet(X,Y,test_percentage):
    data_zipped = list(zip(X, Y))
    random.shuffle(data_zipped) # revuelve la data 
    X, Y = zip(*data_zipped) # descomprime el iterable dado

    total_test = math.ceil(len(Y) * test_percentage)
    total_train = len(Y) - total_test

    X_train = X[:total_train]
    X_test = X[total_train:]
    Y_train = Y[:total_train]
    Y_test = Y[total_train:]

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


# la funcion orginal es θ^T . X, pero dado que θ es un vector, entonces solo cambia si es 
# vector columna o vector fila, y eso al vovler a transponer nos da el resultado correcto.
# DIM(θ) = 1 * N+1 y DIM(X) = M * N+1, y al hacer θ^T . X, hay error en cuanto a las dimenciones,
# por otro lado hacer: θ^T . X, genera un vector de m * N+1
def h(X):
    return [ 1/(1+np.power(np.e,-1*z)) for z in np.dot(X, np.transpose(np.array(θs)))]


# funcion de error, es decir prediccion menos realidad
def error(X, Y):
    return (h(X)-Y)


#Funcion de costo
def J(X, Y):
    predict = np.array(h(X))
    sumando_1 = np.sum(np.log(predict) * Y)
    sumando_2 = np.sum(((predict * (-1)) + 1) * ((Y * (-1)) + 1))

    return (-1)*(sumando_1 + sumando_2)/len(Y)

# funcion derivada parcial 
def J_parcial(X, Y):
    return np.dot(np.transpose(X), error(X, Y)) 


# funcion gradiente que actualiza los valores de th
def gradient(X, Y, α):
    global θs

    θs_temp = θs - (α * J_parcial(X, Y))

    θs = θs_temp

# funcion de entrenamiento con base al nuemro de iteraciones 
def train(X,Y,learning_rate,iterations):
    for i in range(0,iterations):
        gradient(X, Y, learning_rate)


# Regresa una lista de tuplas (real,prediccion)
def test(X,Y,threshold):
    Y_predicciones = h(X)
    resultados = []

    for i in range(0,len(Y)):
        if Y_predicciones[i] < threshold:
            resultados.append([Y[i],0])
        else:
            resultados.append([Y[i],1])

    return resultados


# Regresa los datos de la matriz de confusion en un diccionario
def getConfusionMatrix(test):
    true_positive = 0  # Real = prediccion = 1
    false_negative = 0 # Real = 1 & prediccion = 0
    false_positive = 0 # Real = 0 & prediccion = 1
    true_negative = 0  # Real = 0 & prediccion = 0
    
    ConfusionMatrix = {}

    for i in range(0,len(test)):
        if (test[i][0] == 1) and (test[i][1] == 1): # Real = prediccion = 1
            true_positive += 1
        if (test[i][0] == 1) and (test[i][1] == 0): # Real = 1 & prediccion = 0
            false_negative += 1
        if (test[i][0] == 0) and (test[i][1] == 1): # Real = 0 & prediccion = 1
            false_positive += 1
        if (test[i][0] == 0) and (test[i][1] == 0): # Real = 0 & prediccion = 0
            true_negative += 1

    ConfusionMatrix["true_positive"] = true_positive
    ConfusionMatrix["false_negative"] = false_negative
    ConfusionMatrix["false_positive"] = false_positive
    ConfusionMatrix["true_negative"] = true_negative

    return ConfusionMatrix


# Regresa el dato numerico de precision
def precisionMetric(confusion_matrix):
    return confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_positive"])

def precisionMetricTwo(confusion_matrix):
    numerator =     confusion_matrix["true_positive"] + confusion_matrix["true_negative"]
    denominator =   confusion_matrix["true_positive"] + confusion_matrix["false_positive"] + \
                    confusion_matrix["false_negative"] + confusion_matrix["true_negative"]
    return numerator / denominator


# Regresa el dato numerico de Recall
def recallMetric(confusion_matrix):
    return confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_negative"])


# Regresa el dato numerico F1 Score, calculando de manera interna la precision y recall
def F1Metric(confusion_matrix):
    precision = precisionMetric(confusion_matrix)
    recall = recallMetric(confusion_matrix)

    return 2*((precision * recall)/(precision + recall))


# Regresa el dato numerico F1 Score, calculando a partir de precision y recall ya calculados como parametro
def F1Metric(precision,recall):
    return 2*((precision * recall)/(precision + recall))


def main(): 
    name_file = "SMS_Spam_Corpus_big.txt"
    α = 0.1
    threshold = 0.5

    text = getText(name_file) # texto por linea ya lematizado con base a POS tag

    vocabulary = getvocabulary(text) 

    X,Y = getData(text,vocabulary)
    X_train, Y_train, X_test, Y_test = separateSet(X,Y,0.3)

    train(X_train, Y_train, α, 1000)
    prueba = test(X_test,Y_test,threshold)

    matriz_de_confusion = getConfusionMatrix(prueba)
    metrica_de_precision = precisionMetric(matriz_de_confusion)
    metrica_de_recall = recallMetric(matriz_de_confusion)
    metrica_de_f1 = F1Metric(metrica_de_precision,metrica_de_recall)

    print("\nMatriz de confusión")
    print(tabulate([["Predicción 1",matriz_de_confusion["true_positive"],matriz_de_confusion["false_positive"]],["Predicción 0",matriz_de_confusion["false_negative"],matriz_de_confusion["true_negative"]]],headers=[' ','Real 1','Real 0'],tablefmt="grid", numalign="center"))
    print("Umbral: ", threshold)
    print("Precisión: ", metrica_de_precision)
    print("Precisión other form: ", precisionMetricTwo(matriz_de_confusion))
    print("Recall: ", metrica_de_recall)
    print("F1 Score: ", metrica_de_f1)

    
if __name__ == "__main__":
    main()