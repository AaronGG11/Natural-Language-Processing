from datetime import datetime
import pandas as pd
import numpy as np
import random
import math
from tabulate import tabulate

θs = []

# funcion que agrega uno's en la primera posicion de cada fila de la matriz X
def addOnes(vector_x):
    return  np.insert(vector_x, 0, [1], axis=1)


def getData(name_file):
    global θs

    datos = pd.read_csv(name_file)

    matriz = np.array(datos)
    matriz = matriz[:,1:] # quitamos la columna id

    for f in matriz:
        f[0] = datetime.strptime(f[0], '%M/%d/%Y').date() # convertimos a fecha 

    minimo = min(matriz[:, 0]) # fecha mas antigua

    for f in matriz:
        f[0] = int(abs(minimo - f[0]).days) # favorecemos en nuemro a las fechas mas cercanas

    Y = matriz[:, 1] 
    X = matriz[:, [0]+[f for f in range(2,len(matriz[0]))]] # quitamos la columna del id

    θs = np.array( [0]*(len(X[0])+1) )

    return X,Y


def featureScaling(X):

    resultado = []

    for col in np.transpose(X):
        total_sum = np.sum(col)
        minimum = min(col)
        maximum = max(col)
        average = total_sum / len(col)
        col = (col - average) / (maximum - minimum)
        resultado.append(col)

    resultado = np.transpose(resultado)
    resultado = addOnes(resultado) # aniade 1 al principio de cada vector

    return resultado


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
    return np.dot(X, np.transpose(np.array(θs)))


def error(X, Y):
    return (h(X)-Y)


# costo 
def J(X, Y):
    return np.sum(error(X, Y)**2) / (2*len(Y))

# Es lo mismo que la suma, pero dado que hacemos producto punto, en cada multiplicacion de 
# fila por columna se hace la suma, y nuevamente al multiplicar por un vector columna, no   
# importa el orden, esto con el objetivo de ajustar las dimenciones de la multiplicacion 
def J_parcial(X, Y):
    return np.dot(np.transpose(X), error(X, Y)) / len(Y)


# funcion gradiente que actualiza los valores de th
def gradient(X, Y, α):
    global θs

    θs_temp = θs - (α * J_parcial(X, Y))

    θs = θs_temp


def train(X,Y,learning_rate,iterations):
    for i in range(0,iterations):
        if (i+1)%50 == 0:
            print(i+1, J(X, Y))
        gradient(X, Y, learning_rate)

def test(X,Y):
    Y_predicciones = h(X)

    tabla = []
    error_total = 0 
    for i in range(0,len(Y)):
        eror_particular = abs(100-(100/(Y[i])*Y_predicciones[i]))
        error_total += eror_particular
        tabla.append([Y[i],Y_predicciones[i],eror_particular])
    
    print(tabulate(tabla,headers=['Real','Prediccion', 'Error' ],tablefmt="grid", numalign="center"))
    print("Valor de la funcion de costo: ", J(X,Y))
    print("Error: ", error_total/len(Y))

def main(): 
    name_file = "data.csv"
    α = 0.1
    X, Y = getData(name_file)
    X = featureScaling(X)
    X_train, Y_train, X_test, Y_test = separateSet(X,Y,0.3)

    print(error(X_train,Y_train))
    #train(X_train, Y_train, α, 1000)
    #test(X_test,Y_test)

    
if __name__ == "__main__":
    main()


