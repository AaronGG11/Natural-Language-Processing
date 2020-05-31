from datetime import datetime
import pandas as pd
import numpy as np
import random
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

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

    θs = np.array([0]*(len(X[0])+1))

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


# Funcion que separa el conjunto de datos original en 3 subconjuntos
# recibe como parametros, matriz de caracteristicas original X, 
# al vector de valor objetivo Y,
# el porcentaje de datos que se requiere para entrenamiento
# y el procentaje de datos que serequiere para pruebas
# internamente calcula el porcentaje restante para validacion 
def separateSet(X, Y, train_percentage, test_percentage):

    data_zipped = list(zip(X, Y))
    #random.shuffle(data_zipped) # revuelve la data 
    X, Y = zip(*data_zipped) # descomprime el iterable dado

    total_test = math.ceil(len(Y) * test_percentage)
    total_train = math.ceil(len(Y) * train_percentage)
    total_validate = len(Y) - total_test - total_train

    X_train = X[:total_train]
    X_validate = X[total_train:total_train+total_validate]
    X_test = X[total_train+total_validate:]
    Y_train = Y[:total_train]
    Y_validate = Y[total_train:total_train+total_validate]
    Y_test = Y[total_train+total_validate:]

    return np.array(X_train), np.array(Y_train), np.array(X_validate), np.array(Y_validate), np.array(X_test), np.array(Y_test)


# Funcion de hipotesis OK
def h(X):
    n = [k for k in range(0,12959)]
    X = X**n
    return np.dot(X, np.transpose(np.array(θs)))

# Funcion de error OK
def error(X, Y):
    return (h(X)-Y)


# costo 
def J(X, Y, λ):
    return (np.sum(error(X, Y)**2) + (λ * np.sum(θs[0:]**2))) / (2*len(Y))


# Funcion J parcial 
def J_parcial(X, Y):
    return np.dot(np.transpose(X), error(X, Y)) / len(Y)


# funcion gradiente que actualiza los valores de th
def gradient(X, Y, α, λ):
    global θs

    θs_0_temp = θs[0] - (α * J_parcial(X, Y))[0]
    θs_j_temp = θs[1:]*(1-α*λ/len(Y)) - (α * J_parcial(X, Y))[1:]

    θs[0]  = θs_0_temp
    θs[1:] = θs_j_temp


# Funcion de entrenamiento con descenso gradiente
def train(X,Y,learning_rate,iterations, λ):
    for i in range(0,iterations):
        gradient(X, Y, learning_rate, λ)


# Funcion de prueba 
def test(X,Y,λ):
    Y_predicciones = h(X)
    
    error_total = 0
    for i in range(0,len(Y)):
        eror_particular = abs(100-(100/(Y[i])*Y_predicciones[i]))
        error_total += eror_particular

    return J(X,Y,λ), error_total/len(Y)



# es como train pero aplicado a diferentes tasas de aprendizaje
def analysisRegularization(X_train,Y_train,X_validate,Y_validate,X_test,Y_test,λs,α,iterations):
    global θs
    j_train = []
    j_validate = []
    j_test = []

    for λ in λs:
        train(X_train, Y_train, α, iterations, λ)
        j_train.append(J(X_train,Y_train,λ))
        j_validate.append(J(X_validate, Y_validate, λ))
        j_test.append(J(X_test, Y_test, λ))
        θs = θs*[0]

    return np.array(j_train), np.array(j_validate), np.array(j_test)


# funcion que grafica las j de analsiis con regulariazcion 
def plotter(λ,train,validate,test):
        plt.plot(λ,train, label="J_train")
        plt.plot(λ,validate, label="J_validation")
        plt.plot(λ,test, label="J_test")

        plt.legend() 
        plt.title("Regularization parameter analysis")
        plt.xlabel("Parameter λ")
        plt.ylabel("value cost") 

        plt.show()


def main(): 
    name_file = "data.csv"
    α = 0.1
    λ = 100
    iterations = 1000
    X, Y = getData(name_file)
    X = featureScaling(X)
    
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = separateSet(X,Y,0.6,0.2)

    #λ_values = np.array([0.01,0.1,0.5,0.8,1,2,5,10,20,50,80,100,105,110,112,115,120,130,150,180,200])


    #j_train_values, j_validate_values, j_test_values = analysisRegularization(X_train,Y_train,X_validate,Y_validate,X_test,Y_test,λ_values,α,iterations)
    #plotter(λ_values,j_train_values,j_validate_values,j_test_values)

    train(X_train,Y_train,α,iterations,λ)
    train_error, train_percentage = test(X_train,Y_train,λ)
    validate_error, validate_percentage = test(X_validate,Y_validate,λ)
    test_error, test_percentage = test(X_test,Y_test,λ)

    print("λ: ",  λ)
    linea_1 = ["Error J", train_error, validate_error, test_error]
    linea_2 = ["Percentage error", train_percentage, validate_percentage, test_percentage]

    tabla = [linea_1,linea_2]

    print(tabulate(tabla,headers=['Train error','Validation error', 'Test error'] ,tablefmt="grid", numalign="center"))


    
if __name__ == "__main__":
    main()

