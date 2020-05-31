from datetime import datetime
import pandas as pd
import numpy as np
import random
import math
from tabulate import tabulate
from sklearn import datasets, linear_model, metrics 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# funcion que obtiene los datos a partir de un documento .csv
# el campo fecha no lo desprecia 
def getData(name_file):
    global θs

    datos = pd.read_csv(name_file)

    matriz = np.array(datos)
    matriz = matriz[:,1:] # quitamos la columna id

    for f in matriz:
        f[0] = datetime.strptime(f[0], '%M/%d/%Y').date() # convertimos a fecha 

    minimo = min(matriz[:, 0]) # fecha mas antigua

    for f in matriz:
        f[0] = int(abs(minimo - f[0]).days) # favorecemos en numero a las fechas mas cercanas

    random.shuffle(matriz) # revolvemos los datos

    Y = matriz[:, 1] 
    X = matriz[:, [0]+[f for f in range(2,len(matriz[0]))]] # quitamos la columna de Y

    θs = np.array( [0]*(len(X[0])+1) )

    return X,Y


# funcion que a partir de la prediccion y el dato objetivo es decir el precio de las casas,
# immprime una tabla comparativa entre prediccion y realidad
def printPrediction(Y_predicciones, Y_test):
    tabla = []
    for i in range(0,len(Y_test)):
        eror_particular = abs(100-(100/(Y_test[i])*Y_predicciones[i]))
        tabla.append([Y_test[i],Y_predicciones[i],eror_particular])
    
    print(tabulate(tabla,headers=['Real','Prediccion', 'Error' ],tablefmt="grid", numalign="center"))


def main(): 
    name_file = "data.csv"
    X, Y = getData(name_file)
    X = preprocessing.scale(X) ## feature scalling

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)

    reg = linear_model.LinearRegression() # creamos el objeto de tipo regresion linear
    reg.fit(X_train, Y_train) # entrenamos el modelo 

    Y_predicciones = reg.predict(X_test)

    printPrediction(Y_predicciones, Y_test)
    print('Exactitud: {}'.format(reg.score(X_test, Y_test))) 

    
if __name__ == "__main__":
    main()
