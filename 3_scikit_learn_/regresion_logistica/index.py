import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from tabulate import tabulate


# funcion que a partir de la prediccion y el dato objetivo es decir el precio de las casas,
# immprime una tabla comparativa entre prediccion y realidad
def printPrediction(Y_predicciones, Y_test):
    tabla = []
    for i in range(0,len(Y_test)):
        tabla.append([Y_test[i],Y_predicciones[i]])
    
    print(tabulate(tabla,headers=['Real','Prediccion'],tablefmt="grid", numalign="center"))


if __name__ == "__main__":

    df=pd.read_csv('SMSSpamCollection.txt',sep='\t',names=['Status','Message'])

    #print("Spam: ", len(df[df.Status=='spam']))
    #print("Ham: ", len(df[df.Status=='ham']))

    df.loc[df["Status"]=='ham',"Status",]=0
    df.loc[df["Status"]=='spam',"Status",]=1

    df_x=df["Message"]
    df_y=df["Status"]

    cv = TfidfVectorizer(min_df=1,stop_words='english') # escalar los datos

    df_x = cv.fit_transform(df_x).toarray()
    df_y = np.array(df_y.astype('int'))

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)

    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)

    y_predicciones = logreg.predict(x_test)

    precision = precision_score(y_test,y_predicciones)

    printPrediction(y_predicciones,y_test)
    print(precision)


