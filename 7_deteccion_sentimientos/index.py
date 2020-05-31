import numpy as np
import sklearn
import os 
import re
import mord

from nltk import word_tokenize
from nltk.corpus import stopwords
from pickle import *
from bs4 import BeautifulSoup
from prettytable import PrettyTable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score , f1_score, accuracy_score, classification_report


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


# Return the number of files that are in a path
def getNumberFiles(PATH):
    return int(len(os.listdir(DIR))/5)


# Return a list with list, each of this lsit is a comment
def getText(PATH):
    no_lines = getNumberFiles(PATH)
    comments = []
    
    counter = 0
    for i in range(2,no_lines + 1):
        comment = []
        path_file = PATH + str(i) + ".review.pos"
        
        try:
            f = open(path_file, encoding = 'ISO-8859-1')
            lines = [line for line in f]
            tokens = [ word_tokenize(line) for line in lines ]
            comment = [ line[1] for line in tokens if len(line) > 0 ]
            comments.append( comment )
            counter += 1
        except FileNotFoundError:
            pass
        
    print("Total of comments: ", counter)
    return comments


# Only remove stopwords and special characters 
# and return a list of clean comments 
def cleanText(TEXT):
    comments = []
    stop_words = stopwords.words('spanish')
    
    for comment in TEXT:
        c_comment = [token for token in comment if token not in stop_words and token.isalpha()]
        comments.append( ' '.join(c_comment))

    return comments    
    

# Get ranks from each comment
def getRanks(PATH):
    ranks = []
    no_lines = getNumberFiles(PATH)
    
    for i in range(2,no_lines + 1):
        comment = []
        path_file = PATH + str(i) + ".xml"
        
        try:
            f = open(path_file, encoding = 'ISO-8859-1')
            text = f.read()
            f.close()
            
            soup = BeautifulSoup(text, 'html.parser')
            ranks.append(int(soup.find_all('review')[0]['rank']))
            
        except FileNotFoundError:
            pass
    
    return np.array(ranks)


# Using tfidf
def getVector(COMMENTS):
    vectorizer = TfidfVectorizer(encoding='ISO-8859-1', use_idf=True, smooth_idf=True)

    return vectorizer.fit_transform(COMMENTS)


# Using countVectorizer
def getVectorC(COMMENT):
    vectorizer = CountVectorizer(encoding='ISO-8859-1')
    
    return vectorizer.fit_transform(COMMENT)
    


if __name__ == "__main__":
    DIR = "/Users/aarongarcia/desktop/7_opinion_sentimientos/corpusCriticasCine/"
    
    #saveFilePKL(getText(DIR), "comments.pkl")
    #saveFilePKL(getRanks(DIR), "ranks.pkl")
    comments = getFilePKL("comments.pkl")
    comments = cleanText(comments)
    
    X = getVector(comments).todense() 
    Y = getFilePKL("ranks.pkl")

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # separate data set
    
    clf = mord.LogisticAT()   # instaciate model
    clf.fit(x_train, y_train) # training model
    
    y_prediction = clf.predict(x_test)
    confusion_matrix = confusion_matrix(y_test, y_prediction)

    
    print(classification_report(y_test, y_prediction))

    table = PrettyTable()
    table.field_names = ['  ','Prediction 1','Prediction 2','Prediction 3','Prediction 4','Prediction 5']
    table.add_row(['Real 1', confusion_matrix[0][0],  confusion_matrix[0][1], confusion_matrix[0][2], confusion_matrix[0][3], confusion_matrix[0][4]])
    table.add_row(['Real 2', confusion_matrix[1][0],  confusion_matrix[1][1], confusion_matrix[1][2], confusion_matrix[1][3], confusion_matrix[1][4]])
    table.add_row(['Real 3', confusion_matrix[2][0],  confusion_matrix[2][1], confusion_matrix[2][2], confusion_matrix[2][3], confusion_matrix[2][4]])
    table.add_row(['Real 4', confusion_matrix[3][0],  confusion_matrix[3][1], confusion_matrix[3][2], confusion_matrix[3][3], confusion_matrix[3][4]])
    table.add_row(['Real 5', confusion_matrix[4][0],  confusion_matrix[4][1], confusion_matrix[4][2], confusion_matrix[4][3], confusion_matrix[4][4]])
    print(table)

    print(confusion_matrix)
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    