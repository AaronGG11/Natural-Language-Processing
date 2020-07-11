from pickle import *
from glob import glob # para los archivos .htm
from normalization.normalizacion import *

def getCorpusNames(path, filtro):
    spath=filtro
    return glob(spath) 

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

def writeFileTXT(data,file_name):
    f = open(file_name, "w")
    for k,v in data.items():
        string = str(k) + " " + str(v) + "\n"
        f.write(string)

    f.close()