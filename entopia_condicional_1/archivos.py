from pickle import *
from glob import glob # para los archivos .htm
from normalization.normalizacion import *

def getCorpusNames(path, filtro):
    spath=filtro
    return glob(spath) 

def getGroupNormalization(path, filtro):
	file_names = getCorpusNames(path, filtro)
	tokens_final = []

	for fn in file_names:
		tokens_final += normalizacion(fn)

	return tokens_final; 

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