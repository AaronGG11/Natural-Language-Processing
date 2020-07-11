# 2do programa 
# por cada palabra que llamaremos topico, vamos a buscar su cobertura
# en cada articulo, es decir al final vamos a mandar una palabra y 
# nos mostrara la cobertura en todos los articulos o bien, 
# mandamos un articulo y nos dira la cobertura que tiene en todos los 
# topicos 

# 1. encontrar los topicos 
# 2. encontrar los articulos

import nltk
from pickle import *
from nltk.corpus import cess_esp
from normalization.normalizacion import *
from glob import glob # para los archivos .htm
from archivos import *
from tagging import *
from vectorization import *
from tabulate import tabulate
from titulos_articulos import * 
import re

###################################################################################
########################## OBTENER TOPICOS: PALABRAS ##############################
###################################################################################

########################### LIMPIEZA  Y NORMALIZACION #############################

#saveFilePKL(getGroupNormalization("./", "*.htm"), "tokens.pkl")
tokens = getFilePKL("tokens.pkl")

##################################### TAGGER ######################################

fname_combined_tagger = 'combined_tagger.pkl' 
#make_and_save_combined_tagger(fname_combined_tagger)
#tagger(fname_combined_tagger, tokens)
token_tags = getFilePKL("vocabulary_tags_combine.pkl") # [word, tag]
#saveFilePKL(deleteStopWords(token_tags),"token_with_out_stopwords.pkl")
token_with_out_stopwords = getFilePKL("token_with_out_stopwords.pkl")
vocabulary = getvocabularyTags(token_with_out_stopwords)


###################################################################################
####################### OBTENER DOCUMENTOS: ARTICULOS #############################
###################################################################################

tag_list = ['h3','p']
file_name = "e961024.htm"

#saveFilePKL(getHTMLByTags(file_name,tag_list),"titulos_articulos.pkl")
titulos_articulos = getFilePKL("titulos_articulos.pkl")

#saveFilePKL(tagsTitutlosArchivos(fname_combined_tagger,titulos_articulos),"titulos_articulos_tags")
titulos_articulos_tags = getFilePKL("titulos_articulos_tags")

#saveFilePKL(removeStopWordsTitlesArticles(titulos_articulos_tags),"titulos_articulos_remove_stopwords")
titulos_articulos_less_stopwords = getFilePKL("titulos_articulos_remove_stopwords")

titulos = getTitles("e961024.htm")
print(tabulate(titulos.items(),headers=["No.","Titulo"],tablefmt="presto")) 

articulos_less_stopwords = t_a_combined(titulos_articulos_less_stopwords)

###################################################################################
################################### FRECUENCIAS ###################################
###################################################################################

palabras = ['crisis', 'privatización', 'contaminación', 'política', 'economía', 'tecnología', 'televisar']
palabras_tag = getWordsWithTag(palabras,vocabulary)

#saveFilePKL(getFrecuencies(token_with_out_stopwords, vocabulary),"frecuencias_hmtl.pkl")
frecuencias = getFilePKL("frecuencias_hmtl.pkl")
frecuencias_palabras = getFrecuenciasPalabras(palabras,frecuencias)

x = calculateTopicCoverage(articulos_less_stopwords,frecuencias_palabras)

# solo para dar formato a la tabla que se imprime a continuacion
resultado = []
for k,v in x.items():
    resultado.append([k]+v)  # lo unico que hace es agregar el nuemro de docuemnto a lista junto con sus procentajes  

#print(tabulate(resultado,headers=['articulo','crisis', 'privatización', 'contaminación', 'política', 'economía', 'tecnología', 'televisa'],tablefmt="grid", numalign="center"))