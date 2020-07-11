import nltk
from pickle import *
from nltk.corpus import cess_esp
from normalization.normalizacion import *
from glob import glob # para los archivos .htm
from archivos import *
from tagging import *
from vectorization import *

########################### LIMPIEZA  Y NORMALIZACION #############################

#saveFilePKL(getGroupNormalization("./", "*.htm"), "tokens.pkl")
tokens = getFilePKL("tokens.pkl")

##################################### TAGGER ######################################

#fname_combined_tagger = 'combined_tagger.pkl' 
#make_and_save_combined_tagger(fname_combined_tagger)
#tagger(fname_combined_tagger, tokens)
#token_tags = getFilePKL("vocabulary_tags_combine.pkl") # [word, tag]
#saveFilePKL(deleteStopWords(token_tags),"token_with_out_stopwords.pkl")
token_with_out_stopwords = getFilePKL("token_with_out_stopwords.pkl")
vocabulary = getvocabularyTags(token_with_out_stopwords)

################################### FRECUENCIA ####################################

#saveFilePKL(frecuency(vocabulary,token_with_out_stopwords),"frecuencia.pkl")
frecuencia = getFilePKL("frecuencia.pkl")

#saveFilePKL(normalizedFrecuency(frecuencia),"frecuencia_normalizada.pkl")
frecuencia_normalizada = getFilePKL("frecuencia_normalizada.pkl")

vector_tf = calcuateVectorTF(1.2,frecuencia)
vector_idf = calculateVectorIDF(frecuencia)
vector_tf_idf = calculateVectorTFIDF(vector_tf,vector_idf)

tag = 'n'
token_by_tag = getTokenByTag(vector_tf_idf,tag)
file_name = "tf_idf_by_tag_sustantivo_order.txt"
writeFileTXT(token_by_tag,file_name)