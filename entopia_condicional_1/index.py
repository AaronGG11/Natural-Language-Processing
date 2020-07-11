import nltk
from pickle import *
from nltk.corpus import cess_esp
from normalization.normalizacion import *
from glob import glob # para los archivos .htm
from archivos import *
from tagging import *
from vectorization import *
from lematizacion import *
from sintagmatica import *
import numpy as np

########################### LIMPIEZA DE ORACIONES #################################
 
#saveFilePKL(getGroupNormalization("./","*.htm"),"sentences_group.pkl")
sentences_group = getFilePKL("sentences_group.pkl")

################################## LEMATIZACION ##################################

#make_and_save_lemmas_generate(sentences_group,"lemmas_generate_tokens.pkl")
tokens_lemmas = getFilePKL("lemmas_generate_tokens.pkl")

################################### TAGGING #######################################

#fname_combined_tagger = 'combined_tagger.pkl' 
#make_and_save_combined_tagger(fname_combined_tagger)
#tagger(fname_combined_tagger, tokens_lemmas)
sentences_tag = getFilePKL("vocabulary_tags_combine.pkl")
vocabulary_tags = getVocabulary(sentences_tag)

############################# RELACION SINTAGMATICA #############################

palabra = 'economía n'
saveFilePKL(getProbabilidadesIndividualesSmoothing(vocabulary_tags,sentences_tag),"probabilidades_individuales.pkl")
probabilidades_individuales = getFilePKL("probabilidades_individuales.pkl")
entropia_condicional = mutualInformation(palabra, sentences_tag, probabilidades_individuales, vocabulary_tags)
f_name = "entropia_condicional_mutual_info_" + palabra + ".txt"
write_conditional_entropy_word(f_name,entropia_condicional)









