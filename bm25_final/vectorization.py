def getvectorFrecuenciaTag(vocabulary, contextos, fname):
    import operator
    vector = {}

    for token in vocabulary:
        vector[token] = []
        for k,v in contextos.items():
            vector[token].append(v.count(token))

    output = open(fname, 'wb')
    dump(vector, output, -1)
    output.close()

def compute_cosines_tags(token, vocabulary, vectores):
    import numpy as np
    import operator 
    cosines = {}

    vec = vectores[token]
    vec = np.array(vec)

    for token in vocabulary:
        vector = vectores[token]
        vector = np.array(vector)
        cosine = np.dot(vec, vector)/((np.sqrt(np.sum(vec**2)))*(np.sqrt(np.sum(vector**2))))
        cosines[token] = cosine
    
    # cosines = dict(sorted(dict(zip(cosines.keys(),list(cosines.values()))).items(), key=operator.itemgetter(1), reverse=True))
    return dict(sorted(dict(zip(cosines.keys(),list(cosines.values()))).items(), key=operator.itemgetter(1), reverse=True))

def write_tags_similiar_words_to_file(word, cosines, file_name):
    f = open(file_name, "w")
    for k,v in cosines.items():
        string = str(k) + " " + str(v) + "\n"
        f.write(string)

    f.close()

# recibe un diccionario {token-tag, vector_numerico}
def calculateNormalization(original_frecuency):
	import numpy as np

	normalized_frecuency = {}

	for k,v in original_frecuency.items():
		normalized_frecuency[k] = []
		vector_aux = np.sum(np.array(v))
  
        #normalized_frecuency[k] = np.array(v)/vector_aux
		for n in v: 
			normalized_frecuency[k].append(float(n/vector_aux))
    
	return dict(zip(normalized_frecuency.keys(),list(normalized_frecuency.values())))


def calculateVectorTF(normalized_frecuency,ajuste_k):
	import numpy as np

	vector_tf = {}

	for k,v in normalized_frecuency.items():
		vector_frecuency = np.array(v)
  
		numerator = (ajuste_k+1)*(vector_frecuency)
		denominator = vector_frecuency + ajuste_k
		vector_tf[k] = numerator/denominator

	return dict(zip(vector_tf.keys(),list(vector_tf.values())))


def calculateDocumentFrecuency(contextos): 
# contextos -> {word-tag, conext}
# vocabulary -> word-tag

    result = []
    for k,v in contextos.items():
        contador = 0
        for c,l in contextos.items():
            if k in l:
                contador +=1
        result.append(contador)
    
    return result


def calculateVectorIDF(m,v_document_frecuency):
    import numpy as np
    
    vec = np.array(v_document_frecuency)
    result = list(np.log((m+1)/vec))
    
    return result

def calculateVectorTD_IDF(TF,IDF):
    import numpy as np
    
    vec_idf = np.array(IDF)
    
    result = {}
    
    for k,v in TF.items():
        vec_tf = np.array(v)
        result[k] = np.multiply(vec_tf,vec_idf)
        
    return dict(zip(result.keys(),list(result.values())))
