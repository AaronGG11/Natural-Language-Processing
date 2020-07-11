from nltk.stem import SnowballStemmer 

def getStems(tokens): # tokens -> [word, tag]
	ss = SnowballStemmer("spanish") 
	resultado = []
	for tok in tokens:
		stem = ss.stem(tok[0])
		resultado.append(stem + " " + tok[1])

	return resultado
    

def getContexts(tokens, ventana):# {[pal tag, pal,tag]}
    import operator
    diccionario = {} 

    for i in range(0, len(tokens)): 
        if i<ventana: # desborde izquierdo
            aux = tokens[0:i] + tokens[i+1:i+ventana+1]
        elif i>(len(tokens)-ventana): # desborde derecho
            aux = tokens[i+1:len(tokens)] + tokens[i-ventana:i]
        else:
            aux = tokens[i-ventana:i] + tokens[i+1:i+ventana+1]
        
        if tokens[i] in diccionario: 
            diccionario[tokens[i]] +=  aux
        else: 
            diccionario[tokens[i]] = aux
            
    return diccionario