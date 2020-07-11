from archivos import *

def ocurrenciaIndividual(word, sentences):
    contador = 0

    for s in sentences:
        if(word in s): 
            contador += 1

    return contador

def ocurrenciaSimultanea(w1, w2, sentences):
    contador = 0
    total_sentences = len(sentences)

    for s in sentences:
        if((w1 in s) and (w2 in s)): 
            contador += 1

    return contador/total_sentences


def getProbabilidadesIndividuales(vocabulary,sentences):
    resultado = {}
    total_sentences = len(sentences)

    for token in vocabulary:
        resultado[token] = ocurrenciaIndividual(token,sentences)/total_sentences

    return resultado


def ocurrenciaSimultaneaSmoothing(w1, w2, sentences):
    contador = 0
    total_sentences = len(sentences)

    for s in sentences:
        if((w1 in s) and (w2 in s)): 
            contador += 1

    return (contador+0.25)/(total_sentences+1)


def getProbabilidadesIndividualesSmoothing(vocabulary,sentences):
    resultado = {}
    total_sentences = len(sentences)

    for token in vocabulary:
        resultado[token] = (ocurrenciaIndividual(token,sentences)+0.5)/(total_sentences+1)

    return resultado

def conditionalEntropy(word, sentences, individual_probabilities,vocabulary):
    import math
    import operator

    # p1 = p(w_1 = 0)
    # p2 = p(w_1 = 1)
    # p3 = p(w_2 = 0)
    # p4 = p(w_2 = 1)
    # p5 = p(w_1 = 0, w_2 = 0)
    # p6 = p(w_1 = 0, w_2 = 1)
    # p7 = p(w_1 = 1, w_2 = 0)
    # p8 = p(w_1 = 1, w_2 = 1)

    resultado = {}

    p2 = individual_probabilities[word]
    p1 = 1 - p2

    for token in vocabulary:
        p4 = individual_probabilities[token]
        p3 = 1 - p4
        p8 = ocurrenciaSimultanea(word, token, sentences)
        p7 = p2 - p8
        p5 = p3 - p7
        p6 = p4 - p8

        sumando_1 = 0
        sumando_2 = 0

        if p5 == 0 and p7 > 0:
            sumando_1 = p3*(-1)*p7*math.log2(p7)
        elif p7 == 0 and p5 > 0:
            sumando_1 = p3*(-1)*p5*math.log2(p5)
        elif p5 == 0 and p7 == 0:
            sumando_1 = 0
        elif p6 == 0 and p8 > 0:
            sumando_2 = p4*(-1)*p8*math.log2(p8)
        elif p8 == 0 and p6 > 0:
            sumando_2 = p4*(-1)*p6*math.log2(p6)
        elif p6 == 0 and p8 == 0:
            sumando_2 = 0
        else:
            sumando_1 = p3*((-1)*p5*math.log2(p5) + (-1)*p7*math.log2(p7))
            sumando_2 = p4*((-1)*p6*math.log2(p6) + (-1)*p8*math.log2(p8))

        resultado[token] = sumando_1+sumando_2
    
    return dict(sorted(resultado.items(), key=operator.itemgetter(1), reverse=True))


def conditionalEntropySmoothing(word, sentences, individual_probabilities,vocabulary):
    import math
    import operator

    # p1 = p(w_1 = 0)
    # p2 = p(w_1 = 1)
    # p3 = p(w_2 = 0)
    # p4 = p(w_2 = 1)
    # p5 = p(w_1 = 0, w_2 = 0)
    # p6 = p(w_1 = 0, w_2 = 1)
    # p7 = p(w_1 = 1, w_2 = 0)
    # p8 = p(w_1 = 1, w_2 = 1)

    resultado = {}

    p2 = individual_probabilities[word]
    p1 = 1 - p2

    for token in vocabulary:
        p4 = individual_probabilities[token]
        p3 = 1 - p4
        p8 = ocurrenciaSimultaneaSmoothing(word, token, sentences)
        p7 = p2 - p8
        p5 = p3 - p7
        p6 = p4 - p8

        sumando_1 = 0
        sumando_2 = 0

        sumando_1 = p3*((-1)*p5*math.log2(p5) + (-1)*p7*math.log2(p7))
        sumando_2 = p4*((-1)*p6*math.log2(p6) + (-1)*p8*math.log2(p8))

        resultado[token] = sumando_1+sumando_2
    
    return dict(sorted(resultado.items(), key=operator.itemgetter(1), reverse=True))


def write_conditional_entropy_word(file_name, entropia):
    f = open(file_name, "w")
    for k,v in entropia.items():
        string = str(k) + " " + str(v) + "\n"
        f.write(string)
    f.close()

def sortByTag(token,diccionario):
    tag = token.split()[1]
    result = {}
    for k,v in diccionario.items():
        aux = k
        aux1 = aux.split()[1]
        if(aux1 == tag):
            result[k] = v
    return result


############# INFORMACON MUTUA 

def mutualInformation(word, sentences, individual_probabilities,vocabulary):
    import math
    import operator

    # p1 = p(w_1 = 0)
    # p2 = p(w_1 = 1)
    # p3 = p(w_2 = 0)
    # p4 = p(w_2 = 1)
    # p5 = p(w_1 = 0, w_2 = 0)
    # p6 = p(w_1 = 0, w_2 = 1)
    # p7 = p(w_1 = 1, w_2 = 0)
    # p8 = p(w_1 = 1, w_2 = 1)

    resultado = {}

    p2 = individual_probabilities[word]
    p1 = 1 - p2

    for token in vocabulary:
        p4 = individual_probabilities[token]
        p3 = 1 - p4
        p8 = ocurrenciaSimultaneaSmoothing(word, token, sentences)
        p7 = p2 - p8
        p5 = p3 - p7
        p6 = p4 - p8

        sumando_1 = p5*math.log2((p5)/(p1*p3))
        sumando_2 = p7*math.log2((p7)/(p2*p3))
        sumando_3 = p6*math.log2((p6)/(p2*p4))
        sumando_4 = p8*math.log2((p8)/(p2*p4))

        resultado[token] = sumando_1 + sumando_2 + sumando_3 + sumando_4
    
    return dict(sorted(resultado.items(), key=operator.itemgetter(1), reverse=True))