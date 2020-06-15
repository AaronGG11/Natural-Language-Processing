import nltk
from pickle import *
from nltk.corpus import cess_esp

def make_and_save_combined_tagger(fname):
    patterns=[ (r'.*ar$', 'v'),
                (r'.*er$', 'v'),
                (r'.*ir$', 'v'),
                (r'.*ría$', 'v'),
                (r'.*ria$', 'v'),
                (r'.*arla$', 'v'),
                (r'.*ará$', 'v'),
                (r'.*ara$', 'v'),
                (r'.*ir$', 'v'),
                (r'.*van$', 'v'),
                (r'.*ado$', 'v'),
                (r'.*aba$', 'v'),
                (r'.*ina$', 'a'),
                (r'.*vive$', 'v'),
                (r'.*an$', 'v'),
                (r'.*ando$', 'v'),
                (r'.*endo$', 'v'),
                (r'.*iendo$', 'v'),
                (r'.*ano$', 'n'),
                (r'.*eban$', 'v'),
                (r'.*aban$', 'a'),
                (r'.*amos$', 'v'),
                (r'.*itos$', 'n'),
                (r'.*iendo$', 'v'),
                (r'.*eran$', 'v'),
                (r'.*ro$', 'a'),
                (r'.*ro$', 'a'),
                (r'.*ismo$', 'r')
             ]

    default_tagger = nltk.DefaultTagger('s')
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    combined_tagger = nltk.UnigramTagger(cess_esp.tagged_sents(), backoff=regexp_tagger)
    
    #save the trained tagger in a file
    output=open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()




def tagger(fname, sentences):
    result = []
    
    input=open(fname, 'rb')
    default_tagger = load(input)
    input.close()

    sentences_tagged = []
    for sentence in sentences:
        sentences_tagged += [default_tagger.tag(sentence)]

    for sentence in sentences_tagged:
        aux = []
        for word in sentence:
            aux += [list(word)[0] + " " + list(word)[1][0]]
        result.append(aux)

    return result




def getvocabularyTags(token_tags_lemas):
	aux = {}
	result = []

	for token in token_tags_lemas:
		aux[token[0]] = token[1]

	for k,v in aux.items():
		result.append([k,v])

	result.sort(key=lambda result: result[0])

	return result