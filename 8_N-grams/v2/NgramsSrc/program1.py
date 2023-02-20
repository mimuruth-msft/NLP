import os
import nltk
import string
import pickle
import sys
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter

def readFile(fileName):
    with open(fileName, 'r', encoding='utf8') as f:
        text = f.read()
        
        # and convert to lower case.
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('--', '')
        
        # b. use regex to remove all digits
        text = re.sub("\d+", "", text)
        
        # se regex to replace punctuation with a single space
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)
        
        # Tokenize the text and print the number of tokens. Format the number
        # with commas.
        tokens = nltk.word_tokenize(text)
        bigrams=nltk.bigrams(tokens)
        bigramsDict= dict(Counter(bigrams))
        unigrams = nltk.ngrams(tokens,1)
        unigramsDict = dict(Counter(unigrams))
        print(unigramsDict)
    return unigramsDict,bigramsDict
    
#i. in the main body of code, call the function 3 times for each training file, pickle the 6
#dictionaries, and save to files with appropriate names. The reason we are pickling them in
#one program and unpickling them in another is that NLTK ngrams is slow and if you put this
#all in one program, you may waste a lot of time waiting for ngrams() to finish.
unigramsDict,bigramsDict=readFile('LangId.train.English')
outfile = open("englishunigrams",'wb')
pickle.dump(unigramsDict,outfile)
outfile.close()
outfile = open("englishbigrams",'wb')
pickle.dump(bigramsDict,outfile)
outfile.close()
unigramsDict,bigramsDict=readFile('LangId.train.French')
outfile = open("frenchunigrams",'wb')
pickle.dump(unigramsDict,outfile)
outfile.close()
outfile = open("frenchbigrams",'wb')
pickle.dump(bigramsDict,outfile)
outfile.close()
unigramsDict,bigramsDict=readFile('LangId.train.Italian')
outfile = open("italianunigrams",'wb')
pickle.dump(unigramsDict,outfile)
outfile.close()
outfile = open("italianbigrams",'wb')
pickle.dump(bigramsDict,outfile)
outfile.close()


