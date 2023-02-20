import os
import nltk
import string
import pickle
import sys
import matplotlib.pyplot as plt
import re
#v is the total vocabulary size
def calcProbability(tokens,unigrams,bigrams,v):
    phraseBigrams=nltk.bigrams(tokens)
    result=0.0
    for currentBigram in phraseBigrams:
        u=0.0+unigrams.get(currentBigram[0],0.0)
        b = bigrams.get(currentBigram,0.0)
        result +=(b+1.0)/(u+v)
    return result
infile = open("englishunigrams",'rb')
unigramsEnglish=pickle.load(infile)
infile.close()
infile = open("englishbigrams",'rb')
bigramsEnglish=pickle.load(infile)
infile.close()
infile = open("frenchunigrams",'rb')
unigramsFrench=pickle.load(infile)
infile.close()
infile = open("frenchbigrams",'rb')
bigramsFrench=pickle.load(infile)
infile.close()
infile = open("italianunigrams",'rb')
unigramsItalian=pickle.load(infile)
infile.close()
infile = open("italianbigrams",'rb')
bigramsItalian=pickle.load(infile)
infile.close()
lines= open("LangId.test", "r");
solutionLines= open("LangId.sol", "r");
solutions=[]
lineCount=0
for line in solutionLines:
    solutions.append(line)
    lineCount+=1
inputs=[]
for line in lines:
    inputs.append(line)
#print(solutions)
v=len(unigramsEnglish)+len(unigramsFrench)+len(unigramsItalian)
output=open('LangId.txt','w')
invalidCount=0
for i in range(0,lineCount):
    line=inputs[i]
    text = line.lower()
    text = text.replace('\n', ' ')
    text = text.replace('--', '')
    # b.	use regex to remove all digits
    text = re.sub("\d+", "", text)
    # se regex to replace punctuation with a single space
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)
    # 3.	Tokenize the text and print the number of tokens. Format the number
    # with commas.
    tokens = nltk.word_tokenize(text)
    englishProbability=calcProbability(tokens, unigramsEnglish, bigramsEnglish, v)
    frenchProbability = calcProbability(tokens, unigramsFrench, bigramsFrench, v)
    italianProbability = calcProbability(tokens, unigramsItalian, bigramsItalian, v)
    #print(i+1,englishProbability," ",frenchProbability," ",italianProbability)
    result=''
    if(italianProbability>englishProbability) &(italianProbability>frenchProbability) :
        result=''+ str(i+1)+' Italian\n'
    elif(frenchProbability>englishProbability) &(frenchProbability>italianProbability) :
        result=''+ str(i+1)+' French\n'
    else:
        result = '' + str(i + 1) +' English\n'
    if result!=solutions[i] :
        invalidCount+=1
        print('Invalid indetification at live ',i + 1, englishProbability," ", frenchProbability, " ", italianProbability)
    output.write(result)
#c.	Compute and output your accuracy as the percentage of correctly classified
# instances in the test set
print("Accuracy as the percentage of correctly classified instances:",(100.0*(lineCount-invalidCount))/lineCount)




output.close()

