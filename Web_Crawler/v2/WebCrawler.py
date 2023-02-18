import urllib.request
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *


# This is a sample Python script.
#Build a web crawler function that starts with a URL representing a topic (a sport, your
#favorite film, a celebrity, a political issue, etc.) and outputs a list of at least 15 relevant
#URLs. The URLs can be pages within the original domain but should have a few outside
#the original domain.
def start(url):
    html = urllib.request.urlopen(url)
    htmlParse = BeautifulSoup(html, 'html.parser')
    result=set()
    for link in htmlParse.find_all("a"):
        if link.has_attr('href'):
            if link['href'].startswith("/"):
                result.add(url+link['href'][1:])
            elif (link['href'].startswith(url)):
                result.add(link['href'])
            else:
                result.add(link['href'])
    return list(result)


#Write a function to loop through your URLs and scrape all text off each page. Store each
#page’s text in its own file.
def load_pages(urls):
    i=1
    result=[]
    for url in urls:
        try:
            html = urllib.request.urlopen(url)
            htmlParse = BeautifulSoup(html, 'html.parser')
            html = ""
            for para in htmlParse.find_all("p"):
                html+=para.get_text()+"\n"
            html_file = open(str(i)+".txt", "w")
            n = html_file.write(html)
            html_file.close
            result.append(url)
            i += 1
        except:
            pass
        if i>=15:
            break
    return result
#function to clean up the text from each file. You might need to delete newlines
#and tabs first. Extract sentences with NLTK’s sentence tokenizer. Write the sentences for
#each file to a new file.
def cleanup(urls):
    i = 1
    for url in urls:
        rawText = None
        with open(str(i)+".txt", 'r') as file:
            rawText = file.read()
            file.close()
        sentences=sent_tokenize(rawText.strip(' \n\t'))
        sent_file = open(str(i) + ".sent", "w")
        for sentence in sentences:
            sent_file.write(sentence+"\n")
        sent_file.close()
        i += 1

def create_frequency_matrix(sentences):
    result = {}
    stopWords = set(stopwords.words("english"))
    for sentence in sentences:
        words = word_tokenize(sentence)
        table = {}
        for word in words:
            word = word.lower()
            if word in stopWords:
                continue
            if word in table:
                table[word] += 1
            else:
                table[word] = 1
        result[sentence] = table
    return result

def calculate_tf_matrix(freq_matrix):
    result = {}
    for sent, frequency_table in freq_matrix.items():
        table = {}
        for word, count in frequency_table.items():
            table[word] = count / len(frequency_table)
        result[sent] = table
    return result

def count_sentences_per_word(freqency_matrix):
    result = {}
    for sentence, table in freqency_matrix.items():
        for word, count in table.items():
            if word in result:
                result[word] += 1
            else:
                result[word] = 1
    return result



#a function to extract at least 25 important terms from the pages using an
#importance measure such as term frequency, or tf-idf. First, it’s a good idea to lowercase
# everything, remove stopwords and punctuation. Print the top 25-40 terms.
def select25terms(urls):
    i = 1
    sentences=[]
    print('40 terms')
    for url in urls:
        with open(str(i)+".sent", 'r') as file:
            rawText = file.readlines()
            for line in rawText:
                sentences.append(line)
            file.close()
        i +=1
    frequency_matrix=create_frequency_matrix(sentences)
    counts=count_sentences_per_word(frequency_matrix)
    sorted_counts = list(reversed(list(sorted(counts.items(), key=lambda x: x[1]))))
    for idx,item in enumerate(sorted_counts):
        if item[0].isalpha():
            print(item[0])
        if idx>=40:
            break;
    #tf_matrix=create_tf_matrix(frequency_matrix)
    #print(tf_matrix)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    url_list=start("https://www.nealspelce.com/")
    print(url_list)
    url_list=load_pages(url_list)
    cleanup(url_list)
    select25terms(url_list)

