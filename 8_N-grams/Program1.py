import pickle
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


# a. create a function with a filename as argument
def bi_uni_grams(filename):
    # b. read in the text and remove newlines
    raw_text = None
    # read unicode (UTF-8) files
    with open(filename, 'r', encoding='utf8') as file:
        raw_text = file.read()
        file.close()

    # c. tokenize the text
    tokens = word_tokenize(raw_text.strip(' \n\t').lower())
    # d. use nltk to create a bigrams list

    bigrams_list = list(ngrams(tokens, 2))

    # e. use nltk to create a unigrams list
    unigrams_list = list(ngrams(tokens, 1))

    # f. use the bigram list to create a bigram dictionary of bigrams and counts, [‘token1 token2’] ->
    # count
    bigram_dictionary = {}
    for bigram in bigrams_list:
        if bigram in bigram_dictionary:
            bigram_dictionary[bigram] += 1
        else:
            bigram_dictionary[bigram] = 1
    # g. use the unigram list to create a unigram dictionary of unigrams and counts, [‘token’] ->
    # count
    unigram_dictionary = {}
    for unigram in unigrams_list:
        if unigram[0] in unigram_dictionary:
            unigram_dictionary[unigram[0]] += 1
        else:
            unigram_dictionary[unigram[0]] = 1
    # h. return the unigram dictionary and bigram dictionary from the function
    return unigram_dictionary, bigram_dictionary


# i. in the main body of code, call the function 3 times for each training file, pickle the 6
# dictionaries, and save to files with appropriate names. The reason we are pickling them in
# one program and unpickling them in another is that NLTK ngrams is slow and if you put this
# all in one program, you may waste a lot of time waiting for ngrams() to finish.
if __name__ == '__main__':
    unigram_dictionary, bigram_dictionary = bi_uni_grams("LangId.train.English")
    with open('bigram.english.pickle', 'wb') as handle:
        pickle.dump(bigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('unigram.english.pickle', 'wb') as handle:
        pickle.dump(unigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    unigram_dictionary, bigram_dictionary = bi_uni_grams("LangId.train.French")
    with open('bigram.french.pickle', 'wb') as handle:
        pickle.dump(bigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('unigram.french.pickle', 'wb') as handle:
        pickle.dump(unigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    unigram_dictionary, bigram_dictionary = bi_uni_grams("LangId.train.Italian")
    with open('bigram.italian.pickle', 'wb') as handle:
        pickle.dump(bigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('unigram.italian.pickle', 'wb') as handle:
        pickle.dump(unigram_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
