import nltk
import random
import sys
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk.corpus import wordnet
from nltk.corpus import stopwords


# 3. Write a function to preprocess the raw text:

# b. lemmatize the tokens and use set() to make a list of unique lemmas
# c. do pos tagging on the unique lemmas and print the first 20 tagged
# d. create a list of only those lemmas that are nouns
# e. print the number of tokens (from step a) and the number of nouns (step d)


def preprocess(raw_text):
    # a. tokenize the lower-case raw text, reduce the tokens to only those that are alpha, not in
    # the NLTK stopword list, and have length > 5
    # reduce the tokens to only those that are alpha and not in the NLTK stopword list and have length > 5
    stop_words = set(stopwords.words('english'))
    filtered_tokens = list(filter(lambda word: word.isalpha() and len(word) > 5 and word not in stop_words,
                                  list(word_tokenize(raw_text.lower()))))
    number_of_tokens = len(filtered_tokens)
    # b. lemmatize the tokens and use set() to make a list of unique lemmas
    # lemmatize the filtered tokens
    lemmatizer = nltk.WordNetLemmatizer()
    token_lemma = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # make a list of unique lemmas
    unique_lemmas = set(token_lemma)

    # do pos tagging on the unique lemmas
    tagged = nltk.pos_tag(unique_lemmas)
    # d. create a list of only those lemmas that are nouns
    nouns = []
    for item in tagged:
        if item[1] == "NN":
            nouns.append(item[0])

    # e. print the number of tokens (from step a) and the number of nouns (step d)
    print("number of tokens: " + str(number_of_tokens))
    print("number of nouns: " + str(len(nouns)))
    # f. return tokens (not unique tokens) from step a., and nouns from the function
    return filtered_tokens, nouns


def print_pattern(word, letters):
    pattern = ""
    for letter in word:
        if letter in letters:
            pattern += letter + " "
        else:
            pattern += "_ "
    print(pattern)


def is_win(word, letters):
    pattern = ""
    for letter in word:
        if letter not in letters:
            return False
    return True


# guessing game function:
# a. give the user 5 points to start with; the game ends when their total score is negative, or
# they guess ‘!’ as a letter
# b. randomly choose one of the 50 words in the top 50 list (See the random numbers
# notebook in the Xtras folder of the GitHub)
# c. output to console an “underscore space” for each letter in the word
# d. ask the user for a letter
# e. if the letter is in the word, print ‘Right!’, fill in all matching letter _ with the letter and
# add 1 point to their score
# f. if the letter is not in the word, subtract 1 from their score, print ‘Sorry, guess again’
def guessing_game(words):
    pattern = ""
    print("Let's play a word guessing game!")
    word = ""
    word = words[random.randint(0, len(word))]
    letters = set()
    print_pattern(word, letters)
    letter = input("Guess a letter:")
    score = 5  # give the user 5 points to start with
    while letter != '!' and score > 0:
        if len(letter) > 1:
            print("invalid input! Score is {0}".format(score))
        elif letter in letters:
            print("Duplicate letter! Score is {0}".format(score))
        elif letter in word:
            score += 1
            print("Right! Score is {0}".format(score))
            letters.add(letter)
            if is_win(word, letters):
                print_pattern(word, letters)
                print("You solved it!")
                print("")
                print("Current score: {0}".format(score))
                print("Guess another word")
                word = words[random.randint(0, len(word))]
                letters = set()
        else:
            score -= 1
            print("Sorry, guess again. Score is {0}".format(score))
        print_pattern(word, letters)
        letter = input("Guess a letter:")
    if score <= 0:
        print("you lose!")


# 1. Send the filename to the main program
#    in a system argument
if __name__ == '__main__':
    raw_text = None
    with open(sys.argv[1], 'r') as file:
        # read the input file
        raw_text = file.read()

    # tokenize the lower-case raw text
    tokens = list(filter(lambda word: word.isalpha(),
                         list(word_tokenize(raw_text.lower()))))

    # calculate lexical diversity of the tokenized text
    print("Lexical diversity: ", (len(set(tokens)) / len(tokens)))
    tokens, nouns = preprocess(raw_text)
    # Make a dictionary of {noun:count of noun in tokens} items from the nouns and tokens lists; sort
    # the dict by count and print the 50 most common words and their counts. Save these words to a
    # list because they will be used in the guessing game.
    nounCount = {}
    for token in tokens:
        if token in nounCount:
            nounCount[token] += 1
        elif token in nouns:
            nounCount[token] = 1
    sorted_counts = list(reversed(list(sorted(nounCount.items(), key=lambda x: x[1]))))
    top50 = sorted_counts[0:50]
    print(top50)
    guessing_game([item[0] for item in top50])
