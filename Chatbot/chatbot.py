# -*- coding: utf-8 -*-
"""
@author: Michael M
"""

import os
import random
import string

import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

# nltk.download() # for downloading packages


# Variables

#########################################
# ************** CHATBOT PROJECT ************
# Building a simple Bot from scratch
# Inspired by https://worldwritable.com/chatbot-fundamentals-572db46667a7
# Liz lady
# Predefine some words grouped in terms of parts of speech
# Eg nouns, pronouns, verbs and prepositions
# Create a pattern of how a greeting, request,comment and appreciation would take
# Prepare a response generator returning facts from knowledge base
# Determine and Save user preferences

# '''The chat bot mimicss a car dealer site where visitors can ask for a price of a car
# or check the beautiful brands
# '''
##########################################


# Items to track a conversation

# Data from knowledge base is loaded here, for search and ease of access

botname = 'ChatBot'
current_context = 0
current_subjects_by_me = 0
current_subjects_by_user = 0
lemmer = nltk.stem.WordNetLemmatizer()

bot_input = "ChatBot  :>>  "

build_response = ""
car_type = []
car_names = []
car_brands = []
year_ = []
all_cars = {}

# Top terms
user_querries = []
quiz_indicators = ['what does it', 'how much does', 'where can i', 'how much does a', 'when can i', 'how do i',
                   'what is the cost', 'what is the price']
greetingcon = ["By the way", "well", "alright!", "hold on", "great", "Wow"]
top_terms = ["mercedes", "jeep", "audi", "bmw", "honda", "suv", "sedan", "coupe", "hatch", "wagon"]
usermodel = {}  # holds data about user saved on file

# Track conversation
keep_going = True
current_user = ""

# Constants for expected responses to track when user has changed expected input
USER_NAME = "uName"
CONFIRM_NAME = "confirmName"
ITEM_NAME = "itemName"
GENERAL = "general"
OPINION = "opinion"
expected_response = USER_NAME
prompt_subject = []
intent = ["looking", "want", "buy", "interested", "wish", ]
inquiry = ["do you have", "Can i", "what about", "how much is"]
exitmsg = ["bye", "thanks for your time", "thanks a bunch", "see you later", "nothing", "nothing more", "see you"]

name_prompt = ["What is your first name?", "Welcome, please enter your first Name?",
               "Before we start, can you tell me your name", "Welcome! may I know your name?"]
contexts = ["purchase", "Inquiry", "intro", "exit"],
logicals = ["or", "not", "and"]
negatives = ["don't like", "hate", "no"]
persons = ["I", "we", "they", "you", "us", "am", "it", ]
possesive = ["my", "our", "their"]
positives = ["like", "Prefer", "Desire"]
greetings = ["Hi!", "Hey!", "whats' up", "good morning", "good evening", "hello", "how are you doing"]
adjectives = ["good", "big", "nice", "cheap", "expensive"]
others = ["name", "address"]
days = ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday"]
data_prompt = ["How can I help you?", "What are you looking for?", "What caI do for you?"]
data_appreciate = ["Thank you", "grateful", "great", "Good", "Thanks", "welcome", "come Again"]
data_greetings = ["hi there", "hey ?", "morning", "hello", "what's up!", "good to see you"]
agree_statements = ["yeah", "that's right", "ok", "right", "yes", "yep", "true", "yea", "y", "correct"]
disagree_statements = ["no", "nop", "argh", "false", "nothing", "no thanks", "not"]
prompt = ["Alright, ask me about cars prices. Just mention a car",
          "We sell a wide range of vehicles. What kind of vehicle would you say you are searching for or require data about?",
          "Ask me some information about your favorite vehicle.", "Which sort of vehicle would you say you are searching for?",
          "Enter a vehicle make and model."]
positiveresp = ["Alright", "We found", "We got", "how about"]
negative_response = ["Oh, sorry", "We don't have"]

common_phrases = ["name is", "cheapest", "am ", "cheap", "don't like", "hate", "best", "bye", "no thanks", "low cost",
                  "cheapest car", "desire", "prefer", "nothing"]

response_intro = ["Oook, Just wait a minute", "How about this:", "That was unexpected :-):",
                  "Great, there you go", "Wait! Before you goIf you have to leave, Say bye :-)"]

response_close = [":>> Keep rolling, we have numerous vehicles. Type another vehicle name! ",
                  " :>>  If that is irrelevant, please elaboarate :-( or just type the name of another car",
                  ":>> try body styles also such as sedan, suv, coupe, hatchback, convertible etc", "By the way, sedans are cheaper!"]

attributes = ["size", "capacity", "type", "style", "model", "petrol", "diesel", "miles", "km", "seater", "litres"]
subjects = ["Location", "Model", "Year", "Drive", "Engine", "Body", "Engine", "mileage", "fuel", "seat", "capacity",
            "color"]
actions = ["know", "get", "need", "buy", "ship", "looking", "want", "desire", "covet"]
objects = ["car", "truck", "pickup", "lorry", "suv", "sedan", "hatch", "coupe", "hatchback", "station", "wagon",
           "pick up", "bus", "motorcycle", "crossover", "van", "minivan"]
greeting_responses = ['hi', 'hello', 'sup', 'am good', 'morning']
exit_indicators = ['bye', 'nothing more', 'see you', 'thanks', 'no thank you', 'ok bye', 'ok thanks']
exit_msg = ['nice talking to you, bye', 'thanks, take care', 'ok alright, bye!', 'welcome again']
welcomemsg = ['Great, ask me about cars.', 'what can I do for you', 'how can i help you?']

''' This part describes and shows the data model the bot adapts.
'''

sent_tokens = None
word_tokens = None

# Knowledgebase a list of lines
k_base = []
responses = []


def handle_greeting():
    print(random.choice(greetings))


def handle_intro_phrase():
    print(random.choice(prompt))


def prompt_name():
    print(random.choice(name_prompt))


def handle_exit():
    print(random.choice(exitmsg))


def handle_response_intr():
    pass
    # print(bot_input + random.choice(response_intro))


def handle_response_end():
    print(random.choice(response_close))


# Save the user preferences. Learn the user trends
def update_user(uname, ulike, uhate):
    global usermodel
    global current_user
    current_user = uname
    if uname in usermodel:
        print("old user")
        data = usermodel[uname]
        likes = []
        dislikes = []
        if "likes" in data:
            likes = data["likes"]
        if "dislikes" in data:
            dislikes = data["dislikes"]
        if ulike != "":
            likes.append(ulike)
        if uhate != "":
            dislikes.append(uhate)
        newdt = {}
        newdt["likes"] = likes
        newdt["dislikes"] = dislikes
        usermodel[uname] = newdt
    else:
        # print("new user")
        newdt = {}
        likes = []
        dislikes = []
        if ulike != "":
            likes.append(ulike)
            newdt["likes"] = likes
        if uhate != "":
            dislikes.append(uhate)
            newdt["dislikes"] = dislikes
        if uname != "":
            usermodel[uname] = newdt
            # print(newdt)


def save_model():
    data = ""
    for key in usermodel:
        likes = ','.join(usermodel[key]["likes"])
        dislikes = ','.join(usermodel[key]["dislikes"])
        line = key + "|" + likes + "|" + dislikes + "\n"
        data += line
    f = open(os.path.join("umodel.txt"), "w+", encoding='utf-8')
    f.write(data)
    print('User Model Saved')


# Keep track of users
def user_model_engine():
    # Create a dictionary of dictionaries to read values from file
    thefile = open('umodel.txt', 'r', encoding="utf8")
    for line in thefile:
        if not line.isspace():
            data = {}
            n = line.split("|")
            uname = n[0]
            user_likes = n[1].strip().split(",")

            user_dislikes = n[2].strip().split(",")
            data["likes"] = user_likes
            data["dislikes"] = user_dislikes
            usermodel[uname] = data


# Advanced NLP processing
########################################
def prepare_knowledge_base():
    filepath = 'cleaned/knowledgebase.txt'
    with open(filepath, encoding='utf8') as fp:
        line = fp.readline()
        k_base.append(line)


#  Perform initialization
def initialize_bot():
    global sent_tokens
    global word_tokens
    # print("Initialize Knowledgebase")
    f = open('cleaned/knowledgebase.txt', 'r', errors='ignore')
    raw = f.read()
    raw = raw.lower()  # converts to lowercase
    sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
    word_tokens = nltk.word_tokenize(raw)  # converts to list of words

    f.close()


def lem_tokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]


def Lem_normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=Lem_normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "Well,! I don't get you well. please elaborate!"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# Check if the common phrases are in the sentence

def process_common_phrase(stmt):
    handled = False
    global keep_going
    if stmt.lower() in exit_indicators:
        handled = True
        keep_going = False
        print(bot_input + "Thanks for your time. Bye")
    elif common_phrases[4] in stmt:
        z = stmt[stmt.find(common_phrases[4]) + len(common_phrases[4]) + 1:]
        hated = z[0:z.find(' ')]
        print(bot_input + "Ooh, you hate," + hated + " we won't show you those.")
        # Record the user preference - Learn what the user hates
        update_user(current_user, "", hated)
        handled = True

    elif common_phrases[5] in stmt:
        z = stmt[stmt.find(common_phrases[5]) + len(common_phrases[5]) + 1:]
        hated = z[0:z.find(' ')]
        print(bot_input + "Ooh, you hate " + hated + "\nWe wont show you those.")
        update_user(current_user, "", hated)
        handled = True
    elif common_phrases[11] in stmt:
        z = stmt[stmt.find(common_phrases[11]) + len(common_phrases[11]) + 1:]
        loved = z[0:z.find(' ')]
        print(bot_input + "Wow, you love ," + loved + " Alright! ")
        update_user(current_user, loved, "")
    elif common_phrases[12] in stmt:
        z = stmt[stmt.find(common_phrases[12]) + len(common_phrases[12]) + 1:]
        loved = z[0:z.find(' ')]
        print(bot_input + "Wow, you love ," + loved + " Cool!")
        update_user(current_user, loved, "")
    elif common_phrases[7] in stmt:
        handled = True
        print(bot_input + " See You, Nice time")
        keep_going = False
    elif common_phrases[8] in stmt or common_phrases[13] in stmt:
        handled = True
        keep_going = False
        print(bot_input + "Thanks for your time. Bye")

    return handled


# Process input
##########################################
def process_input(n):
    global word_tokens
    global responses
    user_response = checkSpelling(n)
    # If phrase has not been processed
    if not process_common_phrase(n):
        compile_response(n)
        sent_tokens.append(user_response)
        word_tokens = word_tokens + nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        handle_response_intr()
        print(response(final_words))
        for item in responses:
            print("******************************")
            print(item)
            print("******************************")
        del responses[:]
        sent_tokens.remove(user_response)
        handle_response_end()


# Generate an elaborate response using set theory
def compile_response(input):
    # A list with all important word
    global responses
    m = set()
    n = check_stop_words(input)
    if len(n) > 0:
        m = set(n)
    for line in k_base:
        count = 0
        # make the line  a set and see if the other set is in it
        # If in add to responses count three lines and break
        lineset = set(line)
        if m.issubset(lineset):
            count += 1
            responses.append(line)
            if count == 2:
                break


# Start execution
##########################################
def start_conversation():
    while keep_going:
        newParser(get_input().lower())


def get_input():
    return input(current_user + "  :>> ")


# Method 1: generates Knowledgebase From a the cleaned files...
def generate_data():
    data = ""
    for filename in os.listdir('cleaned/'):
        print(filename)
        with open('cleaned/' + filename, 'r', encoding='utf8') as myfile:
            data += myfile.read()
    print()
    f = open(os.path.join("pages/kb.txt"), "w+", encoding='utf-8')
    f.write(data)
    print('Done!')


# Generate BOT Data
##########################################
def prepare_bot_data():
    initialize_bot()
    prepare_knowledge_base()


# SPELL CHECKER
########################################
def checkSpelling(myline):
    spell = SpellChecker()
    # print(line)
    line = myline
    # find those words that may be misspelled
    line = line.replace(",", "")
    line = line.replace(".", "")
    line = line.replace(";", "")
    line = line.replace("'", "")
    line = line.replace("\"", "")
    line = line.replace("?", "")

    allwords = line.split()
    misspelled = spell.unknown(line.split())
    nwords = []
    for word in misspelled:
        nword = spell.correction(word)
        nwords.append(nword)
        allwords.append(nword)
    if len(nwords) > 0:
        print("Spell Check: >> Corrected Words")
        print(nwords)
    return " ".join(allwords)


# Check stop words
def check_stop_words(line):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(line)
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    return wordsFiltered


def login():
    global current_user
    prompt_name()
    while expected_response != GENERAL:
        resp = input("User:>> ")
        if resp.lower() == "bye":
            return False
        if expected_response == USER_NAME:
            handle_name(resp)
        elif expected_response == CONFIRM_NAME:

            handle_name(resp)
        else:
            print("Return")
            return True
    return True


def find_item(item):
    global build_response
    global sent_tokens
    probable = ""
    found = False
    for token in sent_tokens:
        if item in token:
            build_response += token
            found = True
            break
        else:
            # Get the intersection
            n = set(token)
            m = set(item)
            sameWords = set.intersection(m, n)
            if len(sameWords) == len(n):
                build_response += token
                found = True
                break
            else:

                if len(sameWords) > 1:
                    if len(probable) > 1:
                        c = set.intersection(m, set(probable))
                        if len(c) < len(sameWords):
                            probable = token
                            found = True
    if found:
        # print("Found")
        print(build_response + probable)
    return found


def newParser(stmt):
    # save the user input for future processing
    user_querries.append(stmt)
    processed = False
    for qindicator in quiz_indicators:
        if qindicator in stmt:
            last_part = stmt[stmt.find(qindicator) + len(qindicator):]
            if not find_item(last_part):
                process_input(stmt)
                processed = True
            else:
                processed = True
            break
    if not processed:
        if not find_item(stmt):
            process_input(stmt)


def handle_name(n):
    global expected_response
    global keep_going
    global current_user
    if expected_response == USER_NAME:
        if n.strip().find(' ') == -1:  # One word
            expected_response = CONFIRM_NAME
            current_user = n
            print("You are " + n + " Right?")
        elif common_phrases[0] in n:
            z = n[n.find(common_phrases[0]) + len(common_phrases[0]):].strip()
            if z.find(' ') != -1:
                uname = z[0:z.find(' ')]
            else:
                uname = z
            if len(uname) > 2:
                current_user = uname
                # Confirm the name
                expected_response = CONFIRM_NAME
                print("You are " + uname + " Right?")
            else:
                print("What's Your first Name Again?")
                expected_response = USER_NAME

        elif common_phrases[2] in n:

            z = n[n.find(common_phrases[2]) + len(common_phrases[2]):].strip()

            if z.find(' ') != -1:
                uname = z[0:z.find(' ')]
            else:
                uname = z
            if len(uname) > 2:
                current_user = uname
                # Confirm the name
                expected_response = CONFIRM_NAME
                print("You are " + uname + " Right?")
            else:
                print("What's Your first Name Again?")
                expected_response = USER_NAME

        elif len(n.split()) == 1 or len(n.split()) == 2:
            expected_response = CONFIRM_NAME
            print("Your name is " + n + " Right?")

        else:
            pass

    elif expected_response == CONFIRM_NAME:
        # Check if they agreed
        if n.strip().find(' ') == -1:
            if n in agree_statements:
                print("Welcome " + current_user + "\n" + random.choice(prompt))
                expected_response = GENERAL
            elif n in disagree_statements:
                print("So what is your real name? ")
                expected_response = USER_NAME
            else:
                print("Cant really understand, Just type the name!")
                expected_response = USER_NAME
        else:
            words = n.split()
            if words[0] in agree_statements:
                # keep_going=False
                print("Welcome " + current_user + "\n" + random.choice(prompt))
                expected_response = GENERAL
            elif words[0] in disagree_statements:
                print("So what is your real name? ")
                expected_response = USER_NAME
    elif expected_response == GENERAL:
        return


# Program First method
def start():
    handle_greeting()
    if login():
        print()
        start_conversation()
    else:
        print(bot_input + "Bye ")
        handle_exit()


if __name__ == '__main__':
    prepare_bot_data()
    start()
