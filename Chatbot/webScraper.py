import os

import html2text
import nltk
from bs4 import BeautifulSoup
from selenium import webdriver
from nltk.corpus import stopwords

# Python Program entry.
# Pass for page to be read
# scrap a vehicle dealer website and find their details
# To be specific, traverse different categories (Car Types)
# To be Dynamic complement with Vehicles by different Manufactures
# Crawler Code to crawl main page and get links
# Data Structure: A list  with all urls as Car

cartypes = []
vehicles = []

newlist = set()
newdict = {}


def get_main_links():
    mylinks = []
    url = "https://www.autoauctionmall.com/learning-center/used-cars/"
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)

    # create a new Firefox session
    driver.get(url)
    # Get page with URLs
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    data = soup.find('div', attrs={'class': 'blog-listing-body'})
    posts = data.findAll("div", {"class": "post-item"})
    n = 0
    for post in posts:
        page_name = "File" + str(n)
        page_url = post.find('a', {'class': 'post-item__title'})['href']
        title = post.find('a', {'class': 'post-item__title'}).text
        driver.get(page_url)
        page = driver.page_source
        # Save the file locally..
        f = open(os.path.join("pages/") + page_name + ".html", "w+", encoding='utf-8')
        f.write(str(page))
        driver.execute_script("window.history.go(-1)")
        print("Reading Page: ", page_name)
        n = n + 1
        # Limit to only 12 urls
        if n == 12:
            break
def extract_important_terms():
    # Read all cleaned files and get words and their frequency..
    # Use nltk to remove stop words

        data = ""
        with open('cleaned/knowledgebase.txt', 'r', encoding='utf8') as myfile:
            content = myfile.read()
        data = content.split()

        stop_words = stopwords.words('english')

        # Remove items from list
        for val in data:
            if val not in stop_words:
                # Do a custom check to remove some terms like numeric values and
                if val == "##":
                    # The value is used as start point in data
                    pass
                elif val.isnumeric():
                    # Numeric values cannot be terms
                    pass
                elif len(val) < 4:
                    # Items with length less than 4 are not sufficient terms
                    pass
                else:
                    newlist.add(val.lower())

        # Create a dictionary to hold all occurences of a word using the set
        for val in newlist:
            newdict[val] = 0
            # Read the files again to determine the frequencies
            # Read all data from files increasing the occurences

        with open('cleaned/knowledgebase.txt', 'r', encoding='utf8') as myfile:
            content = myfile.read()
        words = content.split()
        for word in words:
            word = word.lower().strip("|")
            if word in newdict:
                newdict[word] += 1
                # Increment the value
        # Finally, Display Top 40 terms
        print('**********************')
        print('Top VALUES!')
        print('**********************')
        for key in newdict:
            if newdict[key] > 5:
                print(key)
        # Pick 10 Terms to use
        print('**********************')
        print('Top 10 Selected!')
        print('**********************')
        print("Purchase, cars, market, toyota,jeep, money,affordable,dealer, vehicle,seller ")

def cleanPages():
    for filename in os.listdir('pages/'):
        # Remove unnecessary content from the files
        soup = BeautifulSoup(open("pages/" + filename, encoding='UTF-8'), "html.parser")
        b = soup.find("div", class_="single-post__body content")
        # print(str(b))
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True  # Strip Links
        text_maker.bypass_tables = True  # ignore tables
        text_maker.ignore_images = True
        text = text_maker.handle(str(b))

        # Use the Nltk tokenizer to extract sentenses from the text returned
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')  # Sentence detector
        content = '\n-----\n'.join(sent_detector.tokenize(b.strip()))

        print(filename)
        print()
        f = open(os.path.join("cleaned/") + filename, "w+", encoding='utf-8')
        f.write(str(content))

def clean_other_pages(filename):
        with open('pages/' + filename, 'r', encoding='utf8') as myfile:
            data = myfile.read()
            text_maker = html2text.HTML2Text()
            text_maker.ignore_links = True  # Strip Links
            text_maker.bypass_tables = True  # ignore tables
            text_maker.ignore_images = True
            text = text_maker.handle(str(data))

            # Use the Nltk tokenizer to extract sentenses from the text returned
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')  # Sentence detector
            content = '\n-----\n'.join(sent_detector.tokenize(text.strip()))

            print(filename)
            print()
            f = open(os.path.join("cleaned/") + filename, "w+", encoding='utf-8')
            f.write(str(content))
        print("Done cleaning")

# Read the extra sites

def readMoreSites():
    print("Reading site:")
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    # Additional Web Page
    url1="http://thefactfile.org/car-facts/"
    url2="https://www.a1autotransport.com/container-car-shipping-faq-and-answers/"
    url3="https://www.consumerreports.org/car-reliability/10-most-reliable-cars/"
    driver.get(url1)
    page = driver.page_source
    # Save the file locally..
    f = open(os.path.join("pages/") +"facts.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")

    driver.get(url2)
    page = driver.page_source
    # Save the file locally..
    f = open(os.path.join("pages/")  + "shipping.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")
    driver.get(url3)
    page = driver.page_source
    f = open(os.path.join("pages/") +  "reliable.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")
    print("Done reading Pages: ", )

def clean_last_pages():
    clean_other_pages("facts.html")
    clean_other_pages("shipping.html")
    clean_other_pages("reliable.html")

#Start Scraping
print("Starting service")
get_main_links()
readMoreSites()
cleanPages()
clean_last_pages()

extract_important_terms()

print("Data Scraped.")


