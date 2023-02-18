import os

import html2text
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver.common import options

from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType

# driver = webdriver.Chrome(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())

# Build a web crawler function that starts with a URL representing a topic
# (a sport, your favorite film, a celebrity, a political issue, etc.) and
# outputs a list of at least 15 relevant URLs. The URLs can be pages within
# the original domain but should have a few outside the original domain.

# Pass page to be read, scrap a vehicle dealer website and find  their details
# To be specific, traverse the different categories (Car Types)
# To be dynamic complemented with vehicles by different manufactures
# Crawler Code to crawl main page and get links
# Data Structure: A list with all urls as Car

# empty list
car_types = []
vehicles = []

new_list = set()
new_dict = {}


def get_main_links():
    mylinks = []
    url = "https://www.autoauctionmall.com/learning-center/used-cars/"
    driver = webdriver.Firefox()
    # implicit wait tells WebDriver to poll the DOM for a specific amount of time when
    # attempting to find any elements that are not immediately available.
    driver.implicitly_wait(30)  # seconds

    # create a new Chrome session
    driver.get(url)
    # get page with URLs
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    data = soup.find('div', attrs={'class': 'blog-listing-body'})
    posts = data.findAll("div", {"class": "post-item"})
    n = 0
    for post in posts:
        page_name = f"File{str(n)}"
        page_url = post.find('a', {'class': 'post-item__title'})['href']
        title = post.find('a', {'class': 'post-item__title'}).text
        driver.get(page_url)
        page = driver.page_source
        # Save the file locally.
        f = open(os.path.join("pages/") + page_name + ".html", "w+", encoding='utf-8')
        f.write(str(page))
        driver.execute_script("window.history.go(-1)")
        print("Reading Page: ", page_name)
        n = n + 1
        # Limit to only 12 URLs
        if n == 12:
            break


def extract_important_terms():
    # Read all cleaned files and get words and their frequency.
    # Use NLTK to remove stop words

    data = ""
    with open('cleaned/knowledgebase.txt', 'r', encoding='utf8') as myfile:
        content = myfile.read()
    data = content.split()

    stop_words = stopwords.words('english')

    # Remove items from list
    for val in data:
        if val not in stop_words:
            # Do a custom check to remove some terms like numeric values and
            if val == "##" or val != "##" and val.isnumeric():
                # The value is used as start point in data
                pass
            elif val != "##" and not val.isnumeric() and len(val) >= 4:
                new_list.add(val.lower())

    # Create a dictionary to hold all occurrences of a word using the set
    for val in new_list:
        new_dict[val] = 0
        # Read the files again to determine the frequencies
        # Read all data from files increasing the occurrences

    with open('cleaned/knowledgebase.txt', 'r', encoding='utf8') as myfile:
        content = myfile.read()
    words = content.split()
    for word in words:
        word = word.lower().strip("|")
        if word in new_dict:
            new_dict[word] += 1
            # Increment the value
    _extracted_from_extract_important_terms_43('Top VALUES!')
    for key in new_dict:
        if new_dict[key] > 5:
            print(key)
    _extracted_from_extract_important_terms_43('Top 10 Selected!')
    print("Purchase, cars, market, toyota, jeep, money, affordable, dealer, vehicle, seller ")


# TODO Rename this here and in `extract_important_terms`
def _extracted_from_extract_important_terms_43(arg0):
    # finally display top 40 terms
    print('**********************')
    print(arg0)
    print('**********************')


def clean_pages():
    for filename in os.listdir('pages/'):
        # remove unnecessary content from the files
        soup = BeautifulSoup(
            open(f"pages/{filename}", encoding='UTF-8'), "html.parser"
        )
        b = soup.find("div", class_="single-post__body content")
        # print(str(b))
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True  # strip Links
        text_maker.bypass_tables = True  # ignore tables
        text_maker.ignore_images = True
        text = text_maker.handle(str(b))

        # use the NLTK tokenizer to extract sentences from the text returned
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')  # Sentence detector
        content = '\n-----\n'.join(sent_detector.tokenize(text.strip()))

        print(filename)
        print()
        f = open(os.path.join("cleaned/") + filename, "w+", encoding='utf-8')
        f.write(content)


def clean_other_pages(filename):
    with open(f'pages/{filename}', 'r', encoding='utf8') as myfile:
        data = myfile.read()
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True  # strip Links
        text_maker.bypass_tables = True  # ignore tables
        text_maker.ignore_images = True
        text = text_maker.handle(str(data))

        # use the NLTK tokenizer to extract sentences from the text returned
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')  # Sentence detector
        content = '\n-----\n'.join(sent_detector.tokenize(text.strip()))

        print(filename)
        print()
        f = open(os.path.join("cleaned/") + filename, "w+", encoding='utf-8')
        f.write(content)
    print("Done cleaning")


# Read the additional URLs outside the original domain
def read_more_sites():
    print("Reading site:")
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    # additional URLs outside the original domain.
    url1 = "https://thefactfile.org/car-facts/"
    url2 = "https://www.a1autotransport.com/container-car-shipping-faq-and-answers/"
    url3 = "https://www.consumerreports.org/car-reliability/10-most-reliable-cars/"
    url4 = "https://www.caranddriver.com/first-drives/"
    driver.get(url1)
    page = driver.page_source
    # store the file locally with each page’s text in its own file
    f = open(os.path.join("pages/") + "facts.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")

    driver.get(url2)
    page = driver.page_source
    # store the file locally.
    f = open(os.path.join("pages/") + "shipping.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")
    driver.get(url3)
    page = driver.page_source
    f = open(os.path.join("pages/") + "reliable.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")
    driver.get(url4)
    page = driver.page_source
    f = open(os.path.join("pages/") + "reviews.html", "w+", encoding='utf-8')
    f.write(str(page))
    driver.execute_script("window.history.go(-1)")
    print("Done reading Pages: ", )


# Python program entry

def clean_last_pages():
    clean_other_pages("facts.html")
    clean_other_pages("shipping.html")
    clean_other_pages("reliable.html")


# start scraping all text off each page
print("Starting service")
get_main_links()
read_more_sites()
# functions to clean up the text from each file
clean_pages()
clean_last_pages()

extract_important_terms()

print("Data Scraped.")
