Chatbot read me File.
*********************

The chatbot is a fully functional python program build to run on a normal python ide.
It is designed to use a knowledge base scraped from the Internet to give accurate responses to a user based on a given knowledge domain.
It receives user input and tries to reply with the most relevant phrase.
The domain for the bot is cars. Much of the information available is about prices and some other attributes.

************ Installation ********
The bot utilizes a number of python modules as shown below..
To run the chatbot one has to install the following python modules:

1. NLTK --> pip install -U nltk
2. Sklearn  --> pip install -U scikit-learn
3. SpellChecker --> pip install pyspellchecker
4. BEautiful Soup (For the scarper file to scrape data from the internet)
5. Selenium (for Browser automation to ensure sites can be accesed by a bot)

The Nltk module helps with features like stop words stemming and lemmatization of sentences.
Sklearn contains tools for text analysis and comparison, while pyspell checker helps in correcting wrongly spelled words.

The bot file is called "chatbot.py". It contains the logic behind the bot.

****** Execution
The bot begins by relaying a greeting and a request to supply a name.

After the name is supplied a user can ask for a price of any car.
If the car is missing, the bot tries to find relevant information from the knowledge base.
In case there is no text matching the key words supplied then a request for a pharaphrase is raised.
The topics the bot covers include car prices and other general queries. 
It fairs quite well on prices for exact models like Audi A3, Bmw m5,Mazda BT-50,BMW 3 Series,Mitsubishi ASX etc
However sometimes, it can misunderstand and phrase and return an irrelevant response.

The cars models available include audi, bmw, jeep, honda, kia, mitsubishi,subaru among others!
The bot stores user preferences i.e  likes and dislikes in file where the preferences are used to determine responses.
Example, if the user types
"I hate audi, Let me have something like jeep che" then the bot will save that they don't like audi cars."


Below are some examples of conversation questions:
1. What is the price of a jeep?
2. If I want to have a bugatti, Can I get?
3. Do you have some cheap cars?
4. I want a luxury car, which one do you recommend to me?
5. I hate sedans so, can I get a audi suv?
6. What is good about a nissan Murano?
7. What is the price of a mitsubishi pajero?
8. Are there great features in a honda accord?
9. What is the best way to purchase a vehicle?
10. What are the pros and cons of dealer?
11. What are some of the safety regulations I should know?
12. What features does a honda civic have?
13. What is the safety score for a nissan altima?
14. Tell me about car scams and what to lookout for?
15. What are used cars in particular?
16. What does it mean by certified cars?



Sample chatbot dialogue:

<img src="Sample_chatbot_dialogue.PNG" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="300" height="300" />


Knowledge Base:

<img src="Knowledge_Base.PNG.jpg" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="300" height="300" />