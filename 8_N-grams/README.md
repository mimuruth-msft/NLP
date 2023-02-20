**Ngrams**

**Overview** :

- In this homework you will create bigram and unigram dictionaries for English, French, and Italian using the provided training data where the key is the unigram or bigram text and the value is the count of that unigram or bigram in the data. Then for the test data, calculate probabilities for each language and compare against the true labels.
- Please use an IDE

**Instructions** :

1. Program 1: Build separate language models for 3 languages as follows.

1. create a function with a filename as argument.
2. read in the text and remove newlines
3. tokenize the text
4. use nltk to create a bigrams list
5. use nltk to create a unigrams list
6. use the bigram list to create a bigram dictionary of bigrams and counts, ['token1 token2'] -\> count
7. use the unigram list to create a unigram dictionary of unigrams and counts, ['token'] -\> count
8. return the unigram dictionary and bigram dictionary from the function
9. in the main body of code, call the function 3 times for each training file, pickle the 6 dictionaries, and save to files with appropriate names. The reason we are pickling them in one program and unpickling them in another is that NLTK ngrams is slow and if you put this all in one program, you may waste a lot of time waiting for ngrams() to finish.

**2. Program 2.**

1. Read in your pickled dictionaries.
2. For each line in the test file, calculate a probability for each language (see note below) and write the language with the highest probability to a file.
3. Compute and output your accuracy as the percentage of correctly classified instances in the test set. The file LangId.sol holds the correct classifications.
4. utput your accuracy, as well as the line numbers of the incorrectly classified items

**About Ngrams**

1. What are n-grams and how are they used to build a language model

N-grams are contiguous sequences of n words (or characters) extracted from a text corpus. They are used to build language models by counting the frequency of occurrence of each n-gram in the corpus and then estimating the probability of the next word in a sentence given the previous n-1 words.

1. List a few applications where n-grams could be used.

Some applications where n-grams could be used include:

- Text classification
- Text prediction
- Machine translation
- Spell checking
- Information retrieval
- Sentiment analysis

1. A description of how probabilities are calculated for unigrams and bigrams.

Probabilities for unigrams (1-grams) are simply calculated by dividing the frequency of each word in the corpus by the total number of words in the corpus. Probabilities for bigrams (2-grams) are calculated by dividing the frequency of each bigram by the frequency of the preceding unigram.

1. The importance of the source text in building a language model

The source text is very important in building a language model because the model is only as good as the corpus it is trained on. The model will only be able to generate or predict text that is similar to the source text. It is also important to use a corpus that is representative of the target domain or genre.

1. The importance of smoothing and describe a simple approach to smoothing.

Smoothing is important in language modeling to avoid zero probabilities for n-grams that were not observed in the training corpus. A simple approach to smoothing is add-k smoothing, where a small constant k is added to the counts of all n-grams before calculating their probabilities. This ensures that all n-grams have a non-zero probability.

1. Describe how language models can be used for text generation, and the limitations of this approach.

Language models can be used for text generation by sampling words from the model's probability distribution. Starting with a seed word or phrase, the model generates the next word based on the probabilities of the n-grams that contain the seed words. This process is repeated until a desired length is reached. The limitations of this approach include the fact that the generated text may be nonsensical or grammatically incorrect, and the model may also generate text that is repetitive or biased towards the source text.

1. Describe how language models can be evaluated.

Language models can be evaluated using metrics such as perplexity, which measures how well the model predicts a held-out test set, or accuracy, which measures how well the model classifies a set of test documents. Human evaluation can also be used to assess the quality of the generated text.

1. Give a quick introduction to Google's n-gram viewer and show an example.

Google's n-gram viewer is a tool that allows users to search for the frequency of n-grams in the Google Books corpus, which contains millions of books published over the past few centuries. The Google n-gram viewer can be useful for researchers and language enthusiasts who are interested in analyzing the usage of words or phrases over time, for example, to tracking changes in language overtime, identify linguistic trends, and study cultural shifts. An example of its use could be to search for the frequency of the phrase "climate change" in books published between 1900 and 2000.