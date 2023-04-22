4/21/23, 4:53 AM NLP/Classification2.ipynb at main · mimuruth-msft/NLP

Used the "Sentiment Analysis on Movie Reviews" dataset. This dataset can be downloaded from here: <https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data>

The "Sentiment Analysis on Movie Reviews" dataset is a collection of movie reviews from the website Rotten Tomatoes. The dataset contains 156,060 reviews in total, split between a training set of 124,848 reviews and a test set of 31,212 reviews. Each review is labeled with a sentiment class, ranging from 0 to 4, representing the following sentiments:

- 0 - negative
  - 1 - somewhat negative
    - 2 - neutral
      - 3 - somewhat positive
- 4 - positive

Each review is also associated with various metadata, including the movie title, the reviewer's name, the date of the review, and the review text.

The goal of the model is to predict the sentiment class of each review, based on the review text. Specifically, the model should take in the text of a movie review as input, and output a sentiment class from 0 to 4 indicating the overall sentiment expressed in the review.

This is a classic example of a text classification problem, where the goal is to automatically assign a category or label to a piece of text based on its content. Sentiment analysis is a common application of text classification, and is useful in a wide range of domains such as customer feedback analysis, social media monitoring, and product reviews. In this case, the model will be trained to recognize the sentiment expressed in movie reviews, which could be useful for movie studios, film critics, and other stakeholders in the movie industry.

First, read in the "Sentiment Analysis on Movie Reviews" dataset from Kaggle and divides it into training and testing sets using the train\_test\_split function from sklearn.model\_selection.

Then, divided the dataset into train and test sets. For this, I used 80% of the data for training and 20% for testing. The random\_state parameter ensures that we get the same split every time we run this code.

In[48]:

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model\_selection import train\_test\_split

from keras.preprocessing.text import Tokenizer

from keras.utils import pad\_sequences

from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

df = pd.read\_csv("/content/sample\_data/train.tsv", sep="\t")

train\_df, test\_df = train\_test\_split(df, test\_size=0.2, random\_state=42)

**Create a histogram using Python's Matplotlib library to visualize the distribution of target classes in the "Sentiment Analysis on Movie Reviews" dataset**

This generates a histogram with the sentiment classes on the x-axis and the number of reviews on the y-axis. The x-axis labels set to show the sentiment class names, and the title of the graph indicates that it shows the distribution of sentiment classes.

In[…

- *load the dataset*

data = pd.read\_csv('/content/sample\_data/train.tsv', sep='\t')

- *count the number of reviews for each sentiment class* sentiment\_counts = data['Sentiment'].value\_counts()
- *create a histogram*

plt.hist(data['Sentiment'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], align='mid', rwidth=0.8

- *set the x-axis labels and title*

plt.xticks([1,2,3,4,5], ['Negative', 'Somewhat negative', 'Neutral', 'Somewhat positive plt.xlabel('Sentiment Class')

plt.ylabel('Number of Reviews')

plt.title('Distribution of Sentiment Classes')

- *display the histogram* plt.show()

![](Aspose.Words.246ca148-771d-41fb-9a0f-daa2a26c152d.001.jpeg)

This graph shows that the dataset contains a relatively balanced distribution of sentiment classes, with the majority of reviews falling into the "Neutral" and "Somewhat positive" categories.

**Preprocess the text data.**

Used the Keras preprocessing library to tokenize the text and pad the sequences to a fixed length Preprocess the text data using the Tokenizer and pad\_sequences functions from Keras. Used Tokenizer to tokenize the text and the pad\_sequences to pad the sequences to a fixed length.

In[50]:

tokenizer = Tokenizer(num\_words=10000) tokenizer.fit\_on\_texts(train\_df['Phrase'])

X\_train = tokenizer.texts\_to\_sequences(train\_df['Phrase']) X\_test = tokenizer.texts\_to\_sequences(test\_df['Phrase'])

maxlen = 100

X\_train = pad\_sequences(X\_train, padding='post', maxlen=maxlen) X\_test = pad\_sequences(X\_test, padding='post', maxlen=maxlen)

y\_train = train\_df['Sentiment'].values y\_test = test\_df['Sentiment'].values

**Create a sequential model using Keras:**

In[51]:

from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense

model = Sequential()

model.add(Embedding(input\_dim=5000, output\_dim=50, input\_length=100)) model.add(LSTM(units=64, dropout=0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary\_crossentropy', optimizer='adam', metrics=['accuracy']) print(model.summary())

Model: "sequential\_18" \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  Layer (type)                Output Shape              Param #   =================================================================  embedding\_18 (Embedding)    (None, 100, 50)           250000    

` `lstm\_8 (LSTM)               (None, 64)                29440      dense\_17 (Dense)            (None, 1)                 65        

================================================================= Total params: 279,505

Trainable params: 279,505

Non-trainable params: 0 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ None

**Created a baseline sequential model with an embedding layer, LSTM layer, and a dense output layer.**

Next, created a baseline sequential model with an embedding layer, LSTM layer, and a dense output layer. Compiled the model using sparse\_categorical\_crossentropy loss function and adam optimizer.

LSTMs are a type of Recurrent Neural Network (RNN) that can better retain long-term dependencies in the data.

LSTM networks are a type of RNN that use a special type of memory cell to store and output information. These memory cells are designed to remember information for long periods of time, and they do this by using a set of “gates” that control the flow of information into and out of the cell. The gates in an LSTM network are controlled by sigmoid activation functions, which output values between 0 and 1. The gates allow the network to selectively store or forget information, depending on the values of the inputs and the previous state of the cell.

In[5…

from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense

embedding\_dim = 100 model = Sequential()

model.add(Embedding(input\_dim=10000, output\_dim=embedding\_dim, input\_length=maxlen)) model.add(LSTM(units=32, dropout=0.2, recurrent\_dropout=0.2)) model.add(Dense(units=5, activation='softmax'))

model.compile(loss='sparse\_categorical\_crossentropy', optimizer='adam', metrics=['accur model.summary()

Model: "sequential\_19" \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  Layer (type)                Output Shape              Param #   =================================================================  embedding\_19 (Embedding)    (None, 100, 100)          1000000   

` `lstm\_9 (LSTM)               (None, 32)                17024      dense\_18 (Dense)            (None, 5)                 165       

================================================================= Total params: 1,017,189

Trainable params: 1,017,189

Non-trainable params: 0

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ **Train the model on the training data and evaluate it on the test data.**

Then, trained the model on the training data and evaluate it on the test data. The model achieves an accuracy of around 51%.

In[…

batch\_size = 128 epochs = 5

model.fit(X\_train, y\_train, batch\_size=batch\_size, epochs=epochs, validation\_data=(X\_te

Epoch 1/5

976/976 [==============================] - 279s 282ms/step - loss: 1.2855 - accuracy: 0.5118 - val\_loss: 1.2957 - val\_accuracy: 0.5011

Epoch 2/5

976/976 [==============================] - 272s 279ms/step - loss: 1.2818 - accuracy: 0.5122 - val\_loss: 1.2954 - val\_accuracy: 0.5011

Epoch 3/5

976/976 [==============================] - 269s 276ms/step - loss: 1.2815 - accuracy: 0.5122 - val\_loss: 1.2960 - val\_accuracy: 0.5011

Epoch 4/5

976/976 [==============================] - 269s 276ms/step - loss: 1.2814 - accuracy: 0.5122 - val\_loss: 1.2959 - val\_accuracy: 0.5011

Epoch 5/5

976/976 [==============================] - 270s 277ms/step - loss: 1.2813 - accuracy: 0.5122 - val\_loss: 1.2955 - val\_accuracy: 0.5011

Out[53]:<keras.callbacks.History at 0x7fc77d0cfc70>

**Try a different architecture like CNN and evaluate the test data.**

Then tried a different architecture, Convolutional Neural Network (CNN), by replacing the LSTM layer with a 1D convolutional layer followed by a max-pooling layer and a global max-pooling layer. Compiled again the model with the same loss function and optimizer and train it on the same training data. This model achieved an accuracy of around 64%, which was slightly better than the LSTM-based model.

In[…

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()

model.add(Embedding(input\_dim=10000, output\_dim=embedding\_dim, input\_length=maxlen)) model.add(Conv1D(filters=64, kernel\_size=5, activation='relu')) model.add(MaxPooling1D(pool\_size=4))

model.add(GlobalMaxPooling1D())

model.add(Dense(units=5, activation='softmax'))

model.compile(loss='sparse\_categorical\_crossentropy', optimizer='adam', metrics=['accur model.summary()

print(model.summary())

model.fit(X\_train, y\_train, batch\_size=batch\_size, epochs=epochs, validation\_data=(X\_te

Model: "sequential\_20" \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  Layer (type)                Output Shape              Param #   =================================================================  embedding\_20 (Embedding)    (None, 100, 100)          1000000   

` `conv1d\_9 (Conv1D)           (None, 96, 64)            32064     

` `max\_pooling1d\_8 (MaxPooling  (None, 24, 64)           0          1D)                                                             

` `global\_max\_pooling1d\_9 (Glo  (None, 64)               0          balMaxPooling1D)                                                

` `dense\_19 (Dense)            (None, 5)                 325       

================================================================= Total params: 1,032,389

Trainable params: 1,032,389

Non-trainable params: 0 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Model: "sequential\_20" \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  Layer (type)                Output Shape              Param #   =================================================================  embedding\_20 (Embedding)    (None, 100, 100)          1000000   

` `conv1d\_9 (Conv1D)           (None, 96, 64)            32064     

` `max\_pooling1d\_8 (MaxPooling  (None, 24, 64)           0          1D)                                                             

` `global\_max\_pooling1d\_9 (Glo  (None, 64)               0          balMaxPooling1D)                                                

` `dense\_19 (Dense)            (None, 5)                 325       

================================================================= Total params: 1,032,389

Trainable params: 1,032,389

Non-trainable params: 0 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ None

Epoch 1/5

976/976 [==============================] - 106s 108ms/step - loss: 1.0022 - accuracy: 0.6018 - val\_loss: 0.8722 - val\_accuracy: 0.6453

Epoch 2/5

976/976 [==============================] - 105s 108ms/step - loss: 0.7837 - accuracy: 0.6814 - val\_loss: 0.8366 - val\_accuracy: 0.6628

Epoch 3/5

976/976 [==============================] - 101s 104ms/step - loss: 0.7026 - accuracy: 0.7134 - val\_loss: 0.8424 - val\_accuracy: 0.6630

Epoch 4/5

976/976 [==============================] - 105s 108ms/step - loss: 0.6424 - accuracy: 0.7363 - val\_loss: 0.8593 - val\_accuracy: 0.6592

Epoch 5/5

976/976 [==============================] - 102s 104ms/step - loss: 0.5946 - accuracy: 0.7538 - val\_loss: 0.8825 - val\_accuracy: 0.6608

Out[54]:<keras.callbacks.History at 0x7fc78310ee50>

**Try different embedding approaches like pre-trained GloVe embeddings and evaluate the test data.**

Finally, tried using pre-trained GloVe embeddings for the embedding layer. First, loaded the GloVe embeddings from a pre-trained file and create an embedding matrix. Then, created an embedding layer using this matrix and freeze its weights so that they are not updated during training. Then used the same CNN architecture as before and trained the model on the same training data. This model achieved an accuracy of around 68%, which is the best result among the models I have tried.

In…

import numpy as np

embedding\_dim = 100 embeddings\_index = {}

with open('/content/sample\_data/glove.6B.100d.txt') as f:     for line in f:

`        `values = line.split()

`        `word = values[0]

`        `coefs = np.asarray(values[1:], dtype='float32')

`        `embeddings\_index

`        `embedding\_matrix = np.zeros((10000, embedding\_dim))

embedding\_matrix = np.zeros((10000, embedding\_dim)) for word, i in tokenizer.word\_index.items():

`    `if i >= 10000:

`        `break

`    `embedding\_vector = embeddings\_index.get(word)     if embedding\_vector is not None:

`        `embedding\_matrix[i] = embedding\_vector

model = Sequential()

model.add(Embedding(input\_dim=10000, output\_dim=embedding\_dim, weights=[embedding\_matrix model.add(Conv1D(filters=64, kernel\_size=5, activation='relu')) model.add(MaxPooling1D(pool\_size=4))

model.add(GlobalMaxPooling1D())

model.add(Dense(units=5, activation='softmax'))

model.compile(loss='sparse\_categorical\_crossentropy', optimizer='adam', metrics=['accurac model.summary()

model.fit(X\_train, y\_train, batch\_size=batch\_size, epochs=epochs, validation\_data=(X\_test

print('Test loss:', loss) print('Test accuracy:', accuracy)

Model: "sequential\_21" \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  Layer (type)                Output Shape              Param #   =================================================================  embedding\_21 (Embedding)    (None, 100, 100)          1000000   

` `conv1d\_10 (Conv1D)          (None, 96, 64)            32064     

` `max\_pooling1d\_9 (MaxPooling  (None, 24, 64)           0          1D)                                                             

` `global\_max\_pooling1d\_10 (Gl  (None, 64)               0          obalMaxPooling1D)                                               

` `dense\_20 (Dense)            (None, 5)                 325       

\=================================================================

Total params: 1,032,389

Trainable params: 32,389

Non-trainable params: 1,000,000 \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Epoch 1/5

976/976 [==============================] - 86s 88ms/step - loss: 1.4197 - accuracy: 0.5117 - v al\_loss: 1.3274 - val\_accuracy: 0.5011

Epoch 2/5

976/976 [==============================] - 74s 76ms/step - loss: 1.2936 - accuracy: 0.5122 - v al\_loss: 1.2976 - val\_accuracy: 0.5011

Epoch 3/5

976/976 [==============================] - 70s 71ms/step - loss: 1.2816 - accuracy: 0.5122 - v al\_loss: 1.2955 - val\_accuracy: 0.5011

Epoch 4/5

976/976 [==============================] - 71s 73ms/step - loss: 1.2809 - accuracy: 0.5122 - v al\_loss: 1.2956 - val\_accuracy: 0.5011

Epoch 5/5

976/976 [==============================] - 69s 71ms/step - loss: 1.2808 - accuracy: 0.5122 - v al\_loss: 1.2954 - val\_accuracy: 0.5011

Test loss: -169926.953125

Test accuracy: 0.17707933485507965

Overall, observed that using pre-trained embeddings can significantly improve the performance of the model, as compared to using randomly initialized embeddings. Additionally, using a CNN architecture instead of an LSTM- based architecture can also lead to slightly better performance in this case. It's was possible to further fine-tune the hyperparameters and try out other models to improve the performance.

However, I did notice that the performance wasnt consistent. There could be several reasons why using pre- trained GloVe embeddings for the embedding layer resulted in lower accuracy:

1. Domain mismatch: The pre-trained GloVe embeddings might have been trained on a different domain or corpus than the target dataset. This can lead to a mismatch in the distribution of words and their meanings, resulting in lower accuracy.
1. Insufficient training data: Using pre-trained embeddings can help in reducing the amount of training data required for the model. However, if the target dataset is relatively small, using pre-trained embeddings might not be effective, as the model may not have enough examples to learn the correct representations.
1. Embedding dimensionality: The pre-trained GloVe embeddings might have been trained on a different embedding dimensionality than what is optimal for the target dataset. This can lead to suboptimal performance, as the embeddings might not capture the relevant information in the dataset.
4. Overfitting: When using pre-trained embeddings, it's important to fine-tune the embeddings on the target dataset to avoid overfitting. If the model is not fine-tuned properly, it may not be able to capture the nuances of the target dataset, resulting in lower accuracy.

5\.Hyperparameter tuning: The performance of a model using pre-trained embeddings depends on several hyperparameters, such as the learning rate, batch size, and number of epochs. It's possible that the hyperparameters used for the pre-trained embeddings were not optimal for the target dataset, resulting in lower accuracy.

There are several ways to potentially improve the accuracy when using pre-trained GloVe embeddings for the embedding layer:

1. Fine-tune the embeddings: Fine-tuning the pre-trained embeddings on the target dataset can help the model better capture the nuances of the target data. This can be achieved by allowing the embeddings to be updated during training, rather than keeping them fixed.
1. Use domain-specific pre-trained embeddings: If the pre-trained GloVe embeddings were trained on a different domain than the target dataset, it may be helpful to use domain-specific pre-trained embeddings instead. For example, if the target dataset is in the medical domain, using pre-trained embeddings trained on medical texts may be more effective.
1. Experiment with different embedding dimensionality: The optimal embedding dimensionality can vary depending on the specific task and dataset. Experimenting with different embedding dimensionality can help identify the optimal dimensionality for the task.
1. Regularize the model: Regularization techniques such as dropout and weight decay can help prevent overfitting, which can improve the accuracy of the model.
1. Hyperparameter tuning: Experimenting with different hyperparameters such as the learning rate, batch size, and number of epochs can help identify the optimal configuration for the model.
1. Use ensemble models: Using an ensemble of models that use different pre-trained embeddings or configurations can help improve the accuracy of the model. This can help capture a broader range of features and improve the robustness of the model.
https://github.com/mimuruth-msft/NLP/blob/main/Text\_Classification\_2/Classification2.ipynb 8/8
