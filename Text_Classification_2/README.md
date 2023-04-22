**Text Classification 2**

The "Sentiment Analysis on Movie Reviews" dataset. 

This dataset can be downloaded from here: <https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data>

The "Sentiment Analysis on Movie Reviews" dataset is a collection of movie reviews from the website Rotten Tomatoes. The dataset contains 156,060 reviews in total, split between a training set of 124,848 reviews and a test set of 31,212 reviews. Each review is labeled with a sentiment class, ranging from 0 to 4, representing the following sentiments:

- 0 - negative
- 1 - somewhat negative
- 2 - neutral
- 3 - somewhat positive
- 4 - positive

Each review is also associated with various metadata, including the movie title, the reviewer's name, the date of the review, and the review text.

The goal of the model is to predict the sentiment class of each review, based on the review text. Specifically, the model should take in the text of a movie review as input and output a sentiment class from 0 to 4 indicating the overall sentiment expressed in the review.

This is a classic example of a text classification problem, where the goal is to automatically assign a category or label to a piece of text based on its content. Sentiment analysis is a common application of text classification and is useful in a wide range of domains such as customer feedback analysis, social media monitoring, and product reviews. In this case, the model will be trained to recognize the sentiment expressed in movie reviews, which could be useful for movie studios, film critics, and other stakeholders in the movie industry.

![Chart, bar chart

Description automatically generated](image/Distribution_of_Sentiment.png)

Overall, observed that using pre-trained embeddings can significantly improve the performance of the model, as compared to using randomly initialized embeddings. Additionally, using a CNN architecture instead of an LSTM-based architecture can also lead to slightly better performance in this case. It was possible to further fine-tune the hyperparameters and try out other models to improve the performance.

However, I did notice that the performance didn’t result in higher accuracy. There could be several reasons why using pre-trained GloVe embeddings for the embedding layer resulted in lower accuracy:

- 1. Domain mismatch: The pre-trained GloVe embeddings might have been trained on a different domain or corpus than the target dataset. This can lead to a mismatch in the distribution of words and their meanings, resulting in lower accuracy.
- 2. Insufficient training data: Using pre-trained embeddings can help in reducing the amount of training data required for the model. However, if the target dataset is relatively small, using pre-trained embeddings might not be effective, as the model may not have enough examples to learn the correct representations.
- 3. Embedding dimensionality: The pre-trained GloVe embeddings might have been trained on a different embedding dimensionality than what is optimal for the target dataset. This can lead to suboptimal performance, as the embeddings might not capture the relevant information in the dataset.
- 4. Overfitting: When using pre-trained embeddings, it's important to fine-tune the embeddings on the target dataset to avoid overfitting. If the model is not fine-tuned properly, it may not be able to capture the nuances of the target dataset, resulting in lower accuracy.
- 5. Hyperparameter tuning: The performance of a model using pre-trained embeddings depends on several hyperparameters, such as the learning rate, batch size, and number of epochs. It's possible that the hyperparameters used for the pre-trained embeddings were not optimal for the target dataset, resulting in lower accuracy.

There are several ways to potentially improve the accuracy when using pre-trained GloVe embeddings for the embedding layer:

- 1. Fine-tune the embeddings: Fine-tuning the pre-trained embeddings on the target dataset can help the model better capture the nuances of the target data. This can be achieved by allowing the embeddings to be updated during training, rather than keeping them fixed.
- 2. Use domain-specific pre-trained embeddings: If the pre-trained GloVe embeddings were trained on a different domain than the target dataset, it may be helpful to use domain-specific pre-trained embeddings instead. For example, if the target dataset is in the medical domain, using pre-trained embeddings trained on medical texts may be more effective.
- 3. Experiment with different embedding dimensionality: The optimal embedding dimensionality can vary depending on the specific task and dataset. Experimenting with different embedding dimensionality can help identify the optimal dimensionality for the task.
- 4. Regularize the model: Regularization techniques such as dropout and weight decay can help prevent overfitting, which can improve the accuracy of the model.
- 5. Hyperparameter tuning: Experimenting with different hyperparameters such as the learning rate, batch size, and number of epochs can help identify the optimal configuration for the model.
- 6. Use ensemble models: Using an ensemble of models that use different pre-trained embeddings or configurations can help improve the accuracy of the model. This can help capture a broader range of features and improve the robustness of the model.
