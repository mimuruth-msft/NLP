import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.corpus import stopwords

# Program to Classify Amazon Reviews


def pre_process_data():
    # Read csv data in to a pandas Data frame.
    data = pd.read_csv('/content/sample_data/amazon_reviews.csv')
    # Print first five results
    print(data.head())
    print(data.shape)

    # Remove stop words from the data
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, binary=True)

    # set up X and y
    X = vectorizer.fit_transform(data.Review)
    y = data.Rating

    # Create a Test/ train set with 80% test and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1234)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    # make predictions on the test data
    pred = model.predict(X_test)
    print('accuracy score: ', accuracy_score(y_test, pred))
    print('precision score: ', precision_score(y_test, pred))
    print('recall score: ', recall_score(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred))


def naive_bayes(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def svm_linear(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', degree=8)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def svm_radial(X_train, X_test, y_train, y_test):
    model = SVC(kernel='poly', degree=8)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def keras(X_train, X_test, y_train, y_test):
    pass


if __name__ == "__main__":
    # Read Data and Analyze
    X_train, X_test, y_train, y_test = pre_process_data()

    # Step 1: Naive Bayes
    print()
    print("*********************************")
    print("Naive Bayes Algorithm")
    naive_bayes(X_train, X_test, y_train, y_test)

    # Step 2: Logistic Regression
    print()
    print("*********************************")
    print("Logistic Regression Algorithm")
    logistic_regression(X_train, X_test, y_train, y_test)

    # Step 3: Random Forest
    print()
    print("*********************************")
    print("Random Forest Algorithm")
    random_forest(X_train, X_test, y_train, y_test)
    
    # Step 4: SVM Linear Algorithm
    print()
    print("*********************************")
    print("SVM Linear Algorithm")
    svm_linear(X_train, X_test, y_train, y_test)
    
    # Step 4: SVM Radial Algorithm
    print()
    print("*********************************")
    print("SVM Radial Algorithm")
    svm_radial(X_train, X_test, y_train, y_test)
    
    # Step 4: SVM Radial Algorithm
    print()
    print("*********************************")
    print("keras")
    keras(X_train, X_test, y_train, y_test)
    
    print('\nCompleted')
    
