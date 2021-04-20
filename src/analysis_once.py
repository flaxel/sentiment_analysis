import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    # Preprocessed Dataset
    data = pd.read_csv("../data/reviews.csv")
    # data = pd.read_csv("../data/tweets.csv")
    print(data.head())
    print("\n")

    labels = data["sentiment"]
    features = data["tokens"]

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorized_features = vectorizer.fit_transform(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(vectorized_features, labels, test_size=0.3)

    # Prediction
    classifier = RandomForestClassifier()
    classifier.fit(train_features, train_labels)

    prediction = classifier.predict(test_features)
    print(classification_report(test_labels, prediction))
    print(confusion_matrix(test_labels, prediction))


if __name__ == "__main__":
    main()
