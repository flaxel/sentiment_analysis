from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    # Preprocessed Dataset
    reviews_df = pd.read_csv("../data/reviews.csv")
    print(reviews_df.head())
    print("\n")

    reviews_labels = reviews_df["sentiment"]
    reviews_features = reviews_df["tokens"]

    # Vectorization
    vectorizer_reviews = TfidfVectorizer()
    vectorized_reviews_features = vectorizer_reviews.fit_transform(reviews_features)

    train_reviews_features, test_reviews_features, train_reviews_labels, test_reviews_labels = \
        train_test_split(vectorized_reviews_features, reviews_labels, test_size=0.3)

    # Prediction
    classifier = RandomForestClassifier()
    classifier.fit(train_reviews_features, train_reviews_labels)

    prediction = classifier.predict(test_reviews_features)
    print(classification_report(test_reviews_labels, prediction))
    print(confusion_matrix(test_reviews_labels, prediction))


if __name__ == "__main__":
    # configuration of pandas
    pd.set_option("display.max_columns", None)

    start = datetime.now()
    print("Start:", start, "\n")

    main()

    end = datetime.now()
    print("\nEnd:", end)

    print("Runtime:", end - start)
