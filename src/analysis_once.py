import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import plot_confusion_matrix, plot_roc_curve, get_close_predicitions


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

    # Training
    classifier = RandomForestClassifier()
    classifier.fit(train_features, train_labels)

    # Evaluation
    prediction = classifier.predict(test_features)
    prediction_proba = classifier.predict_proba(test_features)

    print(classification_report(test_labels, prediction))
    close_predictions = get_close_predicitions(prediction_proba)
    print(close_predictions)
    print(len(close_predictions))

    plot_roc_curve(test_labels, prediction_proba)
    plot_confusion_matrix(test_labels, prediction)
    plt.show()


if __name__ == "__main__":
    main()
