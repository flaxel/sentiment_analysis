import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import plot_confusion_matrix, plot_roc_curve, plot_wordcloud, plot_top_n_words
from utils import get_close_predicitions


def main():
    # Preprocessed Dataset
    data = pd.read_csv("../data/reviews.csv")
    # data = pd.read_csv("../data/tweets.csv")
    print(data.head())
    print("\n")

    labels = data["sentiment"]
    features = data["tokens"]

    plot_wordcloud(features)
    plot_top_n_words(features)

    # Vectorization
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.3)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorized_train_features = vectorizer.fit_transform(train_features)
    vectorized_test_features = vectorizer.transform(test_features)

    # print(pd.DataFrame(vectorized_train_features.toarray(), columns=vectorizer.get_feature_names()))

    # Training
    classifier = RandomForestClassifier()
    classifier.fit(vectorized_train_features, train_labels)

    # Evaluation
    prediction = classifier.predict(vectorized_test_features)
    prediction_proba = classifier.predict_proba(vectorized_test_features)

    print(classification_report(test_labels, prediction))
    close_predictions = get_close_predicitions(prediction_proba)
    print(close_predictions)
    print("sum of close predictions:", len(close_predictions))

    plot_roc_curve(test_labels, prediction_proba)
    plot_confusion_matrix(test_labels, prediction)
    plt.show()


if __name__ == "__main__":
    main()
