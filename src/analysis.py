import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import plot_confusion_matrix, plot_roc_curve, plot_wordcloud, plot_top_n_words
from utils import get_close_predicitions, train_test_data


def main():
    # Preprocessed Dataset
    reviews_data = pd.read_csv("../data/reviews.csv")
    tweets_data = pd.read_csv("../data/tweets.csv")

    if reviews_data is not None:
        print(27*"#", "REVIEWS", 27*"#")
        print(reviews_data.head())
        print("\n")

    if tweets_data is not None:
        print(27*"#", "TWEETS", 28*"#")
        print(tweets_data.head())
        print("\n")

    # Plot dataset
    if reviews_data is None:
        plot_wordcloud(tweets_data["tokens"])
        plot_top_n_words(tweets_data["tokens"])
    elif tweets_data is None:
        plot_wordcloud(reviews_data["tokens"])
        plot_top_n_words(reviews_data["tokens"])
    else:
        features = pd.concat([reviews_data["tokens"], tweets_data["tokens"]])
        plot_wordcloud(features)
        plot_top_n_words(features)

    # Split dataset
    train_features, test_features, train_labels, test_labels = train_test_data(
        (reviews_data["tokens"], reviews_data["sentiment"], 0.3),
        (tweets_data["tokens"], tweets_data["sentiment"], 0.3) if tweets_data is not None else None
    )

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorized_train_features = vectorizer.fit_transform(train_features)
    vectorized_test_features = vectorizer.transform(test_features)

    # f = pd.DataFrame(vectorized_train_features.toarray(), columns=vectorizer.get_feature_names())
    # print(f)

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
