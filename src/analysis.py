import argparse
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from utils import plot_confusion_matrix, plot_roc_curve, plot_wordcloud, plot_top_n_words
from utils import get_close_predicitions, train_test_data, save_or_print


def visualize_data(reviews_data, tweets_data, save=False):
    if reviews_data is not None:
        save_or_print(save, 27*"#" + " REVIEWS " + 27*"#")
        save_or_print(save, reviews_data.head().to_string())

    if tweets_data is not None:
        save_or_print(save, 27*"#" + " TWEETS " + 28*"#")
        save_or_print(save, tweets_data.head().to_string())

    if reviews_data is None:
        plot_wordcloud(tweets_data["tokens"], save=save)
        plot_top_n_words(tweets_data["tokens"], save=save)
    elif tweets_data is None:
        plot_wordcloud(reviews_data["tokens"], save=save)
        plot_top_n_words(reviews_data["tokens"], save=save)
    else:
        features = pd.concat([reviews_data["tokens"], tweets_data["tokens"]])
        plot_wordcloud(features, save=save)
        plot_top_n_words(features, save=save)


def visualize_evaluate(prediction, prediction_proba, test_labels, save=False):
    report = classification_report(test_labels, prediction)
    close_predictions = get_close_predicitions(prediction_proba)

    save_or_print(save, report)
    save_or_print(save, ", ".join(str(e) for e in close_predictions))
    save_or_print(save, "sum of close predictions: " + str(len(close_predictions)))

    plot_roc_curve(test_labels, prediction_proba, save=save)
    plot_confusion_matrix(test_labels, prediction, save=save)


def main(args):
    # Load Preprocessed Dataset
    reviews_data = pd.read_csv("../data/reviews.csv")
    tweets_data = pd.read_csv("../data/tweets.csv")

    visualize_data(reviews_data, tweets_data, args.save)

    # Split dataset
    train_features, test_features, train_labels, test_labels = train_test_data(
        (reviews_data["tokens"], reviews_data["sentiment"], 0.3),
        (tweets_data["tokens"], tweets_data["sentiment"], 0.3) if tweets_data is not None else None
    )

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    vectorized_train_features = vectorizer.fit_transform(train_features)
    vectorized_test_features = vectorizer.transform(test_features)

    # Training
    # estimators = [
    #    ("svm", SVC(probability=True)),
    #    ("lr", LogisticRegression()),
    #    ("nb", MultinomialNB())
    #]

    classifier = BaggingClassifier(base_estimator=SVC(probability=True))
    # classifier = RandomForestClassifier()
    # classifier = AdaBoostClassifier(base_estimator=SVC(probability=True))
    # classifier = StackingClassifier(estimators=estimators)
    classifier.fit(vectorized_train_features, train_labels)

    # Evaluation
    prediction = classifier.predict(vectorized_test_features)
    prediction_proba = classifier.predict_proba(vectorized_test_features)

    visualize_evaluate(prediction, prediction_proba, test_labels, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", action="store_true", help="save action flag")

    main(parser.parse_args())
