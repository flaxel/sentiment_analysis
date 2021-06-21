import argparse
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from utils import train_test_data, visualize_data, visualize_evaluate


def main(args):
    # Load Preprocessed Dataset
    reviews_data = pd.read_csv("../data/reviews.csv")
    tweets_data = pd.read_csv("../data/tweets.csv")

    visualize_data(reviews_data, tweets_data, args.save)

    # Split dataset
    train_features, test_features, train_labels, test_labels = train_test_data(
        (reviews_data["tokens"], reviews_data["sentiment"], 0.3),
        (tweets_data["tokens"], tweets_data["sentiment"], 0.5) if tweets_data is not None else None
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
    # ]

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
