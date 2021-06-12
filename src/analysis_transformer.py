import argparse

from numpy import array
from transformers import pipeline
from utils import NEGATIVE, POSITIVE, visualize_evaluate, read_tweets


def main(args):
    tweets_df = read_tweets(500)

    print(tweets_df.head())
    print("\n")

    sentences = tweets_df["text"].tolist()
    labels = tweets_df["target"].tolist()

    # model = "distilbert-base-uncased-finetuned-sst-2-english"
    model = "siebert/sentiment-roberta-large-english"
    classifier = pipeline("sentiment-analysis", model=model)

    results = classifier(sentences)
    scores = array([(
        [1-e["score"], e["score"]] if e["label"] == "POSITIVE" else [e["score"], 1-e["score"]]
    ) for e in results])
    predicted_labels = [(POSITIVE if e["label"] == "POSITIVE" else NEGATIVE) for e in results]

    visualize_evaluate(predicted_labels, scores, labels, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", action="store_true", help="save action flag")

    main(parser.parse_args())