import argparse

import numpy as np
from transformers import pipeline
from utils import NEGATIVE, POSITIVE, visualize_evaluate, read_tweets


def main(args):
    tweets_df = read_tweets(500)

    print(tweets_df.head())
    print("\n")

    sentences = tweets_df["text"].tolist()
    labels = tweets_df["target"].tolist()

    # model = "distilbert-base-uncased-finetuned-sst-2-english"
    # model = "siebert/sentiment-roberta-large-english"
    # classifier = pipeline("sentiment-analysis", model=model)
    # results = classifier(sentences)

    # scores = np.array([(
    #     [1-e["score"], e["score"]] if e["label"] == "POSITIVE" else [e["score"], 1-e["score"]]
    # ) for e in results])
    # predicted_labels = [(POSITIVE if e["label"] == "POSITIVE" else NEGATIVE) for e in results]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = classifier(sentences, ["POSITIVE", "NEGATIVE"])

    scores = np.array([(
        e["scores"] if e["labels"] == ["NEGATIVE", "POSITIVE"] else [e["scores"][1], e["scores"][0]]
    ) for e in results])
    predicted_labels = [(
        POSITIVE if e["labels"][np.argmax(e["scores"])] == "POSITIVE" else NEGATIVE
    ) for e in results]

    visualize_evaluate(predicted_labels, scores, labels, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", action="store_true", help="save action flag")

    main(parser.parse_args())
