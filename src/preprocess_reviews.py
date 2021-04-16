import glob as gg
import os
from datetime import datetime
import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# constants
POSITIVE = 4
NEGATIVE = 0


def read(folder, sentiment):
    rows = []

    for path in gg.glob(folder):
        with open(path, "r") as file:
            data = file.read().replace("\n", "")
            name = os.path.splitext(os.path.basename(path))[0]
            rows.append([sentiment, name, data])

    return rows


def preprocess(index, text):
    # Tokenization & Lemmatization
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    text = [token.lemma_ for token in doc]

    # Lowercase
    text = [token.lower() for token in text]

    # Remove Punctuations
    text = [token for token in text if not re.match(r"[^\w\s]", token)]

    # Remove Numbers
    text = [token for token in text if not re.match(r"\d+", token)]

    # Stop Words
    text = [token for token in text if token not in STOP_WORDS]

    # Remove words less than three characters
    text = [token for token in text if not len(token) < 3]

    if index % 100 == 0:
        print("preprocess for dataset", index, "is running...")

    return text


def main():
    rows = read("../data/review_polarity/txt_sentoken/pos/*.txt", POSITIVE) + \
        read("../data/review_polarity/txt_sentoken/neg/*.txt", NEGATIVE)

    reviews_df = pd.DataFrame(rows, columns=["sentiment", "id", "text"])

    print(reviews_df.head())
    print("\n")

    rows = []
    for index, row in reviews_df.iterrows():
        preprocessed_feature = preprocess(index, row[2])
        rows.append([row[0], preprocessed_feature])

    preprocessed_reviews_df = pd.DataFrame(rows, columns=["sentiment", "tokens"])
    preprocessed_reviews_df.to_csv("../data/reviews.csv", index=False)


if __name__ == "__main__":
    # configuration of pandas
    pd.set_option("display.max_columns", None)

    start = datetime.now()
    print("Start:", start, "\n")

    main()

    end = datetime.now()
    print("\nEnd:", end)

    print("Runtime:", end - start)
