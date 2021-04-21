import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from utils import slang_to_text, POSITIVE, NEGATIVE, MODEL


def preprocess(index, text):
    # Lowercase
    text = text.lower()

    # Removing URL links
    text = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+))", "", text)

    # Removing hashtag
    text = re.sub(r"[#]", "", text)

    # Removing usernames
    text = re.sub(r'@(\w+)', "", text)

    # Expanding acronyms
    text = slang_to_text(text)

    # Replacing negative mentions
    text = re.sub(r"won't", r"will not", text)
    text = re.sub(r"can't", r"cannot", text)
    text = re.sub(r"n't", r"not", text)

    # Reverting words that contain repeated letters
    text = re.sub(r"(.)\1{2,}", r"\1\1\1", text)

    # Tokenization
    nlp = spacy.load(MODEL)
    doc = nlp(text)
    text = [token.text for token in doc]

    # Lowercase
    text = [token.lower() for token in text]

    # Removing punctuations
    text = [token for token in text if not (re.match(r"[^\w\s]", token) and len(token) == 1)]
    text = [token for token in text if token not in ["..", "..."]]

    # Removing numbers
    text = [token for token in text if not re.match(r"\d+", token)]

    # Removing stop words
    text = [token for token in text if token not in STOP_WORDS]

    if index % 100 == 0:
        print("preprocess for dataset", index, "is running...")

    return text


def main():
    tweets_df = pd.read_csv(
        "../data/kaggle_archive/training.1600000.processed.noemoticon.csv",
        names=["target", "ids", "date", "flag", "user", "text"]
    ).sample(frac=1)

    tweets_reduced_df = tweets_df[tweets_df["target"] == NEGATIVE].head(1000) \
        .append(tweets_df[tweets_df["target"] == POSITIVE].head(1000))

    print(tweets_reduced_df.head())
    print("\n")

    rows = []
    index = 0
    for _, row in tweets_reduced_df.iterrows():
        preprocessed_feature = preprocess(index, row[5])
        rows.append([row[0], preprocessed_feature])
        index += 1

    preprocessed_reviews_df = pd.DataFrame(rows, columns=["sentiment", "tokens"])
    preprocessed_reviews_df.to_csv("../data/tweets.csv", index=False)


if __name__ == "__main__":
    # configuration of pandas
    pd.set_option("display.max_columns", None)

    #main()
    print(preprocess(0, "Ugh. A fucked up dog and a shitty friend. Who else wants to jump on the swear train?"))