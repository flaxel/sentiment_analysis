import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from utils import slang_to_text, MODEL, read_tweets


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

    # Tokenization
    nlp = spacy.load(MODEL)
    doc = nlp(text)
    text = [token.lemma_ for token in doc]

    # Lowercase
    text = [token.lower() for token in text]

    # Removing punctuations
    text = [token for token in text if not (re.match(r"[^\w\s]", token) and len(token) == 1)]
    text = [token for token in text if token not in ["..", "..."]]

    # Removing stop words
    text = [token for token in text if token not in STOP_WORDS]

    # Remove words less than three characters
    text = [token for token in text if not len(token) < 3]

    # Removing numbers
    text = [token for token in text if not re.match(r"\d+", token)]

    if index % 100 == 0:
        print("preprocess for dataset", index, "is running...")

    return text


def main():
    tweets_df = read_tweets(1000)

    print(tweets_df.head())
    print("\n")

    rows = []
    index = 0
    for _, row in tweets_df.iterrows():
        preprocessed_feature = preprocess(index, row[5])
        rows.append([row[0], preprocessed_feature])
        index += 1

    preprocessed_reviews_df = pd.DataFrame(rows, columns=["sentiment", "tokens"])
    preprocessed_reviews_df.to_csv("../data/tweets.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    main()
