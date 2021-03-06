import pandas as pd
from utils import read, POSITIVE, NEGATIVE


def main():
    # Tweets View
    tweets_df = pd.read_csv(
        "../data/kaggle_archive/training.1600000.processed.noemoticon.csv",
        names=["target", "ids", "date", "flag", "user", "text"]
    )

    print(tweets_df.info())
    print("\n")

    print(tweets_df.head())
    print("\n")

    print(tweets_df.groupby(["target"]).size())
    print("\n")

    # Movie View
    rows = read("../data/review_polarity/txt_sentoken/pos/*.txt", POSITIVE) + \
        read("../data/review_polarity/txt_sentoken/neg/*.txt", NEGATIVE)

    reviews_df = pd.DataFrame(rows, columns=["sentiment", "id", "text"])

    print(reviews_df.info())
    print("\n")

    print(reviews_df.head())
    print("\n")

    print(reviews_df.groupby(["sentiment"]).size())


if __name__ == "__main__":
    # configuration of pandas
    pd.set_option("display.max_columns", None)

    main()
