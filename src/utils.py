import os
import glob as gg
from itertools import chain
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
import numpy as np
from wordcloud import WordCloud

# constants
MODEL = "en_core_web_trf"
POSITIVE = 4
NEGATIVE = 0


def save_or_show(save, name):
    if save:
        plt.savefig(name)
    else:
        plt.show()


def save_or_print(save, content, mode="a"):
    if save:
        with open("metrics.txt", mode) as out:
            out.write(content + "\n")
    else:
        print(content)


def read(folder, sentiment):
    rows = []

    for path in gg.glob(folder):
        with open(path, "r") as file:
            data = file.read().replace("\n", "")
            name = os.path.splitext(os.path.basename(path))[0]
            rows.append([sentiment, name, data])

    return rows


def slang_to_text(text):
    url = "https://www.noslang.com/"
    data = {'action': 'translate', 'p': text, 'noswear': 'noswear', 'submit': 'Translate'}
    prefix_str = '<div class="translation-text">'
    postfix_str = '</div'
    non_found = "None of the words you entered are in our database. Found a word we're missing? " \
                "<a href=\"/addslang\">Add it to our dictionary</a>."

    request = requests.post(url, data)
    start_index = request.text.find(prefix_str) + len(prefix_str)
    end_index = start_index + request.text[start_index:].find(postfix_str)
    result = request.text[start_index:end_index]
    return text if result == non_found else result


def train_test_data(data1, data2=None):
    train_features1, test_features1, train_labels1, test_labels1 = \
        train_test_split(data1[0], data1[1], test_size=data1[2])

    if data2 is None:
        return train_features1, test_features1, train_labels1, test_labels1

    if data2[2] == 0:
        return train_features1, pd.concat([test_features1, data2[0]]), \
            train_labels1, pd.concat([test_labels1, data2[1]])

    train_features2, test_features2, train_labels2, test_labels2 = \
        train_test_split(data2[0], data2[1], test_size=data2[2])

    return pd.concat([train_features1, train_features2]), \
        pd.concat([test_features1, test_features2]), \
        pd.concat([train_labels1, train_labels2]), \
        pd.concat([test_labels1, test_labels2])


def plot_wordcloud(tokens, save=False):
    content = " ".join(list(chain(*(s.strip("][").split("', '") for s in tokens))))
    cloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(content)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    save_or_show(save, "wordcloud.png")


def plot_top_n_words(tokens, n_words=20, save=False):
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(tokens)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_words = words_freq[:n_words]
    data_frame = pd.DataFrame(top_words, columns=["token", "count"])
    data_frame.plot(x="token", y=["count"], kind="bar")
    save_or_show(save, "top_n_words.png")


def get_close_predicitions(y_pred, max_diff=0.05):
    close_predictions = []

    for _, value in enumerate(y_pred):
        diff = abs(value[1] - value[0])
        if diff <= max_diff:
            close_predictions.append([value[0], value[1]])

    return close_predictions


def plot_confusion_matrix(y_test, y_pred, save=False):
    matrix = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)
    visualization = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    visualization.plot()
    save_or_show(save, "cm.png")


def plot_roc_curve(y_test, y_pred, save=False):
    classes = np.unique(y_test)
    _, axes = plt.subplots()

    for i, value in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, i], pos_label=value)
        roc_auc = auc(fpr, tpr)
        visualization = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        visualization.plot(ax=axes, name=f"ROC Curve of class {value}")

    save_or_show(save, "roc.png")
