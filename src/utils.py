import os
import glob as gg
from itertools import chain
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import requests
import numpy as np
from wordcloud import WordCloud


# constants
MODEL = "en_core_web_trf"
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


def plot_wordcloud(tokens):
    content = " ".join(list(chain(*(s.strip("][").split("', '") for s in tokens))))
    cloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(content)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")


def plot_top_n_words(tokens, n_words=20):
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(tokens)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_words = words_freq[:n_words]
    data_frame = pd.DataFrame(top_words, columns=["token", "count"])
    data_frame.plot(x="token", y=["count"], kind="bar")


def get_close_predicitions(y_pred, max_diff=0.05):
    close_predictions = []

    for _, value in enumerate(y_pred):
        diff = abs(value[1]-value[0])
        if diff <= max_diff:
            close_predictions.append([value[0], value[1]])

    return close_predictions


def plot_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)
    visualization = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    visualization.plot()


def plot_roc_curve(y_test, y_pred):
    classes = np.unique(y_test)
    _, axes = plt.subplots()

    for i, value in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, i], pos_label=value)
        roc_auc = auc(fpr, tpr)
        visualization = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        visualization.plot(ax=axes, name=f"ROC Curve of class {value}")
