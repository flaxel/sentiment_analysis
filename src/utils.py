import os
import glob as gg
import requests

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
