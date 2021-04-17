import os
import glob as gg

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
