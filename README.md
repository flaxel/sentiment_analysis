# Sentiment Analysis

This project contains all the data to analyze a text, whether it is positive or negative.

## Getting Started

These instructions will get you a copy of the project on your local machine for development.

### Prerequisites

The project is created with the programming language [Python3][python3] and the manager [pipenv][pipenv].

### Installing

To install all the necessary dependencies, the following command must be executed in the project.

```bash
# create virutal environment
python -m venv .venv

# activate virtual environment
.venv\Scripts\activate.bat # on windows
source .venv/bin/activate # on unix or macos

# installing necessary dependencies
pipenv install --dev

# optional: installing trained spacy pipeline
pipenv run python -m spacy download en_core_web_trf
```

### Usage

You can have a first look at the data if you execute the following command:

```bash
python first_look.py
```

If you want to pre-process the raw data, you can run one of the pre-processing scripts for the tweets or the film reviews.

```bash
python preprocess_tweets.py
python preprocess_reviews.py
```

The pre-processed data can be analysed and the graphics and results can be saved with the option `-s`.

```bash
python analysis_ensemble.py <-s>
```

[python3]: https://www.python.org/
[pipenv]: https://pypi.org/project/pipenv/
