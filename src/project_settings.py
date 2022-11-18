import os
from os import path, makedirs

import spacy

"""
Global variable in this sub-module
"""
SRC = os.path.dirname(os.path.realpath(__file__))
# ROOT = os.path.dirname(os.path.dirname(SRC))
DATA_DIR = os.path.join(SRC, 'data')
OUTPUT = os.path.join(SRC, 'output')
PNG = os.path.join(SRC, 'png')
DEFAULT_MODEL = "it_core_news_sm"
nlp = spacy.load(DEFAULT_MODEL)

"""
Spacy select language
"""
LANGUAGE = "italian"
SENTENCES_COUNT = 10

"""
Make dirs
"""

list_dirs = [DATA_DIR, OUTPUT, PNG]

for x in list_dirs:
    if not path.exists(x):
        makedirs(x)

if __name__ == '__main__':
    print(PNG)
