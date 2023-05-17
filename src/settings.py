import os

import spacy

nlp = spacy.load("it_core_news_lg")

RND_SEED = 42

ROOT_DIR = os.path.dirname(os.getcwd())

DATA_DIR = os.path.join(ROOT_DIR + '/data')
OUTPUT_DIR = os.path.join(ROOT_DIR + '/output')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)