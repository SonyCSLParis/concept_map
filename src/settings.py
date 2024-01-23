"""
Global variables for the module
"""
import os

import spacy

nlp = spacy.load("en_core_web_sm")
API_KEY_GPT = ""
RND_SEED = 42

ROOT_DIR = os.path.dirname(os.getcwd())
SRC_DIR = os.path.join(ROOT_DIR + '/src')

# DATA DIR
DATA_DIR = os.path.join(SRC_DIR + '/data')

# corpora Falke
CORPORA_FALKE = os.path.join(DATA_DIR + '/Corpora_Falke')
ACL = os.path.join(CORPORA_FALKE + '/ACL')
BIOLOGY = os.path.join(CORPORA_FALKE + '/Biology')
WIKI = os.path.join(CORPORA_FALKE + '/Wiki')
WIKI_TRAIN = os.path.join(WIKI + '/train')
WIKI_TEST = os.path.join(WIKI + '/test')
WIKI_FINAL_TEST_DIR = os.path.join(WIKI_TEST + '/final_test')

# other corpora
CMAP_DIR = os.path.join(DATA_DIR + '/CMapSummaries')
CMAP_TEST_DIR = os.path.join(CMAP_DIR + '/test')
CMAP_TEST_GPT = os.path.join(CMAP_DIR + '/chat-gpt-test')
CMAP_FINAL_TEST_DIR = os.path.join(CMAP_TEST_DIR + '/final_test')
CMAP_TRAIN_DIR = os.path.join(CMAP_DIR + '/train')

# OUTPUTS
OUTPUT_DIR = os.path.join(SRC_DIR + '/output')
OUTPUT_PREPROCESSING = os.path.join(OUTPUT_DIR + '/output_preprocessing')
OUTPUT_EXTRACTION_ENTITY = os.path.join(OUTPUT_DIR + '/output_extraction_entity')
OUTPUT_GROUPING_ENTITY_SINGLE_FILES = os.path.join(OUTPUT_DIR + '/output_grouping_entity_single')
OUTPUT_TRIPLETS_FINE_TUNE = os.path.join(OUTPUT_DIR + '/triplets')
OUTPUT_GROUPING_ENTITY_SUBFOLDERS = os.path.join(OUTPUT_DIR + '/output_grouping_entity_subfolders')
OUTPUT_TRIPLETS_PREPORCESSED = os.path.join(OUTPUT_DIR + '/triplets_preprocessed')

HTML_CONCEPTS = os.path.join(OUTPUT_DIR + '/html_images')

list_dir = [DATA_DIR,HTML_CONCEPTS, OUTPUT_TRIPLETS_PREPORCESSED, OUTPUT_TRIPLETS_FINE_TUNE, OUTPUT_DIR, OUTPUT_EXTRACTION_ENTITY, OUTPUT_PREPROCESSING,OUTPUT_GROUPING_ENTITY_SINGLE_FILES,OUTPUT_GROUPING_ENTITY_SUBFOLDERS]
for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)
