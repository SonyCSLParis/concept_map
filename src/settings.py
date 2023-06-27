import os

import spacy

nlp = spacy.load("en_core_web_lg")
API_KEY_GPT =  "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"
RND_SEED = 42

ROOT_DIR = os.path.dirname(os.getcwd())
# print(ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR + '/src')
DATA_DIR = os.path.join(SRC_DIR + '/data')
# print(DATA_DIR)

WIKIPEDIA = os.path.join(DATA_DIR + '/wikipedia')
WIKIPEDIA_TRAIN_DIR = os.path.join(WIKIPEDIA + '/train')

CMAP_DIR = os.path.join(DATA_DIR + '/CMapSummaries')
CMAP_TEST_DIR = os.path.join(CMAP_DIR + '/test')
CMAP_TEST_GPT = os.path.join(CMAP_DIR + '/chat-gpt-test')

CMAP_FINAL_TEST_DIR = os.path.join(CMAP_TEST_DIR + '/test_cmap')

CMAP_TRAIN_DIR = os.path.join(CMAP_DIR + '/train')

OUTPUT_DIR = os.path.join(SRC_DIR + '/output_mds')
OUTPUT_DIR_TRIPLETS_DISJOINT = os.path.join(OUTPUT_DIR + '/triplets_disjoint')
OUTPUT_DIR_TRIPLETS_DISJOINT_TEST = os.path.join(OUTPUT_DIR + '/triplets_disjoint_test')
SUMMARIES = os.path.join(OUTPUT_DIR + '/summary-multiple-doc')
OUTPUT_DIR_TRIPLETS_AGGREGATE = os.path.join(OUTPUT_DIR + '/triplets_aggregate')
OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST = os.path.join(OUTPUT_DIR + '/triplets_aggregate_test')
IMPORTANCE_RANKING = os.path.join(OUTPUT_DIR + '/importance_ranking_folder')

OUTPUT_DIR_GRAPH = os.path.join(OUTPUT_DIR + '/graph_html_aggregated')
OUTPUT_DIR_GRAPH_TEST = os.path.join(OUTPUT_DIR + '/graph_html_aggregated_test')

OUTPUT_DIR_CONCEPT_MAPS = os.path.join(OUTPUT_DIR + '/concept_maps_visualisation')
OUTPUT_DIR_EVALUATION_CSV = os.path.join(OUTPUT_DIR + '/evaluation_csv')

OUTPUT_DIR_SINGLE_DOC = os.path.join(SRC_DIR + '/output_single_documents')
OUTPUT_DIR_SINGLE_DOC_DISJOINT = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/triplets_disjoint')
OUTPUT_DIR_GRAPH_SINGLE_DOC = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/graph_html_aggregated_test')
OUTPUT_DIR_COREF = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/coref')

list_dir = [DATA_DIR, OUTPUT_DIR_SINGLE_DOC_DISJOINT, OUTPUT_DIR_GRAPH_SINGLE_DOC, OUTPUT_DIR_TRIPLETS_DISJOINT_TEST,
            OUTPUT_DIR_EVALUATION_CSV, OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST, OUTPUT_DIR_GRAPH_TEST, OUTPUT_DIR,
            IMPORTANCE_RANKING]

for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)
