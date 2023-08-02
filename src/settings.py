import os

import spacy

nlp = spacy.load("en_core_web_lg")
API_KEY_GPT = ""
RND_SEED = 42

ROOT_DIR = os.path.dirname(os.getcwd())
# print(ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR + '/src')
DATA_DIR = os.path.join(SRC_DIR + '/data')
# print(DATA_DIR)

# corpora Falke
CORPORA_FALKE = os.path.join(DATA_DIR + '/Corpora_Falke')
ACL = os.path.join(CORPORA_FALKE + '/ACL')
BIOLOGY = os.path.join(CORPORA_FALKE + '/Biology')
WIKI = os.path.join(CORPORA_FALKE + '/Wiki')
WIKI_TRAIN = os.path.join(WIKI + '/train')
WIKI_TEST = os.path.join(WIKI + '/test')
WIKI_FINAL_TEST_DIR = os.path.join(WIKI_TEST + '/final_test')

# other corpora

WIKIPEDIA = os.path.join(DATA_DIR + '/wikipedia')
WIKIPEDIA_TRAIN_DIR = os.path.join(WIKIPEDIA + '/train')

CMAP_DIR = os.path.join(DATA_DIR + '/CMapSummaries')
CMAP_TEST_DIR = os.path.join(CMAP_DIR + '/test')
CMAP_TEST_GPT = os.path.join(CMAP_DIR + '/chat-gpt-test')

CMAP_FINAL_TEST_DIR = os.path.join(CMAP_TEST_DIR + '/final_test')

CMAP_TRAIN_DIR = os.path.join(CMAP_DIR + '/train')

# MDS

OUTPUT_DIR = os.path.join(SRC_DIR + '/output_mds')
OUTPUT_DIR_TRIPLETS_DISJOINT = os.path.join(OUTPUT_DIR + '/triplets_disjoint')
OUTPUT_DIR_TRIPLETS_DISJOINT_WIKI = os.path.join(OUTPUT_DIR + '/triplets_disjoint_wiki')

OUTPUT_DIR_TRIPLETS_DISJOINT_TEST = os.path.join(OUTPUT_DIR + '/triplets_disjoint_test')
OUTPUT_DIR_TRIPLETS_DISJOINT_TEST_WIKI = os.path.join(OUTPUT_DIR + '/triplets_disjoint_test_wiki')

SUMMARIES = os.path.join(OUTPUT_DIR + '/summary-multiple-doc')

SUMMARIES_TRAIN = os.path.join(SUMMARIES + '/summary_train')
SUMMARIES_TRAIN_WIKI = os.path.join(SUMMARIES + '/summary_train_wiki')

SUMMARIES_TEST = os.path.join(SUMMARIES + '/summary_test')
SUMMARIES_TEST_WIKI = os.path.join(SUMMARIES + '/summary_test_wiki')

SUMMARIES_TRAIN_GPT = os.path.join(SUMMARIES + '/summary_train_GPT')
SUMMARIES_TEST_GPT = os.path.join(SUMMARIES + '/summary_test_GPT')
SUMMARIES_TEST_GPT_WIKI = os.path.join(SUMMARIES + '/summary_test_GPT_wiki')

OUTPUT_DIR_TRIPLETS_AGGREGATE = os.path.join(OUTPUT_DIR + '/triplets_aggregate')
OUTPUT_DIR_TRIPLETS_AGGREGATE_WIKI = os.path.join(OUTPUT_DIR + '/triplets_aggregate_wiki')
OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST = os.path.join(OUTPUT_DIR + '/triplets_aggregate_test')
OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST_WIKI = os.path.join(OUTPUT_DIR + '/triplets_aggregate_test_wiki')

IMPORTANCE_RANKING = os.path.join(OUTPUT_DIR + '/importance_ranking_folder')
IMPORTANCE_RANKING_TEST = os.path.join(IMPORTANCE_RANKING + '/test')
IMPORTANCE_RANKING_TEST_WIKI = os.path.join(IMPORTANCE_RANKING + '/test_wiki')
IMPORTANCE_RANKING_TRAIN = os.path.join(IMPORTANCE_RANKING + '/train')
IMPORTANCE_RANKING_TRAIN_WIKI = os.path.join(IMPORTANCE_RANKING + '/train_wiki')

OUTPUT_DIR_GRAPH = os.path.join(OUTPUT_DIR + '/graph_html_aggregated')
OUTPUT_DIR_GRAPH_TEST = os.path.join(OUTPUT_DIR + '/graph_html_aggregated_test')
OUTPUT_DIR_GRAPH_TEST_WIKI = os.path.join(OUTPUT_DIR + '/graph_html_aggregated_test_wiki')

OUTPUT_DIR_GRAPH_WIKI = os.path.join(OUTPUT_DIR + '/graph_html_aggregated_wiki')

OUTPUT_DIR_CONCEPT_MAPS = os.path.join(OUTPUT_DIR + '/concept_maps_visualisation')
OUTPUT_DIR_EVALUATION_CSV = os.path.join(OUTPUT_DIR + '/evaluation_csv')
OUTPUT_DIR_EVALUATION_CSV_WIKI = os.path.join(OUTPUT_DIR + '/evaluation_csv_wiki')

OUTPUT_DIR_COREF_MULTIPLE = os.path.join(OUTPUT_DIR + '/coref')
OUTPUT_DIR_COREF_MULTIPLE_TEST = os.path.join(OUTPUT_DIR_COREF_MULTIPLE + '/test')
OUTPUT_DIR_COREF_MULTIPLE_TEST_WIKI = os.path.join(OUTPUT_DIR_COREF_MULTIPLE + '/test_wiki')

OUTPUT_DIR_COREF_MULTIPLE_TRAIN = os.path.join(OUTPUT_DIR_COREF_MULTIPLE + '/train')
OUTPUT_DIR_COREF_MULTIPLE_TRAIN_WIKI = os.path.join(OUTPUT_DIR_COREF_MULTIPLE + '/train_wiki')

OUTPUT_DIR_DB_ENTITY = os.path.join(OUTPUT_DIR, 'db_entities')
OUTPUT_DIR_DB_ENTITY_WIKI = os.path.join(OUTPUT_DIR, 'db_entities_wiki')

# SINGLES

OUTPUT_DIR_SINGLE_DOC = os.path.join(SRC_DIR + '/output_single_documents')
OUTPUT_DIR_SINGLE_DOC_DISJOINT = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/triplets_disjoint')
OUTPUT_DIR_GRAPH_SINGLE_DOC = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/graph_html_aggregated_test')
OUTPUT_DIR_COREF = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/coref')
OUTPUT_DIR_SUMMARIES = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/summaries')
OUTPUT_DIR_SUMMARIES_GPT = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/summary_train_GPT')
OUTPUT_IMPORTANCE_RANKING = os.path.join(OUTPUT_DIR_SINGLE_DOC + '/single_importance_ranking_folder')
OUTPUT_DIR_SINGLE_DB_ENTITY = os.path.join(OUTPUT_DIR_SINGLE_DOC, 'db_entities')
OUTPUT_DIR_SINGLE_DOC_DISJOINT_WITH_IRIS = os.path.join(OUTPUT_DIR_SINGLE_DOC, 'triplets_disjoint_with_iris')

list_dir = [DATA_DIR, OUTPUT_DIR_SINGLE_DOC_DISJOINT, OUTPUT_DIR_GRAPH_SINGLE_DOC, OUTPUT_DIR_TRIPLETS_DISJOINT_TEST,
            OUTPUT_DIR_EVALUATION_CSV, OUTPUT_IMPORTANCE_RANKING, OUTPUT_DIR_SUMMARIES,
            OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST, OUTPUT_DIR_GRAPH_TEST, OUTPUT_DIR,
            IMPORTANCE_RANKING, SUMMARIES_TRAIN_GPT, OUTPUT_DIR_SUMMARIES_GPT, SUMMARIES_TEST_GPT, SUMMARIES_TRAIN,
            SUMMARIES_TEST, IMPORTANCE_RANKING_TEST_WIKI, WIKI_FINAL_TEST_DIR, OUTPUT_DIR_COREF_MULTIPLE_TEST_WIKI, SUMMARIES_TEST_WIKI,
            SUMMARIES_TEST_GPT_WIKI, SUMMARIES_TRAIN_WIKI, OUTPUT_DIR_GRAPH_WIKI, OUTPUT_DIR_TRIPLETS_AGGREGATE_WIKI,
            OUTPUT_DIR_TRIPLETS_DISJOINT_WIKI, OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST_WIKI, OUTPUT_DIR_GRAPH_TEST_WIKI,
            OUTPUT_DIR_DB_ENTITY_WIKI, OUTPUT_DIR_TRIPLETS_DISJOINT_TEST_WIKI, IMPORTANCE_RANKING_TRAIN_WIKI]

for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)
