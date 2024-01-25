from settings import *
from preprocess import *
from group_entities import *
from extract_entities import *
from graph import *
from extract_relations_fine_tuned import *
from evaluation_test_set import compute_metrics
from extract_entities import *

if __name__ == '__main__':

    # PREPROCESS
    # parent_folder_path = BIO_TEST
    # output_folder_path = OUTPUT_PREPROCESSING_TEST + '/output_preprocessing_test_bio.csv'
    # preprocess_bio(parent_folder_path, output_folder_path)
    # preprocess_folder(parent_folder_path,output_folder_path)

    # EXTRACT ENTITY
    # csv_path = OUTPUT_PREPROCESSING_TEST + '/output_preprocessing_test_bio.csv'
    # output_folder = OUTPUT_EXTRACTION_ENTITY_TEST + '/output_extraction_entity_bio.csv'
    # process_csv_bio(csv_path, output_folder)

    # EXTRACT RELATIONSHIP
    # entity_path = OUTPUT_EXTRACTION_ENTITY_TEST
    # text_path = OUTPUT_PREPROCESSING_TEST
    # output_path = OUTPUT_TRIPLETS_FINE_TUNE_TEST
    # entity_extracted_rebel(entity_path, text_path, output_path)

    # PREPROCESS OUTPUT RIPLETS
    # entity_path = OUTPUT_TRIPLETS_FINE_TUNE_TEST
    # output_path = OUTPUT_TRIPLETS_PREPORCESSED
    # process_output_file(entity_path, output_path)

    # EVALUATE
    input_folder_path = OUTPUT_TRIPLETS_PREPORCESSED
    gold_folder_path = WIKI_TEST
    output_folder_path = OUTPUT_EVALUATION
    compute_metrics(input_folder_path, gold_folder_path, output_folder_path)
    # compute_metrics_falke(input_folder_path, gold_folder_path, output_folder_path)

    #----------------
    # parent_folder_path = OUTPUT_EXTRACTION_ENTITY
    # output_folder_path = OUTPUT_GROUPING_ENTITY_SINGLE_FILES
    # set_entities(parent_folder_path, output_folder_path)
    # new_output_folder_path = OUTPUT_GROUPING_ENTITY_SUBFOLDERS
    # get_unique_entities_from_files_in_subfolders(output_folder_path, new_output_folder_path)

    #CREATE GRAPH
    # parent_folder_path = OUTPUT_GROUPING_ENTITY_SUBFOLDERS
    # output_folder_path = HTML_CONCEPTS
    # create_graph(parent_folder_path, output_folder_path)

