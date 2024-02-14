""" Running whole pipeline """
from loguru import logger
from settings import WIKI_TRAIN, OUTPUT_PREPROCESSING, OUTPUT_EXTRACTION_ENTITY, \
     OUTPUT_GROUPING_ENTITY_SINGLE_FILES, OUTPUT_GROUPING_ENTITY_SUBFOLDERS, HTML_CONCEPTS
from preprocess import PreProcessor
from group_entities import EntityGrouper
from graph import GraphBuilder

LOGGER_INFO = {
    "start": "[{}] Starting",
    "input": "[{}] Input folder: {}",
    "inter": "[{}] Inter folder: {}",
    "output": "[{}] Output folder: {}",
    "done": "[{}] Done",
}

def log_start(name: str, input_: str, output_: str, inter_: str = ""):
    """ Logging info when one component is started """
    logger.info(LOGGER_INFO["start"].format(name))
    logger.info(LOGGER_INFO["input"].format(name, input_))
    if inter_:
        logger.info(LOGGER_INFO["inter"].format(name, inter_))
    logger.info(LOGGER_INFO["output"].format(name, output_))

from extract_relations_fine_tuned import *
from evaluation_test_set import compute_metrics, compute_metrics_falke, aggregate_txt_files

if __name__ == '__main__':

    # PREPROCESS
    # parent_folder_path = WIKI_TRAIN
    # output_folder_path = OUTPUT_PREPROCESSING
    # preprocess_folder(parent_folder_path,output_folder_path)

    # EXTRACT ENTITY
    # parent_folder_path = OUTPUT_PREPROCESSING
    # output_folder_path = OUTPUT_EXTRACTION_ENTITY
    # extract_dbpedia_spotlight_entities(parent_folder_path, output_folder_path)

    #GROUP ENTITY
    # parent_folder_path = OUTPUT_EXTRACTION_ENTITY
    # output_folder_path = OUTPUT_GROUPING_ENTITY_SINGLE_FILES
    # set_entities(parent_folder_path, output_folder_path)
    # new_output_folder_path = OUTPUT_GROUPING_ENTITY_SUBFOLDERS
    # get_unique_entities_from_files_in_subfolders(output_folder_path, new_output_folder_path)

    #CREATE GRAPH
    parent_folder_path = OUTPUT_GROUPING_ENTITY_SUBFOLDERS
    output_folder_path = HTML_CONCEPTS
    create_graph(parent_folder_path, output_folder_path)

