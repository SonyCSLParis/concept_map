""" Running whole pipeline """
from loguru import logger
from settings import WIKI_TRAIN, OUTPUT_PREPROCESSING, OUTPUT_EXTRACTION_ENTITY, \
     OUTPUT_GROUPING_ENTITY_SINGLE_FILES, OUTPUT_GROUPING_ENTITY_SUBFOLDERS, HTML_CONCEPTS
from preprocess import PreProcessor
from extract_entities import EntityExtractor
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

if __name__ == '__main__':
    # PREPROCESS
    # PRE_PROCESSOR = PreProcessor()
    # NAME = "Preprocessing"
    # log_start(name=NAME, input_=WIKI_TRAIN, output_=OUTPUT_PREPROCESSING)
    # PRE_PROCESSOR.main_folder(input_folder=WIKI_TRAIN, output_folder=OUTPUT_PREPROCESSING)
    # logger.success(LOGGER_INFO["done"].format(NAME))

    # EXTRACT ENTITY
    # ENTITY_EXTRACTOR = EntityExtractor()
    # NAME = "Entity Extraction"
    # log_start(name=NAME, input_=OUTPUT_PREPROCESSING, output_=OUTPUT_EXTRACTION_ENTITY)
    # ENTITY_EXTRACTOR.main_folder(
    #     input_folder=OUTPUT_PREPROCESSING, output_folder=OUTPUT_EXTRACTION_ENTITY)
    # logger.success(LOGGER_INFO["done"].format(NAME))

    #GROUP ENTITY
    # ENTITY_GROUPED = EntityGrouper()
    # NAME = "Entity Grouping"
    # log_start(name=NAME, input_=OUTPUT_EXTRACTION_ENTITY,
    #           output_=OUTPUT_GROUPING_ENTITY_SINGLE_FILES, inter_=OUTPUT_GROUPING_ENTITY_SUBFOLDERS)
    # ENTITY_GROUPED.main_folder(
    #     input_folder=OUTPUT_EXTRACTION_ENTITY,
    #     inter_folder=OUTPUT_GROUPING_ENTITY_SUBFOLDERS,
    #     output_folder=OUTPUT_GROUPING_ENTITY_SINGLE_FILES)

    #CREATE GRAPH
    # GRAPH_BUILDER = GraphBuilder()
    # NAME = "Graph building"
    # log_start(name=NAME, input_=OUTPUT_PREPROCESSING, output_=HTML_CONCEPTS)
    # GRAPH_BUILDER.main_folder(input_folder=OUTPUT_GROUPING_ENTITY_SUBFOLDERS,
    #                           output_folder=HTML_CONCEPTS)
    # logger.success(LOGGER_INFO["done"].format(NAME))

    # EXTRACT RELATIONS
