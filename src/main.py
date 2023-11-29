from settings import *
from preprocess import *
from group_entities import *
from extract_entities import *
from graph import *
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

    # EXTRACT RELATIONS
