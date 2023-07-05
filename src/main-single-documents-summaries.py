from summary_generation import *
from coref import *
from importance_ranking import *

### ----------------------------------------------------------  SINGLE DOCUMNET SUMMARISATION ----------------------------------------------------------

if __name__ == '__main__':

    ### 2.1. TRAIN

    # generate summaries
    # parent_folder_path = WIKIPEDIA_TRAIN_DIR
    # output_folder_path = OUTPUT_DIR_SUMMARIES
    # summarize_folder(parent_folder_path, output_folder_path)

    # coref
    parent_folder_path = OUTPUT_DIR_SUMMARIES
    output_folder_path = OUTPUT_DIR_COREF
    perform_coreference_resolution(parent_folder_path, output_folder_path)

    # sentence importances
    # parent_folder_path = OUTPUT_DIR_COREF
    # output_folder_path = OUTPUT_IMPORTANCE_RANKING
    # process_parent_folder(parent_folder_path, output_folder_path)

    # extract triplets from summaries
    #  parent_folder_path = OUTPUT_IMPORTANCE_RANKING
    #  output_folder_path = OUTPUT_DIR_SINGLE_DOC_DISJOINT
    #  extract_triplets_from_summaries(parent_folder_path, output_folder_path)

    # generate graph
    #  folder_path = OUTPUT_DIR_SINGLE_DOC_DISJOINT
    #  output_folder_path = OUTPUT_DIR_GRAPH_SINGLE_DOC
    #  generate_graph_from_txt_files(folder_path, output_folder_path)