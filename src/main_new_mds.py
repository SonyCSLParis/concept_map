from settings import *
from importance_ranking import *
from preprocess import *
from coref import *
from extract_triplets import *
from triplets import *

if __name__ == '__main__':
    # PREPROCESS
    # parent_folder_path = WIKI_TRAIN
    # output_folder_path = OUTPUT_DIR_PREPROCESS
    # preprocess_folder(parent_folder_path,output_folder_path)

    # COREF
    # parent_folder_path = OUTPUT_DIR_PREPROCESS
    # output_folder_path = OUTPUT_DIR_COREF_MULTIPLE_TRAIN_WIKI
    # perform_coreference_resolution(parent_folder_path, output_folder_path)

    # extract triplets from summaries
    # parent_folder_path = OUTPUT_DIR_COREF_MULTIPLE_TRAIN_WIKI
    # output_folder_path = OUTPUT_DIR_TRIPLETS_DISJOINT_WIKI
    # preprocess_folder(parent_folder_path, output_folder_path)

    parent_folder_path = OUTPUT_DIR_COREF_MULTIPLE_TRAIN_WIKI
    output_folder_path = OUTPUT_DIR_TRIPLETS_DISJOINT_WIKI
    process_folder(parent_folder_path, output_folder_path)

    # aggregate triplets
    # parent_folder_path_disjoint = OUTPUT_DIR_TRIPLETS_DISJOINT
    # output_folder_path_aggregate = OUTPUT_DIR_TRIPLETS_AGGREGATE
    # combine_triplets_to_joint_summary(parent_folder_path_disjoint, output_folder_path_aggregate)

    # generate graph
    # folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE
    # output_folder_path = OUTPUT_DIR_GRAPH
    # generate_graph_from_txt_files(folder_path, output_folder_path)

    # input_folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST
    # gold_folder_path = CMAP_FINAL_TEST_DIR
    # output_folder_path = OUTPUT_DIR_EVALUATION_CSV
    # compute_metrics(input_folder_path, gold_folder_path, output_folder_path)
