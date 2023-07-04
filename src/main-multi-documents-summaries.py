from summary_generation import *
from triplets import *
from settings import *
from summary_aggregation import *
from concept_maps_construction import *
from evaluation_test_set import *
from coref import *
from importance_ranking import *

### ---------------------------------------------------------- MULTIDOCUMENT SUMMARISATION ----------------------------------------------------------

if __name__ == '__main__':

    ### 1.1. TRAIN

    #generate summaries
    parent_folder_path = CMAP_TRAIN_DIR
    summarize_subfolders(parent_folder_path)

   #  parent_folder_path = SUMMARIES
   #  output_folder_path = IMPORTANCE_RANKING
   #  process_parent_folder(parent_folder_path,output_folder_path)

    # extract triplets from summaries
    # parent_folder_path = CMAP_TRAIN_DIR
    # output_folder_path = OUTPUT_DIR_TRIPLETS_DISJOINT
    # extract_triplets_from_summaries(parent_folder_path, output_folder_path)

    # aggregate triplets
    # parent_folder_path_disjoint = OUTPUT_DIR_TRIPLETS_DISJOINT
    # output_folder_path_aggregate = OUTPUT_DIR_TRIPLETS_AGGREGATE
    # combine_triplets_to_joint_summary(parent_folder_path_disjoint, output_folder_path_aggregate)

    # generate graph
    # folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE
    # output_folder_path = OUTPUT_DIR_GRAPH
    # generate_graph_from_txt_files(folder_path, output_folder_path)

    ### 1.2. TEST

    # generate summaries
    # parent_folder_path = CMAP_TEST_DIR
    # output_folder_path = SUMMARIES
    # summarize_subfolders(parent_folder_path,output_folder_path)
    # summarize_subfolders(CMAP_TEST_DIR,SUMMARIES)

    # sentence importances
    #  parent_folder_path = SUMMARIES
    #  output_folder_path = IMPORTANCE_RANKING
    #  process_parent_folder(parent_folder_path,output_folder_path)

    # extract triplets from summaries
    # parent_folder_path = IMPORTANCE_RANKING
    # output_folder_path = OUTPUT_DIR_TRIPLETS_DISJOINT_TEST
    # extract_triplets_from_summaries(parent_folder_path, output_folder_path)

    # aggregate triplets
    # parent_folder_path_disjoint = OUTPUT_DIR_TRIPLETS_DISJOINT_TEST
    # output_folder_path_aggregate = OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST
    # combine_triplets_to_joint_summary(parent_folder_path_disjoint, output_folder_path_aggregate)

    # generate graph
    # folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST
    # output_folder_path = OUTPUT_DIR_GRAPH_TEST
    # generate_graph_from_txt_files(folder_path, output_folder_path)

    ### 1.3. METEOR AND ROUGE

    # input_folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST
    # gold_folder_path = CMAP_FINAL_TEST_DIR
    # output_folder_path = OUTPUT_DIR_EVALUATION_CSV
    # compute_metrics(input_folder_path, gold_folder_path, output_folder_path)

    # METEOR AND ROUGE CHATGPT
    # input_folder_path = SUMMARIES
    # output_folder_path = CMAP_TEST_GPT
    # aggregate_txt_files(input_folder_path, output_folder_path)

