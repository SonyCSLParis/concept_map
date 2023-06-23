from summary_generation import *
from triplets import *
from settings import *
from summary_aggregation import *
from concept_maps_construction import *
from evaluation_test_set import *
from coref import *

if __name__ == '__main__':

   ### MULTIDOCUMENT SUMMARISATION ----------------------------------------------------------

    ### TRAIN

    # generate summaries
    # parent_folder_path = CMAP_TRAIN_DIR
    # summarize_subfolders(parent_folder_path)

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


    ### EVALUATION ----------------------------------------------------------

    ## RUN ON TEST

    # generate summaries
    # parent_folder_path = CMAP_TEST_DIR
    # output_folder_path = SUMMARIES
    # summarize_subfolders(parent_folder_path)
    # summarize_subfolders(CMAP_TEST_DIR,SUMMARIES)

    # extract triplets from summaries
    # parent_folder_path = CMAP_TEST_DIR
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
    #

    # METEOR AND ROUGE

    # input_folder_path = OUTPUT_DIR_TRIPLETS_AGGREGATE_TEST
    # gold_folder_path = CMAP_FINAL_TEST_DIR
    # output_folder_path = OUTPUT_DIR_EVALUATION_CSV
    #
    # compute_metrics(input_folder_path, gold_folder_path, output_folder_path)

    # METEOR AND ROUGE CHATGPT
    # input_folder_path = SUMMARIES
    # output_folder_path = CMAP_TEST_GPT
    # aggregate_txt_files(input_folder_path, output_folder_path)

### SINGLEDOCUMENT SUMMARISATION ----------------------------------------------------------

    # generate summaries
    # parent_folder_path = WIKIPEDIA_TRAIN_DIR
    # summarize_folder(parent_folder_path)

   # coref
   # parent_folder_path = OUTPUT_DIR_SINGLE_DOC_DISJOINT
   # output_folder_path = OUTPUT_DIR_COREF
   # perform_coreference_resolution(parent_folder_path, output_folder_path)

   # extract triplets from summaries
   #  parent_folder_path = WIKIPEDIA_TRAIN_DIR
   #  output_folder_path = OUTPUT_DIR_SINGLE_DOC_DISJOINT
   #  extract_triplets_from_single_txt(parent_folder_path, output_folder_path)

   # generate graph
    folder_path = OUTPUT_DIR_SINGLE_DOC_DISJOINT
    output_folder_path = OUTPUT_DIR_GRAPH_SINGLE_DOC
    generate_graph_from_txt_files(folder_path, output_folder_path)