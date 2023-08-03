import csv
import os

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def get_folder_name_of_txt_file(file_path):
    folder_name = os.path.dirname(file_path)
    return os.path.basename(folder_name)

def process_files_in_loop(folder_path):
    counter = 0

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            counter += 1
    return counter

def compute_metrics(input_folder_path, gold_folder_path, output_folder_path):
    metrics = []

    input_files = [f for f in os.listdir(input_folder_path) if f.endswith(".txt")]

    # gold_files = [f for f in os.listdir(gold_folder_path) if f.endswith(".cmap")]

    counter = 0
    for input_file_name in input_files:
        counter += 1
        file_number = input_file_name.split("-")[0]
        gold_file_name_txt = f"{file_number}.txt"
        if counter == 1 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/102/" + gold_file_name_txt)
        elif counter == 2 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/103/" + gold_file_name_txt)
        elif counter == 3 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/108/" + gold_file_name_txt)
        elif counter == 4 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/109/" + gold_file_name_txt)
        elif counter == 5 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/119/" + gold_file_name_txt)
        elif counter == 6 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/120/" + gold_file_name_txt)
        elif counter == 7 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/133/" + gold_file_name_txt)
        elif counter == 8 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/135/" + gold_file_name_txt)
        elif counter == 9 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/207/" + gold_file_name_txt)
        elif counter == 10 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/212/" + gold_file_name_txt)
        elif counter == 11 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/213/" + gold_file_name_txt)
        elif counter == 12 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/215/" + gold_file_name_txt)
        elif counter == 13 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/225/" + gold_file_name_txt)
        elif counter == 14 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/227/" + gold_file_name_txt)
        elif counter == 15 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/239/" + gold_file_name_txt)
        elif counter == 16 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/241/" + gold_file_name_txt)
        elif counter == 17 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/310/" + gold_file_name_txt)
        elif counter == 18 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/320/" + gold_file_name_txt)
        elif counter == 19 :
            gold_file_name_txt_path = os.path.join(gold_folder_path + "/320/" + gold_file_name_txt)
        folder_name = get_folder_name_of_txt_file(gold_file_name_txt_path)

        gold_file_name = f"{folder_name}.cmap"

        input_file_path = os.path.join(input_folder_path, input_file_name)
        gold_file_path = os.path.join(gold_folder_path, folder_name, gold_file_name)

        with open(input_file_path, "r") as input_file, open(gold_file_path, "r") as gold_file:
            input_text = input_file.read().strip()
            gold_text = gold_file.read().strip()

            hypothesis_tokens = word_tokenize(input_text)
            reference_tokens = word_tokenize(gold_text)

            meteor = meteor_score([reference_tokens], hypothesis_tokens)

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(input_text, gold_text)
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure

            metrics.append({
                'File': input_file_name,
                'METEOR': meteor,
                'ROUGE-1': rouge1,
                'ROUGE-2': rouge2,
                'ROUGE-L': rougeL
            })

        output_file_path = os.path.join(output_folder_path, "metrics_second_pipeline_lexrank_summary.csv")

        with open(output_file_path, "w", newline="") as csvfile:
            fieldnames = ['File', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(metrics)

        print(f"Metrics computed and saved in: {output_file_path}")


def aggregate_txt_files(parent_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the subfolders in the parent folder
    for foldername in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, foldername)

        # Check if the current item is a folder
        if os.path.isdir(folder_path):
            # Create a list to store the content of each summary file
            content_list = []

            # Iterate through the files in the subfolder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Check if the current file is a summary file
                if filename.endswith('-summary.txt') and os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        # Read the content of the summary file and append it to the list
                        content_list.append(file.read())

            # If there are summary files in the subfolder, create the aggregated file
            if content_list:
                output_filename = 'txt_with_all_text_' + foldername + '.txt'
                output_path = os.path.join(output_folder, output_filename)

                with open(output_path, 'w') as output_file:
                    # Write the aggregated content to the output file
                    output_file.write('\n'.join(content_list))

                print(f'Aggregated file saved: {output_filename}')
