from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import os

def generate_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


def summarize_text_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                text = file.read()

            summary = generate_summary(text)
            summary_file_name = file_name.replace(".txt", "-summary.txt")
            summary_file_path = os.path.join(folder_path, summary_file_name)

            with open(summary_file_path, "w") as summary_file:
                summary_file.write(summary)

def summarize_text_files_output_path(folder_path, output_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                text = file.read()

            summary = generate_summary(text)
            summary_file_name = file_name.replace(".txt", "-summary.txt")
            summary_file_path = os.path.join(output_path, summary_file_name)

            with open(summary_file_path, "w") as summary_file:
                summary_file.write(summary)


def summarize_subfolders(parent_folder_path,output_path):
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            summarize_text_files_output_path(folder_path,output_path)

def summarize_folder(folder_path,output_path):
    if os.path.isdir(folder_path):
        summarize_text_files_output_path(folder_path,output_path)
