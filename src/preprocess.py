"""
Running spacy pipeline for preprocessing : remove stopwords,punctuation, double spaces and citations
"""
from settings import *
import regex as re


def preprocess_text(text):
    doc = nlp(text)

    cleaned_tokens = [token.text for token in doc if not (token.is_punct)]

    cleaned_text = ' '.join(cleaned_tokens).lower()

    cleaned_text_1 = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\[\d+\]\[\d+\]\[\d+\]', '', cleaned_text_1)

    return cleaned_text


def preprocess_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)

                with open(file_path, "r") as file:
                    text = file.read()
                    doc = nlp(text)
                    sentences = [sent.text.strip() for sent in doc.sents]

                    cleaned_text = [preprocess_text(sent) for sent in sentences]
                    my_string = '\n'.join(cleaned_text)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w") as output_file:
                    output_file.write(my_string)
