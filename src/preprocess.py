import os

import spacy


def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    cleaned_tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit)]

    cleaned_text = ' '.join(cleaned_tokens).lower()

    return cleaned_text


def preprocess_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)

                with open(file_path, "r") as file:
                    text = file.read()

                cleaned_text = preprocess_text(text)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w") as output_file:
                    output_file.write(cleaned_text)
