import os
import regex as re
from nltk.tokenize import word_tokenize
from settings import *

def get_token_from_index(doc, index):
    try:
        token = doc[index]
        regex = r"\(\d+\)"
        subst = ""
        str_token = token.pretty_representation
        result = re.sub(regex, subst, str_token, 0, re.MULTILINE)
        return result
    except IndexError:
        return None

def substitute_tokens(list_of_lists, doc, token):
    if isinstance(doc, str):
        try:
            words = word_tokenize(doc)
            for index in list_of_lists:
                for idx in index:
                    words[idx] = token

            modified_text = ' '.join(words)
            return modified_text

        except IndexError:
            print("Index out of range error occurred.")
            return doc

    else:
        try:
            text = doc.text
            words = word_tokenize(text)
            for index in list_of_lists:
                for idx in index:
                    words[idx] = token

            modified_text = ' '.join(words)
            return modified_text

        except IndexError:
            print("Index out of range error occurred.")
            return doc

def perform_coreference_resolution(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nlp.add_pipe('coreferee')

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)

                with open(file_path, 'r') as file:
                    text = file.read()

                doc = nlp(text)
                print(doc._.coref_chains.print())
                x = doc._.coref_chains
                if len(x) > 0:
                    for chain in x:
                        index_first_mention = chain.most_specific_mention_index

                        token = get_token_from_index(chain, index_first_mention)
                        if token:
                            print(f"Token at index {index_first_mention}: {token}")
                            list_chains = chain.mentions
                            doc = substitute_tokens(list_chains, doc, token)

                # Create output folder structure if it doesn't exist
                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_path = os.path.join(output_root, filename)

                if isinstance(doc, str):
                    with open(output_file_path, 'w') as file:
                        file.write(doc)
                else:
                    with open(output_file_path, 'w') as file:
                        file.write(doc.text)

                print(f"Coreference resolution completed for {filename}. Output saved to {output_file_path}")
