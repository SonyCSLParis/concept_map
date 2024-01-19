"""
https://huggingface.co/Babelscape/rebel-large
multiprocessing: https://github.com/huggingface/transformers/issues/14919

rebel model to extract triples from text
"""
from torch.multiprocessing import set_start_method
from transformers import pipeline

from settings import *
import regex as re
set_start_method("spawn", force=True)
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',
                             tokenizer='Babelscape/rebel-large')

triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

def extract_entity(text):
    match = re.match(r'([^ -]+) - ', text)
    if match:
        return match.group(1)
    return None

def extract_triplets(text, entity1, entity2):
    # Extract entity names from provided strings
    entity_name_1 = extract_entity(entity1)
    entity_name_2 = extract_entity(entity2)

    # Initialize variables for triplets
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''

    # Preprocess the input text
    text = text.strip()

    # Initialize a variable to keep track of the current context ('t', 's', or 'o')
    current = 'x'

    # Iterate through tokens in the text
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        # Check if the token indicates the start of a triplet
        if token == "<triplet>":
            current = 't'
            # If a relation is already present, append the triplet to the list
            if relation != '':
                # Check if entity_name_1 or entity_name_2 is the subject or object
                if subject == entity_name_1 or object_ == entity_name_2:
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        # Check if the token indicates the start of a subject
        elif token == "<subj>":
            current = 's'
            # If a relation is already present, append the triplet to the list
            if relation != '':
                # Check if entity_name_1 or entity_name_2 is the subject or object
                if subject == entity_name_1 or object_ == entity_name_2:
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        # Check if the token indicates the start of an object
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            # Update the triplet components based on the current context and entity names
            if current == 't' and token == entity_name_1:
                subject += ' ' + token
            elif current == 's' and token == entity_name_2:
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    # Check if there is a complete triplet at the end of the text
    if subject != '' and relation != '' and object_ != '':
        # Check if entity_name_1 or entity_name_2 is the subject or object
        if subject == entity_name_1 or object_ == entity_name_2:
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})

    # Print and return the extracted triplets
    print(triplets)
    return triplets


def extract_parent_folder(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    return parent_folder


def preprocess_file(file_path):
    base_name = os.path.basename(file_path)
    prefix = ''.join(filter(str.isdigit, base_name))
    new_file_name = f"M{prefix}-preprocessed.txt"
    new_file_path = file_path.replace(base_name, new_file_name)
    return new_file_path


def extract_relationships(parent_folder_path, text_folder_path, output_folder_path):
    nlp = spacy.load("en_core_web_sm")

    for root, dirs, files in os.walk(parent_folder_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                name_of_file = preprocess_file(file_name)
                parent_folder_name = extract_parent_folder(file_path)
                with open(file_path, "r") as file:
                    entities = file.readlines()
                    relationships = []

                    new_path = os.path.join(text_folder_path, parent_folder_name, name_of_file)
                    with open(new_path, "r") as file:
                        text = file.read()
                        doc = nlp(text)

                        # Iterate through pairs of entities and extract linking verbs
                        for i in range(len(entities)):
                            for j in range(i + 1, len(entities)):
                                entity1 = entities[i].strip()
                                entity2 = entities[j].strip()
                                max_length = 1024
                                text_parts = [text[i:i + max_length] for i in
                                              range(0, len(text), max_length)]
                                extracted_texts = []
                                extracted_triplets = []
                                for i in text_parts:
                                    # generated_text = triplet_extractor(i, max_length=1024, return_tensors=True, return_text=True)

                                    extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(i,
                                        return_tensors=True, return_text=False)[0]["generated_token_ids"]])
                                    extracted_texts.append(extracted_text[0])
                                print(extracted_texts)
                                for i in extracted_texts:
                                    extracted_single_triplets = extract_triplets(i, entity1, entity2)
                                    print(extracted_single_triplets)
                                    extracted_triplets.append(extracted_single_triplets)
                                print(extracted_triplets)
                                # verbs = extract_triplets(text, entity1, entity2)
                                # if verbs:
                                #     relationship = f"{entity1} - {', '.join(verbs)} - {entity2}"
                                #     relationships.append(relationship)

                    # Save relationships to a new text file with the same structure in the output folder
                    output_file_path = os.path.join(output_folder_path, new_path)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, "w") as output_file:
                        output_file.write("\n".join(relationships))
