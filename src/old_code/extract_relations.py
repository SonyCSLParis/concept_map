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
    match = re.match(r'(.+?) - ', text)
    if match:
        return match.group(1)
    return None

def extract_triplets(text, entity1, entity2):
    # Extract entity names from provided strings
    entity_name_1 = extract_entity(entity1)
    entity_name_2 = extract_entity(entity2)

    # Initialize variables for triplets
    triplets = []
    relation, subject, object_ = '', '', ''

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
                # if (subject == entity_name_1) or (object_ == entity_name_2):
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation, subject, object_ = '', '', ''
        # Check if the token indicates the start of a subject
        elif token == "<subj>":
            current = 's'
            # If a relation is already present, append the triplet to the list
            if relation != '':
                # Check if entity_name_1 or entity_name_2 is the subject or object
                # if (subject == entity_name_1) or (object_ == entity_name_2):
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation, subject, object_ = '', '', ''
        # Check if the token indicates the start of an object
        elif token == "<obj>":
            current = 'o'
            relation, subject, object_ = '', '', ''
        else:
            # Update the triplet components based on the current context and entity names
            if current == 't' :
                object_ += ' ' + token
            elif current == 's':
                subject += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    # Check if there is a complete triplet at the end of the text
    if subject != '' and relation != '' and object_ != '':
        # Check if entity_name_1 or entity_name_2 is the subject or object
        # if (subject == entity_name_1) or (object_ == entity_name_2):
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

if __name__ == '__main__':
    extract_triplets("['<s><triplet> supreme court <subj> father of our constitution <obj> named after <triplet> father of our constitution <subj> supreme court <obj> court</s>', '<s><triplet> declaration of independence <subj> federalist papers <obj> followed by <triplet> federalist papers <subj> declaration of independence <obj> follows</s>', '<s><triplet> federalist <subj> no 33 <obj> has part <triplet> no 33 <subj> federalist <obj> part of</s>', '<s><triplet> tariff act of 1828 <subj> 1828 <obj> point in time</s>', '<s><triplet> federalist paper no 33 <subj> federalist <obj> part of <triplet> federalist <subj> federalist paper no 33 <obj> has part <subj> federalist no 45 <obj> has part <triplet> federalist no 45 <subj> federalist <obj> part of</s>', '<s><triplet> enumerated powers listed in the constitution <subj> our constitution <obj> part of <triplet> our constitution <subj> enumerated powers listed in the constitution <obj> has part</s>', '<s><triplet> patents <subj> intellectual labors <obj> subclass of <triplet> copyright <subj> intellectual labors <obj> facet of</s>', '<s><triplet> 13th amendment <subj> federal constitution <obj> part of <triplet> federal constitution <subj> 13th amendment <obj> has part <subj> 4th 5th 6th 7th and 8th amendments <obj> has part <triplet> 4th 5th 6th 7th and 8th amendments <subj> federal constitution <obj> part of</s>', '<s><triplet> federalist papers <subj> federalist no 28 <obj> has part <triplet> federalist no 28 <subj> federalist papers <obj> part of</s>', '<s><triplet> 10th amendment <subj> reserved power <obj> has part <triplet> reserved power <subj> 10th amendment <obj> part of</s>', '<s><triplet> federalist no 45 <subj> madison <obj> author</s>', '<s><triplet> federal government <subj> congress <obj> has part <subj> president <obj> has part <triplet> congress <subj> federal government <obj> part of <subj> federal government <obj> part of <triplet> federal government <subj> congress <obj> part of <subj> congress <obj> part of <triplet> federal government <subj> congress <obj> part of <subj> congress <obj> part of</s>', '<s><triplet> 1798 <subj> 1798 <obj> point in time</s>', '<s><triplet> lawyer <subj> legal point <obj> field of this occupation <triplet> legal point <subj> lawyer <obj> practiced by</s>', '<s><triplet> randy barnett <subj> ivory tower <obj> residence</s>', '<s><triplet> supreme court <subj> federal government <obj> part of <subj> federal government <obj> part of <subj> federal government <obj> part of <triplet> federal government <subj> supreme court <obj> has part <triplet> federal government <subj> supreme court <obj> has part <triplet> federal government <subj> supreme court <obj> has part</s>', '<s><triplet> unconstitutional <subj> usurps powers not delegated to it in the constitution <obj> subclass of</s>', '<s><triplet> executive order <subj> president <obj> author</s>', '<s><triplet> virginia resolutions 1799 1800 <subj> 1800 <obj> point in time</s>', '<s><triplet> president <subj> federal government <obj> part of <triplet> federal government <subj> president <obj> has part</s>', '<s><triplet> federal government <subj> united states <obj> country <triplet> united states <subj> federal government <obj> legislative body</s>', '<s><triplet> federal government <subj> states <obj> has part <triplet> states <subj> federal government <obj> part of</s>', '<s><triplet> legislative <subj> executive branch <obj> opposite of <triplet> executive branch <subj> legislative <obj> opposite of</s>', '<s><triplet> executive <subj> federal government <obj> part of <triplet> legislature <subj> federal government <obj> part of <triplet> federal government <subj> executive <obj> has part <subj> legislature <obj> has part</s>', '<s><triplet> virginia resolutions 1799 1800 <subj> 1800 <obj> point in time</s>', '<s><triplet> tariff act of 1828 <subj> 1828 <obj> point in time <subj> 1828 <obj> point in time</s>', '<s><triplet> tariff act of 1828 <subj> 1828 <obj> point in time <subj> 1828 <obj> point in time</s>', '<s><triplet> federal government <subj> united states states <obj> applies to jurisdiction <triplet> united states states <subj> federal government <obj> authority</s>', '<s><triplet> tariff act of 1828 <subj> 1828 <obj> point in time</s>', '<s><triplet> declaration of independence <subj> 2nd para <obj> has part <triplet> 2nd para <subj> declaration of independence <obj> part of</s>', '<s><triplet> founding documents <subj> federalist papers <obj> has part <triplet> federalist papers <subj> founding documents <obj> part of</s>', '<s><triplet> federalist <subj> no 32 <obj> has part <triplet> no 32 <subj> federalist <obj> part of</s>', '<s><triplet> 1799 <subj> 1800 <obj> followed by <triplet> 1800 <subj> 1799 <obj> follows</s>', '<s><triplet> federalist no 39 <subj> federal constitution <obj> main subject</s>', '<s><triplet> kentucky resolutions of 1798 <subj> 1798 <obj> point in time</s>', '<s><triplet> teddy roosevelt <subj> 1912 <obj> candidacy in election <triplet> 1912 <subj> teddy roosevelt <obj> candidate</s>', '<s><triplet> Nullification Act of 1834 <subj> 1834 <obj> point in time</s>']", 'supreme court - http://dbpedia.org/resource/Supreme_Court_of_the_United_Kingdom','federal constitution - http://dbpedia.org/resource/Constitution_of_the_United_States')