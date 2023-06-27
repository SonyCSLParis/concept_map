import os
from settings import *
from extract_triplets import extract_triplets

def extract_triplets_from_summaries(folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            output_file_name = file_name.replace(".txt", "-triplets.txt")
            output_file_path = os.path.join(output_folder_path, output_file_name)

            with open(file_path, "r") as file:
                text = file.read()

                doc = nlp(text)
                triplets = []

                for sentence in doc.sents:
                    for token in sentence:
                        if token.dep_ in ["nsubj", "nsubjpass", "agent", "csubjpass",
                                          "csubj"] and token.head.pos_ in ["VERB", "AUX", "ROOT"]:
                            subject = token.text
                            verb = token.head.text
                            for child in token.head.children:
                                if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                                  "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"]:
                                    obj = child.text
                                    triplets.append((subject, verb, obj))
                                elif child.dep_ in ["prep"]:
                                    obj = child.text
                                    prep_s = [childx.text for childx in child.children if childx.dep_ == 'appos']
                                    prep = ''.join(prep_s)
                                    triplets.append((subject, verb, obj + " " + prep))
                        # Check if the token is part of a prepositional phrase
                        elif token.dep_ == 'prep':
                            pobj = [child.text for child in token.children if child.dep_ == 'pobj']
                            right_edge = [child.right_edge.text for child in token.children if
                                          child.dep_ == 'pobj' and child.right_edge.dep_ == "appos"]
                            left_edge = [child.right_edge.text for child in token.children if
                                         child.dep_ == 'pobj' and child.left_edge.dep_ == "appos"]
                            if pobj and right_edge:
                                object_ = ' '.join([token.text] + pobj + right_edge)
                                triplets.append((subject, verb, object_))
                            if pobj and left_edge:
                                object_ = ' '.join([token.text] + pobj + right_edge)
                                triplets.append((subject, verb, object_))

                with open(output_file_path, "w") as output_file:
                    for triplet in triplets:
                        output_file.write(f"{triplet[0]}, {triplet[1]}, {triplet[2]}\n")

def extract_triplets_from_text(text):
    doc = nlp(text)
    triplets = []

    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ in ["nsubj", "nsubjpass", "agent", "csubjpass", "csubj"] and token.head.pos_ in ["VERB", "AUX", "ROOT"]:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ["dobj","pobj","acomp","attr","agent","ccomp","pcomp","xcomp","csubjpass","dative","nmod","oprd","obj","obl"]:
                        obj = child.text
                        triplets.append((subject, verb, obj ))
                    elif child.dep_ in ["prep"] :
                        obj = child.text
                        prep_s = [childx.text for childx in child.children if childx.dep_ == 'appos']
                        prep = ''.join(prep_s)
                        triplets.append((subject, verb, obj + " " + prep))
        # Check if the token is part of a prepositional phrase
            elif token.dep_ == 'prep':
                pobj = [child.text for child in token.children if child.dep_ == 'pobj']
                right_edge = [child.right_edge.text for child in token.children if child.dep_ == 'pobj' and child.right_edge.dep_=="appos" ]
                left_edge = [child.right_edge.text for child in token.children if child.dep_ == 'pobj' and child.left_edge.dep_=="appos" ]
                if pobj and right_edge:
                    object_ = ' '.join([token.text] + pobj + right_edge)
                    triplets.append((subject, verb, object_))
                if pobj and left_edge:
                    object_ = ' '.join([token.text] + pobj + right_edge)
                    triplets.append((subject, verb, object_))
    return triplets

def extract_triplets_from_single_txt(parent_folder_path, output_folder_path):
    for file_name in os.listdir(parent_folder_path):
        if file_name.endswith("-summary.txt"):
            file_path = os.path.join(parent_folder_path, file_name)
            output_file_name = file_name.replace(".txt", "-triplets.txt")
            output_file_path = os.path.join(output_folder_path, output_file_name)

            with open(file_path, "r") as file:
                text = file.read()

            triplets = extract_triplets_from_text(text)
            triplets_rebel = extract_triplets(text)  # REBEL

            with open(output_file_path, "w") as output_file:
                for triplet in triplets:
                    output_file.write(f"{triplet[0]}, {triplet[1]}, {triplet[2]}\n")
                
                for info in triplets_rebel:
                    output_file.write(f"{info['head']}, {info['type']}, {info['tail']}\n")

            
            