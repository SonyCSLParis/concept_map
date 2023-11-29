# -*- coding: utf-8 -*-
"""
Replacing triplets content with DBpedia IRI (if applicable)
"""
import os

def read_iris(filename):
    """ Retrieve surface form - IRI mapping from DBpedia """
    lines = open(file=filename, encoding='utf-8').readlines()
    lines = [x.split("\t") for x in lines]
    return [(x[0], x[1]) for x in lines]

def read_triplets(filename):
    """ Read triplets """
    lines = open(file=filename, encoding='utf-8').readlines()
    return [x.split(",") for x in lines]

def replace(text, mapping):
    """ replace each surface form in mapping by its iri """
    for (old, new) in mapping:
        text = text.replace(old, new)
    return text

def merge_text_iri(input_folder: str, mapping_folder: str, output_folder: str):
    """ Replace text in triplets by their IRI (if needed)"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)

            output_file_name = file_name.replace(".txt", "-with-iri.txt")
            output_file_path = os.path.join(output_folder, output_file_name)

            mapping_file_name = file_name.replace("_importance_ranking-triplets.txt", ".txt")
            mapping = read_iris(filename=os.path.join(mapping_folder, mapping_file_name))

            triplets_old = read_triplets(filename=file_path)
            triplets_new = []

            for elt in triplets_old:
                triplets_new.append(f"{replace(elt[0], mapping)},{elt[1]},{replace(elt[2], mapping)}")

            output_file = open(output_file_path, "w", encoding='utf-8')
            output_file.write("".join(triplets_new))
            output_file.close()

            print(f"DBpedia IRIs integrated in triplets for {file_name}. Output saved to {output_file_path}")
