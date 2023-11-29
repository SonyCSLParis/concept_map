"""
Running spacy pipeline on data with DBpedia Spotlight
"""
import os
import requests
from spacy.tokens import Doc
from settings import *

def get_entities(doc: Doc) -> list[str]:
    """ Getting dbpedia entities from spacy doc """
    res = [ent for ent in doc.ents if ent._.dbpedia_raw_result]
    return list(set(ent._.dbpedia_raw_result["@URI"] for ent in res))

def extract_dbpedia_spotlight_entities(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    text = file.read()
                    payload = {
                        'text': text,
                        'confidence': 0.35
                    }
                    headers = {'Accept': 'application/json'}
                    response = requests.post('https://api.dbpedia-spotlight.org/en/annotate', data=payload, headers=headers)
                    if response.status_code == 200:
                        api_result = response.json()
                        entities = [(resource["@URI"], resource["@surfaceForm"]) for resource in api_result["Resources"]]
                        print(entities)  # Print entities found by the API
                        output_root = root.replace(input_folder, output_folder)
                        os.makedirs(output_root, exist_ok=True)
                        output_file_name = file_name.replace(".txt", "-entities.txt")  # Change the extension to .txt
                        output_file_path = os.path.join(output_root, output_file_name)
                        with open(output_file_path, "w") as openfile:
                            for entity, surface_form in entities:
                                openfile.write(f"{surface_form} - {entity}\n")  # Write original word and entity to a new line in the text file
                    else:
                        print(f"Error: {response.status_code} - Failed to get entities for {file_path}")