"""
Running spacy pipeline on data with DBpedia Spotlight
"""
import os
import requests
from typing import Union
from loguru import logger


class EntityExtractor:
    """ Extracting entities from text """
    def __init__(self, confidence: float = 0.35):
        """ Init main params """
        self.confidence = confidence
        self.headers = {'Accept': 'application/json'}
        self.dbpedia_spotlight_api = 'https://api.dbpedia-spotlight.org/en/annotate'
        self.timeout = 3600

    def get_payload(self, text: str):
        """ Payload for requests """
        return {'text': text, 'confidence': self.confidence}

    def __call__(self, text: str, file_path: Union[str, None] = None):
        """ Extract entities for one string text """
        response = requests.post(
            self.dbpedia_spotlight_api, data=self.get_payload(text=text),
            headers=self.headers, timeout=self.timeout)
        if response.status_code == 200:
            return [(resource["@URI"], resource["@surfaceForm"]) \
                for resource in response.json()["Resources"]]
        logger.error(f"{response.status_code} - " + \
            f"Failed to get entities for {file_path if file_path else text}")
        return []

    def main_folder(self, input_folder: str, output_folder: str):
        """ Main for one folder """
        for root, _, files in os.walk(input_folder):
            for file_name in [x for x in files if x.endswith(".txt")]:
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    entities = self(text=file.read(), file_path=file_path)
                    logger.info(entities)
                    output_root = root.replace(input_folder, output_folder)
                    os.makedirs(output_root, exist_ok=True)
                    # Change the extension to .txt
                    output_file_name = file_name.replace(".txt", "-entities.txt")  
                    output_file_path = os.path.join(output_root, output_file_name)
                    with open(output_file_path, "w", encoding='utf-8') as openfile:
                        for entity, surface_form in entities:
                            # Write original word and entity to a new line in the text file
                            openfile.write(f"{surface_form} - {entity}\n")
