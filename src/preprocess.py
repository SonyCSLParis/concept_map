"""
Running spacy pipeline for preprocessing : remove stopwords,punctuation, double spaces and citations
"""
import os
import spacy
import regex as re
from tqdm import tqdm
from loguru import logger

class PreProcessor:
    """ Main class for preprocessing """
    def __init__(self, model: str = "en_core_web_lg"):
        """ Init main params"""
        self.nlp = spacy.load(model)

    def __call__(self, text: str):
        """ Preprocessing one string text """
        doc = self.nlp(text)
        cleaned_tokens = [token.text for token in doc if not (token.is_stop or token.is_punct)]

        cleaned_text = ' '.join(cleaned_tokens).lower()
        cleaned_text_1 = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\[\d+\]\[\d+\]\[\d+\]', '', cleaned_text_1)

        return cleaned_text

    def get_sentences(self, file_path: str):
        """ Self explanatory """
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()
            doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def main_folder(self, input_folder: str, output_folder: str):
        """ Instead of one text, running for one folder"""
        for root, _, files in os.walk(input_folder):
            logger.info(f"[Preprocessing] Root: {root}")
            for file_name in tqdm([x for x in files if x.endswith(".txt")]):
                file_path = os.path.join(root, file_name)
                sentences = self.get_sentences(file_path=file_path)

                cleaned_text = [self(sent) for sent in sentences]
                my_string = '\n'.join(cleaned_text)

                output_root = root.replace(input_folder, output_folder)
                os.makedirs(output_root, exist_ok=True)

                output_file_name = file_name.replace(".txt", "-preprocessed.txt")
                output_file_path = os.path.join(output_root, output_file_name)

                with open(output_file_path, "w", encoding='utf-8') as output_file:
                    output_file.write(my_string)
