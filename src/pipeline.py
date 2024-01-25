# -*- coding: utf-8 -*-
"""
Full pipeline
"""
from typing import Union
import spacy
from preprocess import PreProcessor
from entity import EntityExtractor
from relation import RelationExtractor

class CMPipeline:
    """ class for the whole pipeline """
    def __init__(self, preprocess: bool = 1, spacy_model: Union[str, None] = "en_core_web_lg",
                 entity: bool = 1, confidence: float = 0.35,
                 model_tokenizer: str = "Babelscape/rebel-large", model: str = "Babelscape/rebel-large", local_m: bool = 0):

        self.check_params(
            # Preprocessing
            preprocess=preprocess, spacy_model=spacy_model,
            # Entity extraction
            entity=entity, confidence=confidence,
            # Relation extraction
            )

        self.params = {
            "preprocess": preprocess, "spacy_model": spacy_model,
            "entity": entity, "confidence": confidence,
            "model_tokenizer": model_tokenizer, "model": model,
            "local_m": local_m
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(confidence=confidence) if entity else None
        self.relation = RelationExtractor(model_tokenizer=model_tokenizer, model=model, local_m=local_m)

        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def check_params(preprocess, spacy_model, entity, confidence):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")
        if entity and not isinstance(confidence, float):
            raise ValueError("For entity extraction, you need to enter `confidence`")

    def __call__(self, text: str):
        if self.preprocess:
            text = self.preprocess(text)
        if self.entity:
            entities = self.entity(text=text)
            entities = [x[1] for x in entities]

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        res = []
        for sent in sentences:
            res += self.relation(text=sent, entities=entities, preprocess_ent=0)
        return res


if __name__ == '__main__':
    PIPELINE = CMPipeline()
    print(PIPELINE.params)
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer
    Hale.
    """
    RES = PIPELINE(text=TEXT)
    print(RES)
