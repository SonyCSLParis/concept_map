# -*- coding: utf-8 -*-
"""
Full pipeline
"""
from typing import Union, List
import spacy
from preprocess import PreProcessor
from entity import EntityExtractor
from relation import RelationExtractor

class CMPipeline:
    """ class for the whole pipeline """
    def __init__(self, options_rel: List[str],
                 preprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None):

        self.check_params(
            # Preprocessing
            preprocess=preprocess, spacy_model=spacy_model,
            )

        self.params = {
            "preprocess": {"preprocess": preprocess, "spacy_model": spacy_model,},
            "entity": {"options_ent": options_ent, "confidence": confidence,},
            "relation": {
                "model_tokenizer": rebel_tokenizer, "model": rebel_model,
                "local_rm": local_rm}
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(options=options_ent, confidence=confidence) \
            if options_ent else None
        self.relation = RelationExtractor(
            options=options_rel, rebel_tokenizer=rebel_tokenizer,
            rebel_model=rebel_model, local_rm=local_rm)

        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def check_params(preprocess, spacy_model):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")

    def __call__(self, text: str):
        if self.preprocess:
            text = self.preprocess(text)
        if self.entity:
            entities = self.entity(text=text)
            entities = [x[1] for x in entities["dbpedia_spotlight"]]
        else:
            entities = None

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        res = []
        for sent in sentences:
            res.append(self.relation(text=sent, entities=entities))

        # ADD POST PROCESSING? (TBD)
        return [x for elt in res for option, val in elt.items() for x in val], {"text": text, "entities": entities}


if __name__ == '__main__':
    PIPELINE = CMPipeline(
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["dbpedia_spotlight"],
        confidence=0.35,
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="Babelscape/rebel-large", local_rm=False)
    print(PIPELINE.params)
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer
    Hale.
    """
    RES = PIPELINE(text=TEXT)
    print(RES[0])
