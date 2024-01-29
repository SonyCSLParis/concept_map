# -*- coding: utf-8 -*-
"""
Full pipeline
"""
from typing import Union, List
import spacy
from loguru import logger
from preprocess import PreProcessor
from entity import EntityExtractor
from relation import RelationExtractor
from nltk.corpus import wordnet as wn
from settings import *
class CMPipeline:
    """ class for the whole pipeline """
    def __init__(self, options_rel: List[str],
                 preprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 db_spotlight_api: Union[str, None] = 'https://api.dbpedia-spotlight.org/en/annotate',
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None):

        self.check_params(
            # Preprocessing
            preprocess=preprocess, spacy_model=spacy_model,
            )

        self.params = {
            "preprocess": {"preprocess": preprocess, "spacy_model": spacy_model,},
            "entity": {"options_ent": options_ent, "confidence": confidence, "db_spotlight_api": db_spotlight_api},
            "relation": {
                "model_tokenizer": rebel_tokenizer, "model": rebel_model,
                "local_rm": local_rm}
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(options=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api) \
            if options_ent else None
        self.relation = RelationExtractor(
            options=options_rel, rebel_tokenizer=rebel_tokenizer,
            rebel_model=rebel_model, local_rm=local_rm, spacy_model=spacy_model)
        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def check_params(preprocess, spacy_model):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")

    def __call__(self, text: str, verbose: bool = False):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        if verbose:
            logger.info("Preprocessing")
        if self.preprocess:
            sentences = [self.preprocess(x) for x in sentences]

        if verbose:
            logger.info("Entity extraction")
        if self.entity:
            entities = self.entity(text="\n".join(sentences))
            unique_tuples_set = set()  # Set to keep track of unique tuples

            if 'wordnet' in self.params["entity"]["options_ent"]:
                found_wordnet_entities_set = set()
                for pos, synset in entities.get('wordnet', []):
                    words = [lemma.name() for lemma in wn.synset(synset).lemmas()]
                    found_wordnet_entities_set.update(
                        {token.text for token in nlp(text) if token.text.lower() in words})

                found_wordnet_entities = list(found_wordnet_entities_set)
                entities['wordnet'] = found_wordnet_entities
                unique_tuples_set.add(tuple(found_wordnet_entities))  # Add the tuple to the set

            if 'dbpedia_spotlight' in self.params["entity"]["options_ent"]:
                dbpedia_entities = [x[1] for x in entities["dbpedia_spotlight"]]
                unique_tuples_set.add(tuple(dbpedia_entities))  # Add the tuple to the set

            entities = list(unique_tuples_set)  # Convert set back to list

        else:
            entities = None

        if verbose:
            logger.info("Relation extraction")

        res = self.relation(sentences=sentences, entities=entities)

        # ADD POST PROCESSING? (TBD)
        return [x for _, val in res.items() for x in val], {"text": "\n".join(sentences), "entities": entities}


if __name__ == '__main__':
    PIPELINE = CMPipeline(
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["wordnet", "dbpedia_spotlight"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
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
