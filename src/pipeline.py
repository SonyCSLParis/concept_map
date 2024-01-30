# -*- coding: utf-8 -*-
"""
Full pipeline
"""
from typing import Union, List
import spacy
import time
from loguru import logger
from preprocess import PreProcessor
from nltk.corpus import wordnet as wn

from entity import EntityExtractor
from preprocess import PreProcessor
from relation import RelationExtractor
from settings import *
from summary import *

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
                 local_rm: Union[bool, None] = None,
                 summary_parameters: Union[str, None] = None):
        self.check_params(
            preprocess=preprocess, spacy_model=spacy_model,
        )

        self.params = {
            "preprocess": {"preprocess": preprocess, "spacy_model": spacy_model,},
            "entity": {"options_ent": options_ent, "confidence": confidence, "db_spotlight_api": db_spotlight_api},
            "relation": {
                "model_tokenizer": rebel_tokenizer, "model": rebel_model,
                "local_rm": local_rm},
            "summary_parameters": summary_parameters
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(options=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api) \
            if options_ent else None
        self.relation = RelationExtractor(
            options=options_rel, rebel_tokenizer=rebel_tokenizer,
            rebel_model=rebel_model, local_rm=local_rm, spacy_model=spacy_model)
        self.nlp = spacy.load(spacy_model)
        self.summarizer = TextSummarizer(api_key_gpt=API_KEY_GPT, engine="davinci-002")  # Replace with your actual API key

    @staticmethod
    def check_params(preprocess, spacy_model):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")

    def generate_summary(self, text: str, method: str = "lex-rank") -> str:
        """
        Generate a summary of the given text using the specified method.

        Parameters:
            text (str): The input text to summarize.
            method (str): The summarization method to use ("lex-rank" or "chat-gpt").

        Returns:
            str: The generated summary.
        """
        if method == "lex-rank":
            return self.summarizer.generate_lex_rank_summary(text)
        elif method == "chat-gpt":
            return self.summarizer.generate_summary_with_gpt(text, summary_percentage=80, temperature=0.7)
        else:
            raise ValueError(f"Invalid summary method: {method}")

    def __call__(self, text: str, verbose: bool = False, summary_method: str = "lex-rank"):
        start_time = time.time()
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        if verbose:
            logger.info("Preprocessing")
        if self.preprocess:
            sentences = [self.preprocess(x) for x in sentences]

        preprocessing_time = time.time() - start_time

        if verbose:
            logger.info("Summary generation")
        summary_generation_start_time = time.time()
        summary = self.generate_summary(sentences, method=summary_method)
        summary_generation_time = time.time() - summary_generation_start_time

        if verbose:
            logger.info("Entity extraction")

        if self.entity:
            entities_start_time = time.time()
            entities = self.entity(text=summary)
            unique_tuples_set = set()

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

            entities = list(unique_tuples_set)

            entities_extraction_time = time.time() - entities_start_time

        else:
            entities = None
            entities_extraction_time = 0

        if verbose:
            logger.info("Relation extraction")

        relation_extraction_start_time = time.time()
        res = self.relation(sentences=sentences, entities=entities)
        relation_extraction_time = time.time() - relation_extraction_start_time

        total_time = time.time() - start_time

        # ADD POST PROCESSING? (TBD)
        logger.info(f"Total execution time: {total_time:.4f}s")
        logger.info(f"Preprocessing time: {preprocessing_time:.4f}s")
        logger.info(f"Summary generation time: {summary_generation_time:.4f}s")
        logger.info(f"Entity extraction time: {entities_extraction_time:.4f}s")
        logger.info(f"Relation extraction time: {relation_extraction_time:.4f}s")

        return [x for _, val in res.items() for x in val], {"text": "\n".join(sentences), "entities": entities,"summary": summary}


if __name__ == '__main__':
    API_KEY_GPT = ""
    PIPELINE = CMPipeline(
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["wordnet", "dbpedia_spotlight"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="Babelscape/rebel-large", local_rm=False,
        summary_parameters=["lex-rank", "chat-gpt"]
    )
    print(PIPELINE.params)
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale.
    """
    RES = PIPELINE(text=TEXT, verbose=True)
    print(RES[0])
    print("Summary:")
    print(RES[1]["summary"])
