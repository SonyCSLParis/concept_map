# -*- coding: utf-8 -*-
"""
Full pipeline
"""
import sys
from typing import Union, List
import spacy
import time
from tqdm import tqdm
from loguru import logger
from preprocess import PreProcessor
from nltk.corpus import wordnet as wn
from entity import EntityExtractor
from preprocess import PreProcessor
from relation import RelationExtractor
from settings import API_KEY_GPT
from summary import TextSummarizer
from importance_ranking import *

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
                 summary_how: Union[str, None] = None,
                 summary_method: Union[str, None] = None,
                 api_key_gpt: Union[str, None] = None,
                 engine: Union[str, None] = None,
                 temperature: Union[str, None] = None,
                 summary_percentage: Union[str, None] = None,
                 options_ranker : Union[List[str], None] = None,
                 num_sentences: Union[int, None] = None):

        # Summary options: 
        # - `single`: summarising each text one by one
        # - `all`: summarising all texts
        self.summary_p = ["single", "all"]
        self.check_params(
            preprocess=preprocess, spacy_model=spacy_model,
            summary_method=summary_method, summary_how=summary_how
        )

        self.params = {
            "preprocess": {"preprocess": preprocess, "spacy_model": spacy_model,},
            "entity": {"options_ent": options_ent, "confidence": confidence, "db_spotlight_api": db_spotlight_api},
            "relation": {
                "model_tokenizer": rebel_tokenizer, "model": rebel_model,
                "local_rm": local_rm}, "importance_ranker": {"options_ranker": options_ranker},
            "summary": {"method": summary_method, "engine": engine, "temperature": temperature, "summary_percentage": summary_percentage, "num_sentences": num_sentences}
        }

        self.preprocess = PreProcessor(model=spacy_model) if preprocess else None
        self.entity = EntityExtractor(options=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api) \
            if options_ent else None
        self.relation = RelationExtractor(
            options=options_rel, rebel_tokenizer=rebel_tokenizer,
            rebel_model=rebel_model, local_rm=local_rm, spacy_model=spacy_model)
        self.nlp = spacy.load(spacy_model)
        self.summary_how = summary_how
        self.summarizer = TextSummarizer(
            method=summary_method, api_key_gpt=api_key_gpt, engine=engine, temperature=temperature, summary_percentage=summary_percentage, num_sentences=num_sentences
        ) if summary_method else None
        self.importance_ranker = ImportanceRanker(options=options_ranker) \
            if options_ranker else None

    def check_params(self, preprocess, spacy_model, summary_method, summary_how):
        """ Check consistency of params """
        if preprocess and (not spacy_model):
            raise ValueError("For preprocessing, you need to enter `spacy_model`")
        if summary_how and summary_how not in self.summary_p:
            raise ValueError(f"For summarisation, `summary_how` should be in {self.summary_p}")

    def is_list_of_lists(self,lst):
        return isinstance(lst, list) and all(isinstance(elem, list) for elem in lst)

    def __call__(self, input_content: Union[str, List[str]], verbose: bool = False):

        if isinstance(input_content, str):
            input_content = [input_content]
        start_time = time.time()

        # Retrieving sentences for each element in input_content
        docs = [self.nlp(text) for text in input_content]
        docs = [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in docs]

        # Preprocessing
        if verbose:
            logger.info("Preprocessing")
        if self.preprocess:
            sentences = [[self.preprocess(x) for x in sentences] for sentences in docs]
        else:
            sentences = docs
        preprocessing_time = time.time() - start_time

        # importance ranking
        if verbose:
            logger.info("Importance Ranking")

        if self.importance_ranker:
            if not isinstance(sentences, list):
                raise ValueError("Input must be a list of sentences")
            else :
                print(sentences)
                flattened_list = [sentence for sublist in sentences for sentence in sublist]
                print(flattened_list)
                ranking_generation_start_time = time.time()
                if "page_rank" in self.params["importance_ranker"]["options_ranker"]:
                    ranking = ImportanceRanker.compute_page_rank(self,flattened_list)
                if "text_rank" in self.params["importance_ranker"]["options_ranker"]:
                    ranking = ImportanceRanker.compute_text_rank(self,flattened_list)
                if "tfidf" in self.params["importance_ranker"]["options_ranker"]:
                    ranking = ImportanceRanker.compute_page_rank(self,flattened_list)
                if "word2vec" in self.params["importance_ranker"]["options_ranker"]:
                    ranker.train_word2vec_model(sentences)
                    ranking = ImportanceRanker.word_embedding_similarity(self,flattened_list)
                logger.info(f"Ranking : {ranking}")
                print(ranking)
                ranking_extraction_time = time.time() - ranking_generation_start_time

        else:
            ranking = None
            ranking_extraction_time = 0

        # Summary generation
        if verbose:
            logger.info("Summary generation")

        if self.summarizer:
            summary_generation_start_time = time.time()
            if self.summary_how == "single":  # summarising each document one by one
                texts = ["\n".join(elt) for elt in sentences]
                summary = []
                for text in tqdm(texts):
                    summary.append(self.summarizer(text=text))
                sentences_input = [self.nlp(text) for text in summary]
                sentences_input = list(set([sent.text.strip() for x in sentences_input for sent in x.sents if sent.text.strip()]))
            else:  # self.summary_how == "all" -> summarising all documents in one go
                test = "\n".join(["\n".join(elt) for elt in sentences])
                print(f'SUMMARY INPUT: {test}')
                summary = self.summarizer("\n".join(["\n".join(elt) for elt in sentences]))
                sentences_input = self.nlp(summary)
                sentences_input = list(set([sent.text.strip() for sent in sentences_input.sents if sent.text.strip()])) 

            summary_generation_time = time.time() - summary_generation_start_time
            # logger.info(f"Summary found is :{summary}")
        else:
            sentences_input = [x for y in sentences for x in y]
            summary = None

        if verbose:
            logger.info("Entity extraction")

        sentences_input = [x for x in sentences_input if len(x.split(" ")) > 10]

        # Entity extraction
        if self.entity:
            entities_start_time = time.time()
            entities = self.entity(text="\n".join(sentences_input))
            if "dbpedia_spotlight" in self.params["entity"]["options_ent"]:
                entities["dbpedia_spotlight"] = [x[1] for x in entities["dbpedia_spotlight"]]
            entities = list(set(x for _, v in entities.items() for x in v))

            logger.info(f"Entities extracted : {entities}")
            entities_extraction_time = time.time() - entities_start_time

        else:
            entities = None
            entities_extraction_time = 0

        # Relation Extraction
        if verbose:
            logger.info("Relation extraction")

        # total_time = time.time() - start_time
        relation_extraction_start_time = time.time()
        for sent in sentences_input:
            print(sent)
        res = self.relation(sentences=sentences_input, entities=entities)
        relation_extraction_time = time.time() - relation_extraction_start_time

        total_time = time.time() - start_time

        # ADD POST PROCESSING? (TBD)
        logger.info(f"Total execution time: {total_time:.4f}s")
        logger.info(f"Preprocessing time: {preprocessing_time:.4f}s")
        logger.info(f"Ranking extraction time: {ranking_extraction_time:.4f}s")
        logger.info(f"Summary generation time: {summary_generation_time:.4f}s")
        logger.info(f"Entity extraction time: {entities_extraction_time:.4f}s")
        logger.info(f"Relation extraction time: {relation_extraction_time:.4f}s")

        text_to_save = "\n".join(["\n".join(x) for x in sentences])
        return [x for _, val in res.items() for x in val], {"text": "\n".join(["\n".join(x) for x in sentences]), "entities": entities,"summary": summary}


if __name__ == '__main__':
    PIPELINE = CMPipeline(
        preprocess=True, spacy_model="en_core_web_lg",
        # options_ent=["wordnet", "dbpedia_spotlight", "spacy"],
        options_ent=["dbpedia_spotlight"],
        options_ranker=["page_rank","text_rank","tfidf","word2vec"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
        options_rel=["rebel","dependency"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="Babelscape/rebel-large", local_rm=False,
        summary_how="all", summary_method="chat-gpt",
        api_key_gpt=API_KEY_GPT, engine="davinci-002",
        summary_percentage=80, temperature=0.0, num_sentences=3)
    print(PIPELINE.params)
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale.
    """
    RES = PIPELINE(input_content=TEXT, verbose=True)
    print(RES[0])
    print("Ranker:")
    print(RES[1]["ranker"])
