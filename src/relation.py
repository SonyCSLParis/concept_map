# -*- coding: utf-8 -*-
"""
Relation extractor
"""
import spacy
from typing import Union, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fine_tuning_rebel.run_rebel import extract_triples


class RelationExtractor:
    """ Extracting relations from text """
    def __init__(self, spacy_model: str, options: List["str"] = ["rebel"],
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None, local_rm: Union[bool, None] = None):
        """ local_m: whether the model is locally stored or not """
        self.options_p = ["rebel"]
        self.options_to_f = {
            "rebel": self.get_rebel_rel
        }
        self.check_params(options=options, rebel_t=rebel_tokenizer,
                          rebel_m=rebel_model, local_rm=local_rm)
        self.params = {
            "options": options,
            "rebel": {
                "tokenizer": rebel_tokenizer,
                "model": rebel_model,
                "local": local_rm
            }
        }
        self.options = options

        if "rebel" in options:
            self.rebel = {
                "tokenizer": AutoTokenizer.from_pretrained(rebel_tokenizer),
                "model": self.get_rmodel(model=rebel_model, local_rm=local_rm),
                "gen_kwargs": {"max_length": 256, "length_penalty": 0,
                               "num_beams": 3, "num_return_sequences": 3,}
            }
        else:
            self.rebel = None
        
        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def get_rmodel(model: str, local_rm: bool):
        """ Load rebel (fine-tuned or not) model """
        if not local_rm:  # Downloading from huggingface
            return AutoModelForSeq2SeqLM.from_pretrained(model)
        return torch.load(model)

    def check_params(self, options, rebel_t, rebel_m, local_rm):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "rebel" in options:
            if any(not isinstance(x, y) for (x, y) in \
                [(rebel_t, str), (rebel_m, str), (local_rm, bool)]):
                raise ValueError("To extract relations with REBEL, you need to specify: " + \
                    "`rebel_tokenizer` as string, `rebel_model` as string, `local_rm` as bool")

    def tokenize(self, text: str):
        """ Text > tensor """
        return self.rebel['tokenizer'](
            text, max_length=256, padding=True,
            truncation=True, return_tensors='pt')

    def predict(self, input_m):
        """ Text > predict > human-readable """
        output = self.rebel['model'].generate(
            input_m["input_ids"].to(self.rebel['model'].device),
            attention_mask=input_m["attention_mask"].to(self.rebel['model'].device),
            **self.rebel['gen_kwargs'],)

        decoded_preds = self.rebel['tokenizer'].batch_decode(output, skip_special_tokens=False)
        return decoded_preds

    def get_rebel_rel(self, sentences: List[str], entities: Union[List[str], None]):
        """ Extracting relations with rebel """
        input_m = self.tokenize(text=sentences)
        output_m = self.predict(input_m=input_m)

        if not entities:
            res = [y for x in output_m for y in self.post_process_rebel(x)]
        else:
            res = []
            #TO CHECK (triples can appear multiple times)
            for entity in entities:
                cands = [x for x in output_m if entity in x]
                res += [y for x in cands for y in self.post_process_rebel(x)]
        return res

    @staticmethod
    def post_process_rebel(x):
        """ Clean rebel output"""
        res = extract_triples(x)
        return [(elt['head'], elt['type'], elt['tail']) for elt in res]

    def __call__(self, sentences: List[str], entities: Union[List[str], None] = None):
        """ Extract relations for one string text """
        res = {}
        for option in self.options:
            res[option] = self.options_to_f[option](sentences=sentences, entities=entities)
        return res


if __name__ == '__main__':
    REL_EXTRACTOR = RelationExtractor(
        options=["rebel"], rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./src/triples_from_text/finetuned_rebel.pth", local_rm=True,
        spacy_model="en_core_web_lg")
    TEXT = """
    The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
    7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer
    Hale.
    """
    ENTITIES = {'dbpedia_spotlight': [
        ('http://dbpedia.org/resource/7_World_Trade_Center', '7 World Trade Center'),
        ('http://dbpedia.org/resource/Benchmarking', 'benchmark'),
        ('http://dbpedia.org/resource/Safety', 'safety'),
        ('http://dbpedia.org/resource/7_World_Trade_Center', '7 WTC'),
        ("http://dbpedia.org/resource/Moody's_Investors_Service", 'Moody'),
        ('http://dbpedia.org/resource/New_York_City', 'New York'),
        ('http://dbpedia.org/resource/Joe_Mansueto', 'Mansueto Ventures'),
        ('http://dbpedia.org/resource/MSCI', 'MSCI'),
        ('http://dbpedia.org/resource/Elisha_Cook_Jr.', 'Wilmer'),
        ('http://dbpedia.org/resource/Hale,_Greater_Manchester', 'Hale')]}
    ENTITIES = [x[1] for x in ENTITIES["dbpedia_spotlight"]]

    print("## WITHOUT ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT)
    print(RES)
    print("==========")
    print("## WITH ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT, entities=ENTITIES)
    print(RES)
