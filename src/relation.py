# -*- coding: utf-8 -*-
"""
Relation extractor
"""
from typing import Union, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fine_tuning_rebel.run_rebel import extract_triples

def preprocess_entities(entities: List[str]):
    """ Only keep label, not URI """
    res = []
    for elt in entities:
        parts = elt.strip().split(' - ')
        if len(parts) == 2:
            res.append(parts[0])
    return res

class RelationExtractor:
    """ Extracting relations from text """
    def __init__(self, model_tokenizer: str = "Babelscape/rebel-large",
                 model: str = "Babelscape/rebel-large", local_m: bool = 0):
        """ local_m: whether the model is locally stored or not """
        self.params = {
            "model_tokenizer": model_tokenizer,
            "model": model,
            "local_m": local_m
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)

        if not local_m:  # Downloading from huggingface
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        else:
            self.model = torch.load(model)

        self.gen_kwargs = {"max_length": 256, "length_penalty": 0,
                           "num_beams": 3, "num_return_sequences": 3,}

    def tokenize(self, text: str):
        """ Text > tensor """
        return self.tokenizer(text, max_length=256, padding=True,
                              truncation=True, return_tensors='pt')

    def predict(self, input_m):
        """ Text > predict > human-readable """
        output = self.model.generate(
            input_m["input_ids"].to(self.model.device),
            attention_mask=input_m["attention_mask"].to(self.model.device),
            **self.gen_kwargs,)

        decoded_preds = self.tokenizer.batch_decode(output, skip_special_tokens=False)
        return decoded_preds
    
    @staticmethod
    def extract_triples(x):
        res = extract_triples(x)
        return [(elt['head'], elt['type'], elt['tail']) for elt in res]

    def __call__(self, text: str, entities: Union[List[str], None],
                 preprocess_ent: Union[bool, None] = 1):
        if entities and preprocess_ent is None:
            raise ValueError("If you give `entities` as parameters, you need to " + \
                "specify if they should be preprocessed or not")
        if entities and preprocess_ent:
            entities = preprocess_entities(entities=entities)

        input_m = self.tokenize(text=text)
        output_m = self.predict(input_m=input_m)

        if not entities:
            res = [y for x in output_m for y in self.extract_triples(x)]
        else:
            res = []
            #TO CHECK (triples can appear multiple times)
            for entity in entities:
                cands = [x for x in output_m if entity in x]
                res += [y for x in cands for y in self.extract_triples(x)]
        return res
