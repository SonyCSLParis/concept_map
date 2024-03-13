# -*- coding: utf-8 -*-
"""
Relation extractor
"""
import spacy
from typing import Union, List
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from openai import OpenAI
from openie import StanfordOpenIE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.fine_tune_rebel.run_rebel import extract_triples
from src.settings import *

client = OpenAI(api_key=API_KEY_GPT)


class RelationExtractor:
    """ Extracting relations from text """

    def __init__(self, spacy_model: str, options: List[str] = ["rebel", "dependency", "chat-gpt"],
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None, local_rm: Union[bool, None] = None):
        """ local_m: whether the model is locally stored or not """
        self.options_to_f = {
            "rebel": self.get_rebel_rel,
            "dependency": self.get_dependencymodel,
            "chat-gpt": self.get_chat_gpt,
            "corenlp": self.get_corenlp_rel,

        }
        self.options_p = list(self.options_to_f.keys())
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
                               "num_beams": 3, "num_return_sequences": 3, }
            }
        else:
            self.rebel = None

        self.nlp = spacy.load(spacy_model)
    
    def get_corenlp_rel(self, sentences: List[str], entities: Union[List[str], None]):
        """ Extracting relations with rebel """
        triples = []
        with StanfordOpenIE() as client:
            triples = client.annotate("\n".join(sentences))
        triples = [(x["subject"], x["relation"], x["object"]) for x in triples]
        if isinstance(entities, List):
            triples = [(a, b, c) for a, b, c in triples if any((x.lower() in a.lower()) or (x.lower() in b.lower()) for x in entities)]
        return triples

    @staticmethod
    def get_rmodel(model: str, local_rm: bool):
        """ Load rebel (fine-tuned or not) model """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        if not local_rm:  # Downloading from huggingface
            model = AutoModelForSeq2SeqLM.from_pretrained(model)
        else:
            model = torch.load(model)
        model.to(device)
        return model

    @staticmethod
    def get_dependencymodel(sentences: str, entities: Union[List[str], None]):
        triplets = []
        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass", "agent", "csubjpass",
                                  "csubj", "compound"] and token.head.pos_ in ["VERB", "AUX", "ROOT", "VB", "VBD", "VBG", "VBN", "VBZ"]:
                    subject = token.text
                    verb = token.head.text
                    subject_pos = token.pos_
                    # print(subject_pos)
                    obj = None
                    if any(entity in subject for entity in entities):
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"] :
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)

                                if subject_pos in ["NOUN", "PROPN"] and obj_pos in ["NOUN", "PROPN"]:
                                    triplets.append((subject, verb, obj))
                    else:
                        for child in token.head.children:
                            if child.dep_ in ["dobj", "pobj", "acomp", "attr", "agent", "ccomp", "pcomp",
                                              "xcomp", "csubjpass", "dative", "nmod", "oprd", "obj", "obl"] :
                                obj = child.text
                                obj_pos = child.pos_
                                # print(obj_pos)
                                if subject_pos in ["NOUN", "PROPN","ADP"] and obj_pos in ["NOUN", "PROPN"] and any(entity in obj for entity in entities):
                                    triplets.append((subject, verb, obj))
        return triplets

    def check_params(self, options, rebel_t, rebel_m, local_rm):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_p for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_p}")

        if "rebel" in options:
            if any(not isinstance(x, y) for (x, y) in \
                   [(rebel_t, str), (rebel_m, str), (local_rm, bool)]):
                raise ValueError("To extract relations with REBEL, you need to specify: " + \
                                 "`rebel_tokenizer` as string, `rebel_model` as string, `local_rm` as bool")

    # def tokenize(self, text: str):
    #     """ Text > tensor """
    #     return self.rebel['tokenizer'](
    #         text, max_length=256, padding=True,
    #         truncation=True, return_tensors='pt')

    def predict(self, input_m):
        """ Text > predict > human-readable """
        for key in ["input_ids", "attention_mask"]:
            if len(input_m[key].shape) == 1:
                #  Reshaping, has a single sample
                input_m[key] = input_m[key].reshape(1, -1)

        output = self.rebel['model'].generate(
            input_m["input_ids"].to(self.rebel['model'].device),
            attention_mask=input_m["attention_mask"].to(self.rebel['model'].device),
            **self.rebel['gen_kwargs'], )

        decoded_preds = self.rebel['tokenizer'].batch_decode(output, skip_special_tokens=False)
        return decoded_preds

    def get_dataloader(self, sent_l: List[str], batch_size: int = 16):
        if not sent_l:
            return None
        sent_l = [x for x in sent_l if x]
        sent_l = [x for x in sent_l if len(x.split()) <= 256]
        
        dataset = Dataset.from_dict({"text": sent_l})
        dataset = dataset.map(lambda examples: self.rebel['tokenizer'](examples["text"], max_length=256, padding=True, truncation=True, return_tensors='pt'), batched=True)
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])
        return DataLoader(dataset, batch_size=batch_size)
    
    def get_chat_gpt(self, sentences: List[str], entities: Union[List[str], None]):
        res = []
        for sent in sentences:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "user",
                    "content": f"Extract triples from this sentence:\n{sent}\n\n. One triple per line, in the format a|b|c"}
                ],
                temperature=0)

            try:
                output = completion.choices[0].message.content.split("\n")
                output = [tuple(x.split('|')) for x in output]
                output = [x for x in output if len(x) == 3]
                res += output
            except Exception as e:
                print(e)
                raise ValueError("Something went wrong with the summary")
        
        if entities:
            res = [(a, b, c) for a, b, c in res if any((x.lower() in a) or (x.lower() in b) for x in entities)]
        
        return res

    def get_rebel_rel(self, sentences: List[str], entities: Union[List[str], None]):
        """ Extracting relations with rebel """

        # input_m = self.tokenize(text=sentences)
        dataloader = self.get_dataloader(sent_l=sentences)
        if not dataloader:  # empty sentences
            return []
        output_m = []
        for batch in dataloader:
            try:
                output_m += self.predict(input_m=batch)
            except:
                pass

        unique_triples_set = set()  # Set to store unique triples
        res = []

        if not entities:
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if triple not in unique_triples_set:
                        res.append(triple)
                        unique_triples_set.add(triple)
        else:
            for entity in entities:
                entity_strings = [item for tuple_item in entities for item in tuple_item]
                cands = [x for x in output_m if any(entity_string in x for entity_string in entity_strings)]

                for x in cands:
                    for triple in self.post_process_rebel(x):
                        if triple not in unique_triples_set:
                            res.append(triple)
                            unique_triples_set.add(triple)

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
            curr_res = self.options_to_f[option](sentences=sentences, entities=entities)
            curr_res = [x for x in curr_res if x[0].lower() != x[2].lower()]
            # print(curr_res)
            res[option] = list(set(curr_res))
        return res


if __name__ == '__main__':
    REL_EXTRACTOR = RelationExtractor(
        options=["corenlp"], spacy_model="en_core_web_lg")
    SENTENCES = [
        "The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.",
        "7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale."
    ]
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
    RES = REL_EXTRACTOR(sentences=SENTENCES)
    print(RES)
    print("==========")
    print("## WITH ENTITIES")
    RES = REL_EXTRACTOR(sentences=SENTENCES, entities=ENTITIES)
    print(RES)
