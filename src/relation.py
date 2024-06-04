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
from src.settings import API_KEY_GPT, nlp
from src.entity import EntityExtractor

client = OpenAI(api_key=API_KEY_GPT)


class RelationExtractor:
    """ Extracting relations from text """

    def __init__(self, spacy_model: str, options: List[str] = ["rebel", "dependency", "chat-gpt", "corenlp"],
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


    def get_corenlp_rel(self, sentences: List[str], entities: Union[List[Union[str, spacy.tokens.Span, spacy.tokens.Token]], None]):
        """ Extracting relations with corenlp"""
        with StanfordOpenIE() as client:
            triples = client.annotate("\n".join(sentences))
        triples = [(x["subject"], x["relation"], x["object"]) for x in triples]
        if entities is not None:
            # Convert all entities to their string representations
            entity_strings = [self._to_string(entity) for entity in entities]
            triples = [(a, b, c) for a, b, c in triples if
                       any((x in a) or (x in b) for x in entity_strings)]
        return triples

    def _to_string(self, entity):
        """ Helper method to convert entities to string representations """
        if isinstance(entity, spacy.tokens.Span) or isinstance(entity, spacy.tokens.Token):
            return entity.text
        return str(entity)

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
        SENTENCES = [sent.strip() for sent in TEXT.split('\n') if sent.strip()]  # Split text into sentences
        for sentence in SENTENCES:
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
    
    def get_chat_gpt(self, sentences: List[str], entities: Union[List[Union[str, spacy.tokens.Span, spacy.tokens.Token]], None]):
        res = []
        for sent in sentences:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Extract triples from this sentence:\n{sent}\n\n. One triple per line, in the format a|b|c"}
                ],
                temperature=0
            )

            try:
                output = completion.choices[0].message['content'].strip().split("\n")
                output = [tuple(x.split('|')) for x in output]
                output = [x for x in output if len(x) == 3]
                res += output
            except Exception as e:
                print(e)
                raise ValueError("Something went wrong with the summary")

        if entities:
            entity_strings = [self._to_string(entity) for entity in entities]
            res = [(a, b, c) for a, b, c in res if any((x.lower() in a.lower()) or (x.lower() in b.lower()) or (x.lower() in c.lower()) for x in entity_strings)]

        return res

    def _to_string(self, entity):
        """ Helper method to convert entities to string representations """
        if isinstance(entity, spacy.tokens.Span) or isinstance(entity, spacy.tokens.Token):
            return entity.text
        return str(entity)

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
        unique_triples_set_2 = set()  # Set to store unique triples

        res = []

        if entities:
            entity_strings = [str(entity).lower() for entity in entities if
                              isinstance(entity, str)]  # Convert entities to lowercase strings
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if any(entity.lower() in triple_part.lower() for entity in entity_strings for triple_part in
                           triple):
                        if triple not in unique_triples_set:
                            res.append(triple)
                            unique_triples_set.add(triple)
        else :
            for x in output_m:
                for triple in self.post_process_rebel(x):
                    if triple not in unique_triples_set_2:
                        res.append(triple)
                        unique_triples_set_2.add(triple)

        return res

    @staticmethod
    def post_process_rebel(x):
        """ Clean rebel output"""
        res = extract_triples(x)
        return [(elt['head'], elt['type'], elt['tail']) for elt in res]

    def __call__(self, text: Union[str, List[str]], entities: Union[List[str], None] = None):
        """ Extract relations for input text """
        if isinstance(text, list):
            sentences = text
        elif isinstance(text, str):
            sentences = [sent.strip() for sent in text.split('\n') if sent.strip()]
        else:
            raise ValueError("Input text must be either a string or a list of strings.")

        res = {}
        for option in self.options:
            curr_res = self.options_to_f[option](sentences=sentences, entities=entities)
            curr_res = [x for x in curr_res if x[0].lower() != x[2].lower()]
            res[option] = list(set(curr_res))
        return res


if __name__ == '__main__':
    REL_EXTRACTOR = RelationExtractor(
        options=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./fine_tune_rebel/finetuned_rebel.pth",
        local_rm=True,
        spacy_model="en_core_web_lg",
    )
    TEXT = """
            The 52-story, 1.7-million-square-foot 7 World Trade Center is a benchmark of innovative design, safety, and sustainability.
            7 WTC has drawn a diverse roster of tenants, including Moody's Corporation, New York Academy of Sciences, Mansueto Ventures, MSCI, and Wilmer Hale.
            The quick brown fox jumps over the lazy dog.
            This is a test sentence without any entities.
            A long list of random words: apple, banana, orange, pineapple, watermelon, kiwi, mango, strawberry, blueberry, raspberry.
            """
    # SENTENCES = [sent.strip() for sent in TEXT.split('\n') if sent.strip()]  # Split text into sentences
    # ENTITIES = {'dbpedia_spotlight': [
    #     ('http://dbpedia.org/resource/7_World_Trade_Center', '7 World Trade Center'),
    #     ('http://dbpedia.org/resource/Benchmarking', 'benchmark'),
    #     ('http://dbpedia.org/resource/Safety', 'safety'),
    #     ('http://dbpedia.org/resource/7_World_Trade_Center', '7 WTC'),
    #     # ("http://dbpedia.org/resource/Moody's_Investors_Service", 'Moody'),
    #     # ('http://dbpedia.org/resource/New_York_City', 'New York'),
    #     ('http://dbpedia.org/resource/Joe_Mansueto', 'Mansueto Ventures'),
    #     ('http://dbpedia.org/resource/MSCI', 'MSCI'),
    #     ('http://dbpedia.org/resource/Elisha_Cook_Jr.', 'Wilmer'),
    #     ('http://dbpedia.org/resource/Hale,_Greater_Manchester', 'Hale')]}
    # ENTITIES = [x[1] for x in ENTITIES["dbpedia_spotlight"]]

    ENTITY_EXTRACTOR = EntityExtractor(options=["dbpedia_spotlight"], confidence=0.35,
                                       db_spotlight_api="http://localhost:2222/rest/annotate",
                                       threshold=1)
    ENTITIES = ENTITY_EXTRACTOR(text=TEXT)
    ENTITIES = [x[1] for x in ENTITIES["dbpedia_spotlight"]]
    print(ENTITIES)


    print("## WITHOUT ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT)
    print(RES)
    print("==========")
    print("## WITH ENTITIES")
    RES = REL_EXTRACTOR(text=TEXT, entities=ENTITIES)
    print(RES)
