# -*- coding: utf-8 -*-
"""
https://huggingface.co/Babelscape/rebel-large
multiprocessing: https://github.com/huggingface/transformers/issues/14919

rebel model to extract triples from text
"""
import pickle
import argparse
import os
import multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
from transformers import pipeline

set_start_method("spawn", force=True)
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',
                             tokenizer='Babelscape/rebel-large')


def extract_triplets(text):
    """ Extract triplets from text using REBEL model """
    text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(text, return_tensors=True, return_text=False) \
             [0]["generated_token_ids"]])[0]
    triplets, relation, subject, object_ = [], '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),
                                 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),
                                 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),
                         'tail': object_.strip()})
    return triplets


def process_file(input_path, output_path):
    multi_pool = Pool(processes=mp.cpu_count())

    input_list = [x.replace("\n", "") for x in open(input_path, "r", encoding="utf-8").readlines()]
    predictions = multi_pool.map(extract_triplets, input_list)

    data = {i: {'text': input_list[i], 'triplets': predictions[i]} for i in range(len(input_list))}
    with open(output_path, "wb") as openfile:
        pickle.dump(data, openfile)


def preprocess_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                input_path = os.path.join(root, file_name)
                output_path = input_path.replace(input_folder, output_folder).replace(".txt", ".pkl")

                process_file(input_path, output_path)