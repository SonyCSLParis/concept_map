# -*- coding: utf-8 -*-
"""
https://huggingface.co/Babelscape/rebel-large
multiprocessing: https://github.com/huggingface/transformers/issues/14919

rebel model to extract triples from text
"""
import pickle
import argparse
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=True,
                    help=".txt file: one line/text")
    ap.add_argument('-o', "--output", required=True,
                    help=".pkl file to save output")
    args_main = vars(ap.parse_args())

    multi_pool = Pool(processes=mp.cpu_count())
    input_list = [x.replace("\n", "") for x in open(
        args_main["input"], "r", encoding="utf-8").readlines()]
    predictions = multi_pool.map(extract_triplets, input_list)
    multi_pool.close()
    multi_pool.join()

    data = {i: {'text': input_list[i], 'triplets': predictions[i]} for i in range(len(input_list))}
    with open(args_main["output"], "wb") as openfile:
        pickle.dump(data, openfile)
