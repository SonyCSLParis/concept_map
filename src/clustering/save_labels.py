# -*- coding: utf-8 -*-
"""
Using embeddings to cluster topics of dataset
"""
import os
import re
import spacy

def pre_process(text):
    """ Pre-processing: remove parenthesis """
    text = re.sub("\\(.+\\)", " ", text)
    return re.sub("\\[.+\\]", " ", text)

if __name__ == '__main__':
    FOLDER = "src/data/vikidia-en-fr-analysis/en/"
    NLP = spacy.load("en_core_web_sm")

    TEXTS = [open(os.path.join(FOLDER, x), encoding='utf-8') \
        .read().replace("\n", " ") \
            for x in sorted(os.listdir(FOLDER))]
    TEXTS = [pre_process(text.split(".")[0]).strip() for text in TEXTS]

    f = open("src/data/vikidia-en-fr/labels/en.txt", "w+", encoding="utf-8")
    f.write("\n".join(TEXTS))
    f.close()
