# -*- coding: utf-8 -*-
"""
Running spacy pipeline on data with DBpedia Spotlight

Before running this script: install dbpedia spotlight locally and start the server
To run the server locally: java -Xmx8G -jar /path/to/dbpedia-spotlight/rest-1.1-jar-with-dependencies.jar \
     en http://localhost:2222/rest
"""
import pickle
import argparse
import multiprocessing as mp
import spacy
from spacy.tokens import DocBin, Doc

NLP = spacy.load("en_core_web_sm")
NLP.add_pipe("dbpedia_spotlight",
             config={
                'confidence': 0.5,
                'dbpedia_rest_endpoint': 'http://localhost:2222/rest'})

def get_entities(doc: Doc) -> list[str]:
    """ Getting dbpedia entities from spacy doc """
    res = [ent for ent in doc.ents if ent._.dbpedia_raw_result]
    return list(set(ent._.dbpedia_raw_result["@URI"] for ent in res))


def extract_dbpedia_spotlight_entities(data_file: str, nlp: spacy.lang.en.English, save_file: str):
    """
    - data_file: .txt file containing the data, one sentence/line
    """
    lines = [x.replace("\n", "") for x in open(data_file, "r", encoding="utf-8").readlines()]
    # Spacy + DBpedia Spotlight
    docs = nlp.pipe(lines, n_process=mp.cpu_count())
    docs = list(docs)

    for doc in docs:
        print(type(doc))
        print(get_entities(doc))

    # Converting to bytes to store as .pkl file
    doc_bin = DocBin(store_user_data=True)
    for doc in docs:
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()

    with open(save_file, "wb") as openfile:
        pickle.dump(bytes_data, openfile)


if __name__ == '__main__':
    # python src/extract_dbpedia_spotlight_entities.py -i ./src/data/sample-spotlight.txt \
    #  -o ./src/data/sample-spotlight.pkl
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=True,
                    help=".txt file: one line/text")
    ap.add_argument('-o', "--output", required=True,
                    help=".pkl file to save output")
    args_main = vars(ap.parse_args())

    extract_dbpedia_spotlight_entities(
        data_file=args_main["input"], nlp=NLP,
        save_file=args_main["output"])
