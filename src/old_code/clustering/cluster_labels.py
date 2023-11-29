# -*- coding: utf-8 -*-
import os
import argparse
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')


def cluster_texts(texts, output_folder):
    """ Main """
    topic_model = BERTopic(representation_model=MODEL)
    topic_model.fit_transform(texts)

    topic_model.get_topic_info().to_csv(os.path.join(output_folder, "texts_topic_info.csv"))
    topic_model.get_document_info(texts) \
        .to_csv(os.path.join(output_folder, "texts_document_info.csv"))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', "--labels", required=True,
                    help=".txt file containing labels")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    args_main = vars(ap.parse_args())

    TEXTS = list(set([x.replace("\n", "") for x in \
        open(args_main["labels"], encoding="utf-8").readlines()]))
    cluster_texts(texts=TEXTS, output_folder=args_main["output"])
