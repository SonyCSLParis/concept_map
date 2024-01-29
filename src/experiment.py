# -*- coding: utf-8 -*-
"""
Running experiments
"""
import json
import os
import time
from datetime import datetime
from typing import Union, List

from datetime import datetime
from tqdm import tqdm
import spacy
from loguru import logger
from loguru import logger
from tqdm import tqdm

from data_load import DataLoader
from evaluation import EvaluationMetrics
from pipeline import CMPipeline
from settings import *


def get_save_folder():
    """ Save folder """
    date = str(datetime.now())
    return f"{date[:10]}-{date[11:19]}"


def create_folders(folder_path: str):
    """ Create folders to save intermediate steps """
    os.makedirs(folder_path)
    for name in ["preprocess", "entity", "relation"]:
        os.makedirs(os.path.join(folder_path, name))


def save_data(preprocess, entities, relations, save_folder, name):
    """ Save intermediate steps data """
    with open(os.path.join(
            save_folder, "relation", f"{name}.txt"), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join([", ".join([x for x in rel]) for rel in relations]))

    with open(os.path.join(
            save_folder, "preprocess", f"{name}.txt"), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(preprocess))

    with open(os.path.join(
            save_folder, "entity", f"{name}.json"), "w", encoding="utf-8") as openfile:
        json.dump({"entities": entities}, openfile, indent=4)


def get_gs_triples(file_path):
    res = open(file_path, "r").readlines()
    return [x.replace("\n", "").split("\t") for x in res]


class ExperimentRun:
    """ Running a full experiment """

    def __init__(self,
                 # Param for data
                 folder_path: str, type_data: str, one_cm: bool,
                 # Param for pipeline
                 options_rel: List[str],
                 preprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 db_spotlight_api: Union[str, None] = 'https://api.dbpedia-spotlight.org/en/annotate',
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None):
        self.data = DataLoader(path=folder_path, type_d=type_data, one_cm=one_cm)

        print("Data Loader done!")

        self.pipeline = CMPipeline(
            options_rel=options_rel, preprocess=preprocess, spacy_model=spacy_model,
            options_ent=options_ent, confidence=confidence,
            db_spotlight_api=db_spotlight_api,
            rebel_tokenizer=rebel_tokenizer, rebel_model=rebel_model, local_rm=local_rm
        )
        self.evaluation_metrics = EvaluationMetrics()

        self.params = self.pipeline.params

        data = self.data.params
        data.update({"files": self.data.files})
        self.params.update({"data": data})

    def __call__(self, save_folder: str):
        """ A folder will be created in save_folder to store the results of experiments """
        metrics = {}
        logs = {}
        logger.info(f"Running experiments for the following parameters: {self.params}")

        # Save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_folder = os.path.join(save_folder, get_save_folder())
        os.makedirs(save_folder)

        # Save params
        with open(os.path.join(save_folder, "params.json"), "w", encoding="utf-8") as openfile:
            json.dump(self.params, openfile, indent=4)

        # Run pipeline for each folder and each file
        nb_folder = len(self.data.files)
        for i_folder, folder_info in enumerate(tqdm(self.data.files)):
            folder = folder_info['folder']
            all_relations = []
            logs[folder] = {}

            curr_folder = os.path.join(save_folder, folder_info["folder"])
            folder_t_log = f"[Folder {folder}][{i_folder+1}/{nb_folder} ({round(100*(i_folder+1)/nb_folder)}%)]"
            logger.info(folder_t_log)
            create_folders(folder_path=curr_folder)

            nb_file = len(folder_info["text"])
            for i_file, (name, path) in enumerate(tqdm(folder_info["text"])):
                file_t_log = f"[File {name}][{i_file+1}/{nb_file} ({round(100*(i_file+1)/nb_file)}%)]"
                logger.info(file_t_log + folder_t_log)
                start_ = datetime.now()
                logs[folder][name] = {"start": str(start_)}
                with open(path, "r", encoding="utf-8") as openfile:
                    text = openfile.read()
                    # doc = self.nlp(text)
                    # sentences = [sent.text.strip() for sent in doc.sents]
                    preprocess, entities, relations = [], [], []

                    # nb_sent = len(sentences)
                    # for i_sent, sent in enumerate(sentences):
                        # sent_t_log = f"[Sentence {i_sent+1}][{i_sent+1}/{nb_sent} ({round(100*(i_sent+1)/nb_sent)}%)]"
                        # logger.info(sent_t_log + file_t_log + folder_t_log)
                    c_relations, c_info = self.pipeline(text=text, verbose=True)
                    preprocess.append(c_info["text"])
                    entities += c_info["entities"]
                    relations += c_relations
                    save_data(relations=relations, preprocess=preprocess, entities=entities,
                              save_folder=curr_folder, name=name)
                    all_relations += relations
                print("Pipeline & Preprocessing done")

                #  Run evaluation
                gs_triples = get_gs_triples(file_path=folder_info["gs"])
                all_relations = list(set(all_relations))
                curr_metrics = self.evaluation_metrics(
                    triples=all_relations, gold_triples=gs_triples)
                metrics[folder] = curr_metrics
                print("Evaluation done, saving metrics..")

                # Save metrics and logs
                with open(os.path.join(save_folder, "metrics.json"),
                          "w", encoding="utf-8") as openfile:
                    json.dump(metrics, openfile, indent=4)

                end_ = datetime.now()
                logs[folder][name].update({"end": str(end_), "total": str(end_-start_)})
                with open(os.path.join(save_folder, "logs.json"),
                          "w", encoding="utf-8") as openfile:
                    json.dump(logs, openfile, indent=4)


if __name__ == '__main__':
    EXPERIMENTR = ExperimentRun(
        folder_path=WIKI_TRAIN + "101",
        # folder_path=WIKI_TRAIN,
        type_data="multi", one_cm=True,
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["wordnet"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model=REBEL_DIR, local_rm=True)
    print(EXPERIMENTR.params)
    EXPERIMENTR(save_folder="experiments")
