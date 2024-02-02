# -*- coding: utf-8 -*-
"""
Running experiments
"""
import json
from typing import List, Union
from datetime import datetime
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
                 folder_path: str, type_data: str, one_cm: bool,
                 options_rel: List[str],
                 preprocess: bool = False,
                 spacy_model: Union[str, None] = None,
                 options_ent: Union[List[str], None] = None,
                 confidence: Union[float, None] = None,
                 db_spotlight_api: Union[str, None] = 'https://api.dbpedia-spotlight.org/en/annotate',
                 rebel_tokenizer: Union[str, None] = None,
                 rebel_model: Union[str, None] = None,
                 local_rm: Union[bool, None] = None,
                 summary_parameters: Union[str, None] = None):  # Add summary_parameters
        self.data = DataLoader(path=folder_path, type_d=type_data, one_cm=one_cm)

        logger.info("Data Loader done!")

        self.pipeline = CMPipeline(
            options_rel=options_rel, preprocess=preprocess, spacy_model=spacy_model,
            options_ent=options_ent, confidence=confidence, db_spotlight_api=db_spotlight_api,
            rebel_tokenizer=rebel_tokenizer, rebel_model=rebel_model, local_rm=local_rm,  # Add comma here
            summary_parameters=summary_parameters  # Pass summary_parameters to CMPipeline
        )
        self.evaluation_metrics = EvaluationMetrics()

        self.params = self.pipeline.params

        data = self.data.params
        data.update({"files": self.data.files})
        self.params.update({"data": data, "summary_parameters": summary_parameters})  # Add summary_parameters to params

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
                    preprocess, entities, relations = [], [], []

                    c_relations, c_info = self.pipeline(text=text, verbose=True)
                    preprocess.append(c_info["text"])

                    # Check if "entities" key is present and not None
                    if "entities" in c_info and c_info["entities"] is not None:
                        entities += c_info["entities"]
                    else :
                        print("entities is None!!!!")

                    relations += c_relations
                    save_data(relations=relations, preprocess=preprocess, entities=entities, save_folder=curr_folder,
                              name=name)
                    all_relations += relations
                logger.info("Pipeline & Preprocessing done")

                #  Run evaluation
                gs_triples = get_gs_triples(file_path=folder_info["gs"])
                all_relations = list(set(all_relations))
                curr_metrics = self.evaluation_metrics(
                    triples=all_relations, gold_triples=gs_triples)
                metrics[folder] = curr_metrics
                logger.info("Evaluation done, saving metrics..")

                # Save metrics and logs
                with open(os.path.join(save_folder, "metrics.json"),
                          "w", encoding="utf-8") as openfile:
                    json.dump(metrics, openfile, indent=4)

                end_ = datetime.now()
                logs[folder][name].update({"end": str(end_), "total": str(end_-start_)})
                with open(os.path.join(save_folder, "logs.json"),
                          "w", encoding="utf-8") as openfile:
                    json.dump(logs, openfile, indent=4)

                logger.info(f"Total execution time: {(end_ - start_).total_seconds():.4f}s")

if __name__ == '__main__':
    EXPERIMENTR = ExperimentRun(
        folder_path="./src/data/Corpora_Falke/Wiki/train",
        # folder_path=WIKI_TRAIN + "101",
        type_data="multi", one_cm=False,
        preprocess=True, spacy_model="en_core_web_lg",
        options_ent=["dbpedia_spotlight"],
        # options_ent=["wordnet", "dbpedia_spotlight","spacy"],
        confidence=0.35,
        db_spotlight_api="http://localhost:2222/rest/annotate",
        options_rel=["rebel"],
        rebel_tokenizer="Babelscape/rebel-large",
        rebel_model="./src/fine_tune_rebel/finetuned_rebel.pth", local_rm=True,
        # rebel_model=REBEL_DIR, local_rm=True,
        summary_parameters="chat-gpt")  # or "lex-rank"
    # print(EXPERIMENTR.params)
    EXPERIMENTR(save_folder="experiments")
