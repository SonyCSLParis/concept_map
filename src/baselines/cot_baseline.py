# -*- coding: utf-8 -*-
"""
LLM Baselines for CM building
"""
import os
import json
from typing import List, Union
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
import pandas as pd
from src.data_load import DataLoader
from src.preprocess import PreProcessor
from src.evaluation import EvaluationMetrics
from settings import API_KEY_GPT

CLIENT = OpenAI(api_key=API_KEY_GPT)
MODEL = "gpt-3.5-turbo-0125"

def get_texts_from_folder(folder):
    """ Retrieves all .txt file in folder """
    res = [x for x in os.listdir(folder) if x.endswith(".txt")]
    return res, [open(os.path.join(folder, x), encoding="utf-8").read() for x in res]

def run_gpt(prompt: str, content: Union[str, List[str]], **add_content):
    """ Get answer from GPT from prompt + content """
    if isinstance(content, str):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    else:  # list of text
        messages = \
            [{"role": "system", "content": prompt}] + \
            [{"role": "user", "content": c} for c in content]

    if add_content and add_content.get("entities"):
        messages += [{"role": "user", "content": add_content.get("entities")}]
    completion = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0)
    return completion.choices[0].message.content

def save_to_folder(folder: str, content: Union[List[str], str], names: str):
    """ Save content to folder """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if isinstance(content, list):
        for i, name in enumerate(names):
            f = open(os.path.join(folder, name), 'w', encoding='utf-8')
            f.write(content[i].replace("```csv", "").replace("```", "").strip())
            f.close()
    else:
        f = open(os.path.join(folder, names[0]), 'w', encoding='utf-8')
        f.write(content)
        f.close()

def get_gs_triples(file_path):
    """ Get ground truth triples """
    res = open(file_path, "r", encoding='utf-8').readlines()
    return [x.replace("\n", "").split("\t") for x in res]

class LLMCOTBaseline:
    """ LLM Baseline (using one model of OpenAI) for CM extraction """
    def __init__(self, model: str = MODEL):
        self.model = model
        self.start_prompt = self.get_start_prompt()
        self.preprocess = PreProcessor()

        self.prompt_summary = """
        You need to write a summary of the text that will be sent in the following message.

        The summary is:
        """
        self.prompt_entity = """
        You need to extract all the entities from the text that will be sent in the following message.

        In your answer, you must give the output in a .csv file with the columns `entity` and `surface`. `entity` contains one label, whereas `surface` contains all the surface forms of that entity. The columns are separated by `;`, and the surface forms in the `surface` column by `,`.

        The output is:
        ```csv
        ```
        """
        self.prompt_group_entity = """
        You need to group all the entities from the .csv that will be sent in the following messages.

        In your answer, you must give the output in a .csv file with the columns `entity` and `surface`. `entity` contains one label, whereas `surface` contains all the surface forms of that entity. The columns are separated by `;`, and the surface forms in the `surface` column by `,`.

        The output is:
        ```csv
        ```
        """
        self.prompt_relation = """
        You need to extract all the relations from the text that will be sent in the following message. Each relation is in the form of a triple (subject, predicate, object), where subject and object are entities that you idenfitied in the previous step. `subject` and `object` should be from the list of entities you will be sent.

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """
        self.prompt_group_relation = """
        You need to group all the relations from the .csv that will be sent in the following messages.

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """
        self.prompt_ir = """
        You first need to process the triples by removing redundant triples, i.e. triples that have the same meaning. Then, you need to select the most important triples from the set of triples that will be sent. 

        In your answer, you must give the output in a .csv file with the columns with the columns `subject`,  `predicate` and `object`. The columns are separated by `;`.

        The output is:
        ```csv
        ```
        """

    def get_start_prompt(self):
        """ [deprecated] old prompt for whole pipeline """
        return """
        You need to extract concept maps from a set of documents. 

        Concept Map Extraction can be framed as a summarisation task where the output is a graph, i.e. a set of triples with entities as nodes.

        This task will be decomposed into the following subtasks:
        1- Summarization. Summarize the content from the set of documents. The output must be a .csv file with the columns `id` and `summary`. `id` is the text number and `summary` its summary.

        From step 2 on, you work on the summarised text from step 1.
        2- Entity extraction. Extract entities from the text. This include named entities and noun phrase entities. If entities refer to the same entity, they should have only one representant. The output must be a .csv file with the columns `entity` and `surface`. `entity` contains one label, whereas `surface` contains all the surface forms of that entity.
        3- Relation extraction. Extract relation from the text. Each relation is in the form of a triple (subject, predicate, object), where subject and object are entities. The output must be a .csv file with the columns `subject`,  `predicate` and `object`.
        4- Importance ranking. Select the most important triples that will constitute the final concept map. The output must be a .csv file with the columns `subject`,  `predicate` and `object`.

        The following messages will each contain one text from the set of documents. When the set of documents are done, I will send one last message containing: `[END]`

        In your answer, you must give one output per step.
        """

    def __call__(self, folder: str, save_folder: Union[str, None] = None):
        logger.info(f"Generating concept maps from texts in folder {folder}")
        names, texts = get_texts_from_folder(folder=folder)

        logger.info("Preprocessing texts")
        texts = [self.preprocess(x) for x in texts]

        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "preprocess"),
                           content=texts, names=names)


        logger.info("Generating summaries")
        summaries = []
        for text in tqdm(texts):
            summaries.append(run_gpt(
                prompt=self.prompt_summary, content=text))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "summary"),
                           content=summaries, names=names)

        logger.info("Extracting entities")
        entities = []
        for text in tqdm(summaries):
            entities.append(run_gpt(
                prompt=self.prompt_entity, content=text))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "entity"),
                           content=entities, names=names)

        logger.info("Grouping entities")
        grouped_entities = run_gpt(
            prompt=self.prompt_group_entity, content=entities)
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "grouped_entity"),
                           content=grouped_entities, names=["grouped_entities.csv"])

        logger.info("Extracting relations")
        relations = []
        info = {"entities": grouped_entities}
        for text in tqdm(summaries):
            relations.append(run_gpt(
                prompt=self.prompt_relation, content=text, **info))
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "relation"),
                           content=relations, names=names)

        logger.info("Grouping relations")
        grouped_relations = run_gpt(
            prompt=self.prompt_group_relation, content=relations)
        if save_folder:
            save_to_folder(folder=os.path.join(save_folder, "grouped_relation"),
                           content=grouped_relations, names=["grouped_relations.csv"])

        logger.info("Extracting most important relations")
        output = run_gpt(
            prompt=self.prompt_ir, content=grouped_relations)
        if save_folder:
            save_to_folder(folder=save_folder,
                           content=output, names=["output.csv"])

        logger.success("Finished process")


class LLMCOTExperimentRun:
    """ Running experiment on a dataset """
    def __init__(self, data_path: str, type_d: str = "multi", one_cm: bool = False):
        """ Init dataset """
        self.dataset = DataLoader(
            path=data_path, type_d=type_d, one_cm=one_cm
        )
        self.model = LLMCOTBaseline()
        self.evaluation_metrics = EvaluationMetrics()

    def run_evaluation(self, gs_path, rel_path):
        """ Running evaluation metrics """
        gs_triples = get_gs_triples(file_path=gs_path)
        system_triples = pd.read_csv(rel_path, sep=";")
        system_triples = [list(x) for x in system_triples.values]

        return self.evaluation_metrics(
            triples=system_triples, gold_triples=gs_triples)

    def __call__(self, save_folder):
        """ Running on all files of the dataset """
        metrics = {}
        for info in self.dataset.files:
            folder = os.path.join(self.dataset.path, info["folder"])
            curr_sf = os.path.join(save_folder, info["folder"])
            if not os.path.exists(curr_sf):
                os.makedirs(curr_sf)
                self.model(folder=folder, save_folder=curr_sf)
                curr_metrics = self.run_evaluation(
                    gs_path=info['gs'], rel_path=os.path.join(curr_sf, 'output.csv'))

                metrics[info["folder"]] = curr_metrics
                with open(os.path.join(curr_sf, "metrics.json"),
                            "w", encoding="utf-8") as openfile:
                    json.dump(metrics, openfile, indent=4)




if __name__ == '__main__':
    FOLDER = "src/data/Corpora_Falke/Wiki/test/102"
    SAVE_FOLDER = "test"
    BASELINE = LLMCOTBaseline()
    BASELINE(folder=FOLDER, save_folder=SAVE_FOLDER)
