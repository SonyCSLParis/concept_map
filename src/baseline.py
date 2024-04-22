# -*- coding: utf-8 -*-
"""
LLM Baselines for CM building
"""
from typing import List
from settings import API_KEY_GPT
from openai import OpenAI

CLIENT = OpenAI(api_key=API_KEY_GPT)
MODEL = "davinci-002"

def run_gpt(prompt):
    """ Get answer from GPT """
    completion = CLIENT.chat.completions.create(
        model=MODEL,
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature=0)
    return completion.choices[0].message.content

class LLMBaseline:
    """ LLM Baseline (using one model of OpenAI) for CM extraction """
    def __init__(self, model: str = MODEL):
        self.model = model
        self.start_prompt = self.get_start_prompt()

    def get_start_prompt(self):
        return """
        You need to extract concept maps from a set of documents. 

        Concept Map Extraction can be framed as a summarisation task where the output is a graph, i.e. a set of triples with entities as nodes.

        This task will be decomposed into the following subtasks:
        1- Summarization. Summarize the content from the set of documents.

        From step 2 on, you work on the summarised text from step 1.
        2- Entity extraction. Extract entities from the text. This include named entities and noun phrase entities. If entities refer to the same entity, they should have only one representant.
        3- Relation extraction. Extract relation from the text. Each relation is in the form of a triple (subject, predicate, object), where subject and object are entities.
        4- Importance ranking. Select the most important triples that will constitute the final concept map.

        The following messages will each contain one text from the set of douments.

        In your answer, you must give one output per step.
        """

    def __call__(self, docs: List[str]):
        messages = \
            [{"role": "user", "content": self.start_prompt}] + \
            [{"role": "user", "content": text} for text in docs]
        completion = CLIENT.chat.completions.create(
            model=self.model, messages=messages, temperature=0)
        return completion.choices

