import json
import os

import pandas as pd

from src.evaluation import EvaluationMetrics


class ConceptMapEvaluator:
    def __init__(self):
        self.evaluation_metrics = EvaluationMetrics()

    def run_evaluation(self, gs_triples, system_triples):
        """ Running evaluation metrics """
        return self.evaluation_metrics(triples=system_triples, gold_triples=gs_triples)

    def evaluate_concept_maps(self, golden_folder_path, system_folder_path):
        """ Evaluate concept maps generated for a folder """
        metrics = {}

        # Extract folder numbers from the paths
        folder_number = os.path.basename(path).split("_")[1].split(".")[0]
        print(folder_number)

        # Get golden triples
        gs_path = os.path.join(golden_folder_path, f"concept_map_{folder_number}.csv")
        gs_triples = pd.read_csv(gs_path, sep=";").values.tolist()

        # Get system triples
        rel_path = os.path.join(system_folder_path, f"concept_map_{folder_number}.csv")
        system_triples = pd.read_csv(rel_path, sep=";").values.tolist()

        # Run evaluation
        metrics[system_folder_number] = self.run_evaluation(gs_triples, system_triples)

        # Save metrics to JSON
        with open(os.path.join(system_folder_path, "metrics.json"), "w", encoding="utf-8") as openfile:
            json.dump(metrics, openfile, indent=4)

if __name__ == "__main__":
    evaluator = ConceptMapEvaluator()
    golden_folder_path = "/Users/martina/Desktop/concept_map/src/baselines/data"
    system_folder_path = "/Users/martina/Desktop/concept_map/src/baselines/output_baseline"
    evaluator.evaluate_concept_maps(golden_folder_path, system_folder_path)