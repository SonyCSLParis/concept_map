# -*- coding: utf-8 -*-
"""
Get aggregated results from all experiments
"""
import os
import json
import numpy as np
import pandas as pd
from loguru import logger
from src.build_table import build_table
####### PARAMS BELOW TO UPDATE
SAVE_FOLDER = "./experiments"
DATA_PATH = "./src/data/Corpora_Falke/Wiki/train/"
FOLDERS_CMAP = [x for x in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, x))]
DATE_START = "2024-02-08-11:00:00"
####################

COLUMNS = [
    'summary_method', 'summary_percentage',
    'ranking', 'ranking_perc_threshold',
    'confidence',
    'relation',
    'meteor_pr', 'meteor_re', 'meteor_f1',
    'rouge-2-pr', 'rouge-2-re', 'rouge-2-f1'
]

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        data = json.load(openfile)
    return data

def avg_results(metrics):

    res = {
        x+"_"+y: [] for x in ["meteor", "rouge-2"] for y in ["pr", "re", "f1"]
    }

    for _, info in metrics.items():
        for k1, val in info.items():
            for k2, metric in val.items():
                res[f"{k1}_{k2[:2]}"].append(metric)
    
    for k1, v in res.items():
        res[k1] = round(np.mean(v), 1)

    return res

def get_folders_exp_finished():
    exps = os.listdir(SAVE_FOLDER)
    exps = [x for x in exps if \
        all(y in os.listdir(os.path.join(SAVE_FOLDER, x)) \
            for y in ["metrics.json", "params.json", "logs.json"] + FOLDERS_CMAP) and \
                x >= DATE_START]
    exps = [
        (read_json(os.path.join(SAVE_FOLDER, x, "logs.json")),
         read_json(os.path.join(SAVE_FOLDER, x, "params.json")),
         read_json(os.path.join(SAVE_FOLDER, x, "metrics.json"))) for x in exps]
    exps = [x for x in exps if x[0].get("finished") == "yes"]
    return [x[1:] for x in exps]

def get_rebel_opt(params):
    options_rel = params["relation"]["options_rel"]
    local_rm = params["relation"]["local_rm"]
    if "rebel" in options_rel:
        x1 = "rebel\\_ft" if local_rm else "rebel\\_hf"
        x2 = "\\_dependency" if "dependency" in options_rel else ""
        return x1+x2
    return "+".join(options_rel)

def main():
    df_output = pd.DataFrame(columns=COLUMNS)
    folders_exp = get_folders_exp_finished()
    logger.info(f"Results on {len(folders_exp)} experiments")

    for params, metrics in folders_exp:
        avg_metrics = avg_results(metrics)
        curr_l = [
            params["summary"]["summary_method"],
            params["summary"]["summary_percentage"],
            params["ranking"]["ranking"],
            params["ranking"]["ranking_perc_threshold"] * 100,
            params["entity"]["confidence"],
            get_rebel_opt(params)
        ] + \
            [avg_metrics[f"{x}_{y}"] for x in ["meteor", "rouge-2"] \
                for y in ["pr", "re", "f1"]]
        df_output.loc[len(df_output)] = curr_l
        # df_output = df_output.append(pd.Series(curr_l, index=COLUMNS), ignore_index=True)
    print(df_output)
    df_output.sort_values(by=COLUMNS[:6]).to_csv("results.csv")

    latex_table = build_table(
        columns=["Summary", "Ranking", "Entity", "Relation", "METEOR", "ROUGE-2"],
        alignment="r"*len(COLUMNS),
        caption="Results for all systems on Wiki TRAIN",
        label="res-wiki-train-all-hyperparams",
        position="h",
        data=df_output.sort_values(by=COLUMNS[:6]).values,
        sub_columns=[x.replace("_", "\\_") for x in COLUMNS[:6]] + ["Pr", "Re", "F1"]*2,
        multicol=[2, 2, 1, 1, 3, 3],
        resize_col=2
    )
    print(latex_table)
    
    



if __name__ == '__main__':
    main()