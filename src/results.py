import json
import numpy as np

def main(json_path):
    with open(json_path, "r", encoding="utf-8") as openfile:
        metrics = json.load(openfile)
    res = {
        "meteor": {x: [] for x in ["precision", "recall", "f1"]},
        "rouge-2": {x: [] for x in ["precision", "recall", "f1"]}
    }

    for _, info in metrics.items():
        for k1, val in info.items():
            for k2, metric in val.items():
                res[k1][k2].append(metric)
    
    for k1, v in res.items():
        for k2, l in v.items():
            res[k1][k2] = np.mean(l)

    return res
    

if __name__ == '__main__':
    JSON_PATH = "/mnt/disk2/ines/Projects/concept_map/experiments/2024-02-01-18:18:35/metrics.json"
    RES = main(JSON_PATH)

    for k1, v in RES.items():
        print(f"{k1}\t{v}")
