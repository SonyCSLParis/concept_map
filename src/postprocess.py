import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

class PostProcessor:
    """ Main class for post-processing """
    def __init__(self, embed_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 threshold: float = 0.7):
        self.embed_model = SentenceTransformer(embed_model)
        self.threshold = threshold

    def filter_on_embeddings(self, triples: Dict[str, List[tuple]]):
        """ filter based on cosine similarity of embeddings """
        output = {"output": []}
        triples_orig = [x for _, v in triples.items() for x in v]
        triples = [" ".join(x) for _, v in triples.items() for x in v]
        nb = len(triples)
        embeddings = self.embed_model.encode(triples)

        cosine_similarity = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                cosine_similarity[i][j]=util.pytorch_cos_sim(embeddings[i], embeddings[j])

        indexes_triples = []
        for i, triple in enumerate(triples_orig):
            values = cosine_similarity[i, indexes_triples]
            if all(x < self.threshold for x in values):
                indexes_triples.append(i)
                output["output"].append(triple)
        return output


    def remove_redundant_triples(self, triples: Dict[str, List[tuple]]) -> Dict[str, List[tuple]]:
        """ Find unique triples """
        processed_triples = {}
        for category, values in triples.items():
            unique_triples = []
            for triple in values:
                elements = triple
                unique_elements = set(elements)
                if len(unique_elements) == 3:
                    # Check for overlapping of elements in the triple
                    overlap_threshold = 0.6
                    for existing_triple in unique_triples:
                        existing_elements = existing_triple
                        overlap_count = sum(1 for element in elements if element in existing_elements)
                        if overlap_count / len(unique_elements) >= overlap_threshold:
                            # If overlap exceeds 60%, skip the triple
                            break
                    else:
                        # If no overlap exceeds 60%, add the triple to unique triples
                        unique_triples.append(elements)
            processed_triples[category] = unique_triples
        return processed_triples
    
    @staticmethod
    def update_mapping(mapping):
        """
        Original: {<text label>: <KG IRI>}
        Output: {<text label>: <text label unique representant>}
        """
        iris = set(val for _, val in mapping.items())
        iri_to_label = {iri: sorted([x for x, val in mapping.items() if val == iri], key=len) for iri in iris}
        iri_to_ulabel = {k: v[0] for k, v in iri_to_label.items()}
        return {k: iri_to_ulabel[v] for k, v in mapping.items()}

    @staticmethod
    def replace_text(y, mapping):
        """ replace labels in text """
        for k, v in mapping.items():
            y = y.replace(k, v)
        return y

    def replace_ulabel(self, res, mapping):
        """ Replace labels by unique representants in the mapping """
        new_res = {x: [] for x in res}
        for x, tl in res.items():
            for triple in tl:
                ntl = [self.replace_text(y, mapping) for y in triple]
                new_res[x].append(tuple(ntl))
        return new_res

    def __call__(self, res, mapping):
        """ Complete postprocessing """
        new_mapping = self.update_mapping(mapping=mapping)
        res = self.replace_ulabel(res=res, mapping=new_mapping)
        # triples = self.remove_redundant_triples(triples=res)
        triples = self.filter_on_embeddings(triples=res)

        return triples


if __name__ == "__main__":
    post_processor = PostProcessor()
    MAPPING = {"7 World Trade Center": "http://WTC", "WTC": "http://WTC"}
    TRIPLES = {'rebel': [('WTC', 'area', '52'), ('7 World Trade Center', 'elevation above sea', '52'), ('7 World Trade Center', 'instance of', 'building')], 'dependency': [('A', 'B', 'C'),('A', 'B', 'C')]}
    U_TRIPLES = post_processor(res=TRIPLES, mapping=MAPPING)
    for c, triples_list in U_TRIPLES.items():
        print(f"{c}:")
        for t in triples_list:
            print(t)
