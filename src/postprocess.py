import spacy
from typing import List

class PostProcessor:
    """ Main class for post-processing """
    def __init__(self, model: str = "en_core_web_lg"):
        """ Init main params"""
        self.nlp = spacy.load(model)

    def remove_redundant_triples(self, file_path: str) -> List[str]:
        """ Remove redundant triples from a file """
        triples = self._read_triples_from_file(file_path)
        unique_triples = self._find_unique_triples(triples)
        return unique_triples

    def _read_triples_from_file(self, file_path: str) -> List[str]:
        """ Read triples from a file """
        triples = []
        with open(file_path, 'r') as file:
            for line in file:
                triples.append(line.strip())
        return triples

    def _find_unique_triples(self, triples: List[str]) -> List[str]:
        """ Find unique triples """
        unique_triples = []
        processed_triples = set()
        for triple in triples:
            elements = triple.split(', ')
            unique_elements = set(elements)
            if len(unique_elements) == 3:
                # Check for overlapping of elements in the triple
                overlap_threshold = 0.6
                for existing_triple in unique_triples:
                    existing_elements = existing_triple.split(', ')
                    overlap_count = sum(1 for element in elements if element in existing_elements)
                    if overlap_count / len(unique_elements) >= overlap_threshold:
                        # If overlap exceeds 80%, skip the triple
                        break
                else:
                    # If no overlap exceeds 60%, add the triple to unique triples
                    unique_triples.append(', '.join(elements))
        return unique_triples

if __name__ == "__main__":
    post_processor = PostProcessor()
    unique_triples = post_processor.remove_redundant_triples("/Users/martina/Desktop/concept_map/src/experiments/2024-01-30-17:14:04/101/relation/M1.txt")
    for triple in unique_triples:
        print(triple)