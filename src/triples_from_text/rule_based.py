# -*- coding: utf-8 -*-
"""
Rule-based triple extraction

For each sentence in the text
- getting noun phrases
- optional: filtering pronoun NPs, merging them
- finding dependency path between each pair of NPS
- filter and extract in the form (subject, predicate, object)
"""
import spacy
from collections import defaultdict

NLP = spacy.load("en_core_web_lg")

def filter_np(noun_phrases):
    """ Remove pronouns """
    return [x for x in noun_phrases if x.root.pos_ != 'PRON']


def merge_text(t1, s1, e1, t2, s2, e2):
    """ Merging text """
    if e1 < s2:
        return t1 + " " + t2, s1, e2
    return t2 + t1, s2, e1


def merge_np(noun_phrases):
    """ Merge NPs 
    Return : [(head, text)] """
    res = []
    heads = [x.root for x in noun_phrases]

    while noun_phrases:
        np = noun_phrases.pop()

        # checking if np root has children that are heads of other nps
        cands = [x for x in np.root.children if x in heads]
        if cands:
            corr_np = [child_np for child_np in noun_phrases if child_np.root in cands]
            for child_np in corr_np:
                noun_phrases.remove(child_np)
            text, start, end = merge_text(np.text, np[0].i, np[-1].i, corr_np[0].text, corr_np[0][0].i, corr_np[0][-1].i)
            curr_res = (np.root, text)
            for x in corr_np[1:]:
                text, start, end = merge_text(text, start, end, x.text, x[0].i, x[-1].i)
                curr_res = (np.root, text)
            res.append((np.root, text))
        
        elif np.root.head in heads:
            noun_phrases = [np] + noun_phrases

        else:
            res.append((np.root, np.text))
    
    return res[::-1]


def find_most_common_ancestor(t_1, t_2):
    """ Most common ancestor between two tokens """
    candidates = list(set(list(t_1.ancestors)).intersection(set(list(t_2.ancestors))))
    if len(candidates) ==  1:
        return candidates[0]
    for cand in candidates:
        if not any(cand in x.ancestors for x in [elt for elt in candidates if elt != cand]):
            return cand


def get_dep_path_child_ancestor(child, ancestor):
    """ Direct dependency path from a child to its ancestor """
    path = [child]
    curr_head = child.head
    while curr_head != ancestor:
        path.append(curr_head)
        curr_head = curr_head.head
    path.append(curr_head)
    return path


def find_dependency_path(t_1, t_2):
    """ Shortest dependency path between t_1 and t_2 
    Input = two tokens
    Output = path + root
    """
    if t_2 in t_1.ancestors:
        return find_dependency_path(t_2, t_1)

    # Either (1) t_1 is most common ancestor of t_2
    # (2) t_1 and t_2 have most common ancestor mca

    if t_1 in t_2.ancestors:  # (1)
        path = get_dep_path_child_ancestor(child=t_2, ancestor=t_1)

    else:  # (2)
        mca = find_most_common_ancestor(t_1=t_1, t_2=t_2)
        path = get_dep_path_child_ancestor(child=t_1, ancestor=mca)[:-1] + \
            get_dep_path_child_ancestor(child=t_2, ancestor=mca)[::-1]
    
    if path[0].i > path[-1].i:
        path = path[::-1]

    return path


def convert_path_triple_to_triple(path_triple, map_noun_head_to_text):
    """ Input = path of dependencies + mapping NP head -> NP text 
    Output = (subject, predicate, object) """
    subject_t = map_noun_head_to_text[path_triple[0]]
    predicate_t = ""
    object_t = map_noun_head_to_text[path_triple[-1]]
    index_verb = [i for i, x in enumerate(path_triple) if x.pos_ == "VERB"][-1]

    for i, token in enumerate(path_triple):
        if i not in [0, len(path_triple) -1]:
            if i < index_verb and not token.text in subject_t and not token.pos_ == "VERB":
                subject_t = f"{subject_t} {token.text}"
            if i == index_verb:
                predicate_t = token.text + " " +  \
                     ' '.join([x.text for x in token.children if \
                        x.dep_ in ['dobj'] and x.text not in object_t.split(" ")])
            if i > index_verb:
                if token.text not in object_t:
                    predicate_t = f"{predicate_t} {token.text}"

    return (subject_t.strip(), predicate_t.strip(), object_t.strip())


def get_triple_from_text_rb(text, filtering=True, merging=True):
    """ Extracting triples at sentence level """
    output = []
    doc = NLP(text)
    for sent in doc.sents:
        # Getting, filtering, merging NPs
        noun_chunks = list(sent.noun_chunks)
        noun_chunks = filter_np(noun_phrases=noun_chunks) if filtering else noun_chunks

        if merging:
            noun_chunks = merge_np(noun_phrases=noun_chunks)
        else:
            noun_chunks = [(x.root, x.text) for x in noun_chunks]

        # Dependency path between each pair of noun_chunks in the sentence
        dep_path = defaultdict(list)
        for i, t_1 in enumerate(noun_chunks):
            for t_2 in noun_chunks[i+1:]:
                curr_path = find_dependency_path(t_1[0], t_2[0])
                if any(x.pos_ == "VERB" for x in curr_path[1:-1]):
                    dep_path[t_1[0]].append(curr_path)
                    dep_path[t_2[0]].append(curr_path)
        dep_path = {k: sorted(triples, key=len) for k, triples in dep_path.items()}

        # Extract unique dependency path
        path_triples, text_triples = [], []
        for path in [x[0] for _, x in dep_path.items() if x]:
            curr_text = " ".join([y.text for y in path])
            if curr_text not in text_triples:
                path_triples.append(path)
                text_triples.append(curr_text)

        # Convert dependency paths to triples
        mapping_np = {token: text for (token, text) in noun_chunks}
        for path in path_triples:
            triple = convert_path_triple_to_triple(
                path_triple=path, map_noun_head_to_text=mapping_np)
            output.append(triple)

    return output


if __name__ == '__main__':
    TEXT = "Cellulose was founded in 1838 by the French chemist Anselme Payen, who isolated it from plant matter and determined its chemical formula."
    TRIPLES = get_triple_from_text_rb(text=TEXT, filtering=True, merging=True)
    for triple in TRIPLES:
        print(triple)
