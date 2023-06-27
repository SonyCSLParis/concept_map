import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from settings import *


def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def compute_text_rank(sentences):
    if len(sentences) == 0:
        ranked_senteces = []
        return ranked_senteces

    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    if len(vectorizer.vocabulary_) == 0:
        ranked_senteces = []
        return ranked_senteces

    similarity_matrix = cosine_similarity(sentence_vectors)
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    return ranked_sentences


def process_parent_folder(parent_folder, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)  # Create the output folder if it doesn't exist

    for filename in os.listdir(parent_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(parent_folder, filename)
            output_file_name = filename.replace(".txt", "_importance_ranking.txt")
            output_file_path = os.path.join(output_folder_path, output_file_name)

            with open(input_file_path, "r") as input_file:
                text = input_file.read()
                sentences = preprocess_text(text)

            ranked_sentences = compute_text_rank(sentences)
            ranked_sentences_20 = ranked_sentences[:20]

            with open(output_file_path, "w") as output_file:
                for triplet in ranked_sentences_20:
                    output_file.write(f"{triplet}\n")

            print(ranked_sentences_20)
