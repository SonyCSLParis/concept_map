import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
from typing import Union, List

class ImportanceRanker:
    def __init__(self, options: List[str] = ["page_rank", "text_rank", "tfidf","word2vec"]):
        self.options_ranker = ["page_rank", "text_rank", "tfidf","word2vec"]
        self.options_to_f = {
            "page_rank": self.compute_page_rank,
            "text_rank": self.compute_text_rank,
            "tfidf": self.tfidf_importance_ranking,
            "word2vec": self.word_embedding_similarity,

        }
        self.check_params(options=options)

        self.params = {
            "options": options,
        }
        self.options = options

    def check_params(self, options):
        """ Check that each parameter is correct for the options """
        if any(x not in self.options_ranker for x in options):
            raise ValueError(f"All options in `options` must be from {self.options_ranker}")

    def compute_page_rank(self, sentences):
        """Compute the importance ranking of a list of sentences based on page rank"""
        if len(sentences) == 0:
            return []

        vectorizer = CountVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)

        if len(vectorizer.vocabulary_) == 0:
            return []

        similarity_matrix = cosine_similarity(sentence_vectors)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        print(ranked_sentences)
        return ranked_sentences

    def tfidf_importance_ranking(self, sentences):
        """Compute the importance ranking of a list of sentences based on tf-idf embedding"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_scores = tfidf_matrix.sum(axis=1).A1
        ranked_indices = tfidf_scores.argsort()[::-1]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        print(ranked_sentences)
        return ranked_sentences

    def compute_text_rank(self, sentences):
        """Using the Summa library --> be careful sometimes return empty summary"""
        text = ' '.join(sentences)
        summary = summarizer.summarize(text)
        sentences = summary.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        print(sentences)
        return sentences

    def train_word2vec_model(self, sentences, model_path='word2vec.model'):
        """Train and save a Word2Vec model"""
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
        model.save(model_path)

    def load_word2vec_model(self, word2vec_file):
        """Load Word2Vec model from file"""
        return Word2Vec.load(word2vec_file)

    def average_embedding(self, sentence, model, dim=100):
        """Compute the average word embedding for a sentence"""
        embedding = np.zeros(dim)
        count = 0
        for word in sentence.split():
            if word in model.wv:
                embedding += model.wv[word]
                count += 1
        if count > 0:
            embedding /= count
        return embedding

    def word_embedding_similarity(self, sentences, model_path='word2vec.model'):
        """Compute the importance ranking of a list of sentences based on Word2Vec embeddings"""
        word2vec_model = ImportanceRanker.load_word2vec_model(model_path)
        sentence_embeddings = [ImportanceRanker.average_embedding(sentence, word2vec_model) for sentence in sentences]
        similarity_matrix = cosine_similarity(sentence_embeddings)
        importance_scores = np.sum(similarity_matrix, axis=1)
        ranked_indices = importance_scores.argsort()[::-1]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        print(ranked_sentences)
        return ranked_sentences

if __name__ == '__main__':
    if __name__ == "__main__":
        sentences = [
            "Automatic summarization is the process of reducing a text document with a computer program in order to create a summary that retains the most important points of the original document",
            "As the problem of information overload has grown, and as the quantity of data has increased, so has interest in automatic summarization",
            "Technologies that can make a coherent summary take into account variables such as length, writing style and syntax",
            "An example of the use of summarization technology is search engines such as Google. Document summarization is another."
        ]

        ranker = ImportanceRanker()
        ranked_page_rank = ranker.compute_page_rank(sentences)
        ranked_tfidf = ranker.tfidf_importance_ranking(sentences)
        ranked_text_rank = ranker.compute_text_rank(sentences)
        ranker.train_word2vec_model(sentences)
        ranked_word2vec = ranker.word_embedding_similarity(sentences)