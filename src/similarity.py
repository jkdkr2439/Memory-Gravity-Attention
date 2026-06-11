"""
Similarity module (§3 of spec)
Toy mode: TF-IDF cosine similarity (no external models needed)
Embedding mode: sentence-transformers (optional, heavier)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ToySimilarity:
    """TF-IDF based similarity. Fast, no GPU needed."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.fitted = False
        self.embeddings = None
    
    def fit(self, texts):
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.fitted = True
        return self.embeddings
    
    def encode(self, text):
        if not self.fitted:
            raise ValueError("Call fit() first")
        return self.vectorizer.transform([text]).toarray()[0]
    
    def query_similarity(self, query_text, node_texts=None):
        """Compute similarity between query and all fitted nodes."""
        q_vec = self.encode(query_text).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.embeddings)[0]
        return sims  # (n_nodes,)
    
    def pairwise(self):
        """Full pairwise similarity matrix."""
        return cosine_similarity(self.embeddings)


class EmbeddingSimilarity:
    """Sentence-transformer based. Better quality, slower."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
    
    def fit(self, texts):
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        return self.embeddings
    
    def encode(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0]
    
    def query_similarity(self, query_text, node_texts=None):
        q_vec = self.encode(query_text).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.embeddings)[0]
        return sims
