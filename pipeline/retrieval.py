from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

class PaperRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.abstracts = []

    def build_index(self, abstracts):
        """Build FAISS index from a list of paper abstracts."""
        self.abstracts = abstracts
        embeddings = self.model.encode(abstracts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"FAISS index built: {len(abstracts)} papers, {dimension} dimensions")

    def find_similar(self, query_abstract, top_k=5):
        """Find the top_k most similar abstracts to the query."""
        query_embedding = self.model.encode([query_abstract]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "abstract": self.abstracts[idx],
                "distance": float(distances[0][i]),
                "rank": i + 1
            })
        return results

    def save(self, path):
        """Save index and abstracts to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "papers.index"))
        with open(os.path.join(path, "abstracts.json"), "w") as f:
            json.dump(self.abstracts, f)

    def load(self, path):
        """Load index and abstracts from disk."""
        self.index = faiss.read_index(os.path.join(path, "papers.index"))
        with open(os.path.join(path, "abstracts.json"), "r") as f:
            self.abstracts = json.load(f)
        print(f"Loaded index: {len(self.abstracts)} papers")