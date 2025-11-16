import faiss
import numpy as np
from utils.config import FAISS_INDEX_PATH
from pathlib import Path

class FaissStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = None
        self._load_index()

    def _load_index(self):
        if Path(FAISS_INDEX_PATH).exists():
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                return
            except Exception as e:
                print("Failed to read index:", e)
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray):
        # assume embeddings normalized for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def search(self, query_emb, k=5):
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb.astype('float32'), k)
        return distances, indices
