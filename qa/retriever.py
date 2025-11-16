import numpy as np
from qa.embeddings import Embedder
from qa.vectorstore import FaissStore
from utils.config import TOP_K
import pandas as pd

class Retriever:
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embedder = Embedder()
        self.store = FaissStore(embeddings.shape[1])
        # If store empty, add embeddings
        if self.store.index.ntotal == 0:
            self.store.add(embeddings)

    def retrieve(self, query: str, k: int = TOP_K):
        q_emb = self.embedder.encode([query])
        dists, inds = self.store.search(q_emb, k=k)
        rows = []
        for i in inds[0]:
            if i < len(self.df):
                rows.append(self.df.loc[i].to_dict())
        return rows
