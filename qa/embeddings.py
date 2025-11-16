import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import EMBED_MODEL_NAME, EMBEDDINGS_PATH, BATCH_SIZE
import pandas as pd

class Embedder:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def prepare_texts(self, df: pd.DataFrame):
        return df.apply(lambda row: f"{row['user_name']} at {row['timestamp']}: {row['message']}", axis=1).tolist()

    def encode(self, texts):
        return self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)

    def load_or_compute(self, df):
        if EMBEDDINGS_PATH.exists():
            return np.load(EMBEDDINGS_PATH)
        texts = self.prepare_texts(df)
        embeddings = self.encode(texts)
        np.save(EMBEDDINGS_PATH, embeddings)
        return embeddings
