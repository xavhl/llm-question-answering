from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
DATASET_PATH = ROOT / DATA_DIR / "dataset.csv"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
TOP_K = int(os.getenv("TOP_K", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
