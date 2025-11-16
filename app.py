import os
import gradio as gr
import pandas as pd
from utils.config import DATASET_PATH
from qa.embeddings import Embedder
from qa.retriever import Retriever
from qa.model import load_generator
from qa.rag import answer_question

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Build or load embeddings + retriever (fast startup thanks to disk persistence)
embedder = Embedder()
embeddings = embedder.load_or_compute(df)
retriever = Retriever(df, embeddings)

# Load generator (this is the heavier op)
generator = load_generator(os.getenv("LLM_MODEL","Qwen/Qwen2-0.5B-Instruct"))

def respond(question, top_k=8):
    ans = answer_question(generator, retriever, question, top_k=top_k)
    return ans

with gr.Blocks() as demo:
    gr.Markdown("# LLM QA — RAG demo")
    with gr.Row():
        txt = gr.Textbox(label="Ask a question")
        k = gr.Slider(minimum=1, maximum=20, step=1, value=8, label="Top-k retrieval")
    out = gr.Textbox(label="Answer")
    submit = gr.Button("Ask")
    submit.click(respond, inputs=[txt, k], outputs=[out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
