
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')

work_dir = './'
df = pd.read_csv(work_dir + 'dataset.csv')

# Step 1: Initialize embedding model (lightweight: all-MiniLM-L6-v2 ~80MB, fast on CPU/GPU)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Generate embeddings for messages; load if exist

def generate_embeddings(df):
    df['text_for_embedding'] = df.apply(lambda row: f"{row['user_name']} at {row['timestamp']}: {row['message']}", axis=1)
    embeddings = embedding_model.encode(df['text_for_embedding'].tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

embeddings_path = work_dir + 'embeddings.npy'

if not os.path.exists(embeddings_path):
    embeddings = generate_embeddings(df)
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")
else:
    embeddings = np.load(embeddings_path)
    print(f"Embeddings loaded from {embeddings_path}")


# Step 3: Create FAISS index (lightweight in-memory vector DB)
dimension = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalize for cosine)
faiss.normalize_L2(embeddings)  # Normalize for cosine sim
index.add(embeddings.astype('float32'))

# Store metadata (original rows) for retrieval
metadata = df.to_dict('records')

# Function to retrieve top-k relevant messages
def retrieve_relevant_docs(query, k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    relevant_docs = [metadata[i] for i in indices[0] if i < len(metadata)]  # Safe indexing
    return relevant_docs


# Step 4: Initialize lightweight LLM for generation

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # Use pipeline for easier generation
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return generator

generator = get_model(model_name='Qwen/Qwen2-0.5B-Instruct')


# Step 5: RAG Query Function

def get_prompt(question, top_k):
    # Retrieve
    relevant_docs = retrieve_relevant_docs(question, top_k)
    if not relevant_docs:
        return "No relevant information found."
    
    # Build context
    context = "\n".join([f"User: {doc['user_name']} | Time: {doc['timestamp']} | Msg: {doc['message']}" for doc in relevant_docs])
    
    # Prompt template (simple for lightweight LLM)
    prompt = f"""Based on the following chat messages, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""

    return prompt

def rag_query(generator, question, top_k=10):
    
    prompt = get_prompt(question, top_k)
    
    # Generate
    response = generator(prompt, return_full_text=False)
    answer = response[0]['generated_text'].strip()
    
    # Post-process: Extract just the answer part
    if answer.startswith('Answer:'):
        answer = answer[7:].strip()
    
    # print('prompt', prompt)
    
    return answer
