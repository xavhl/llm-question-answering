from qa.retriever import Retriever

def build_prompt(question: str, docs: list):
    if not docs:
        return "No context available.\n\nQuestion: " + question + "\nAnswer:"
    context = "\n".join([f"User: {d['user_name']} | Time: {d['timestamp']} | Msg: {d['message']}" for d in docs])
    prompt = f"""Based on the following chat messages, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def answer_question(generator, retriever: Retriever, question: str, top_k: int = 8):
    docs = retriever.retrieve(question, k=top_k)
    prompt = build_prompt(question, docs)
    out = generator(prompt, return_full_text=False)
    text = out[0].get('generated_text', '')
    if text.startswith('Answer:'):
        text = text[len('Answer:'):].strip()
    return text
