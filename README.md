# Demo

[HuggingFace App](https://huggingface.co/spaces/xavhl/aurora-question-answeirng) or via [API endpoint](./api.yaml)

Example:
```shell
$ curl -X POST https://xavhl-aurora-question-answeirng.hf.space/gradio_api/call/respond -s -H "Content-Type: application/json" -d '{
	"data": [
							"When is Layla planning her trip to London?",
							8
	]}' \
	| awk -F'"' '{ print $4}'  \
	| read EVENT_ID; curl -N https://xavhl-aurora-question-answeirng.hf.space/gradio_api/call/respond/$EVENT_ID
```

# Structure
```
llm-qa/
├─ app.py                   # Gradio app (entrypoint for Hugging Face Spaces)
├─ qa/
│  ├─ __init__.py
│  ├─ embeddings.py         # embedding model & persistence
│  ├─ vectorstore.py        # FAISS wrapper (disk-backed + safe loads)
│  ├─ retriever.py          # retrieval logic (kNN)
│  ├─ model.py              # LLM loading/generation wrapper
│  └─ rag.py                # RAG prompt assembly + orchestrator
├─ requirements.txt
├─ dataset.csv
├─ data/
│  ├─ embeddings.npy
│  └─ faiss.index
├─ utils/
│  └─ config.py             # constants, env-driven settings
├─ README.md
```

# Analysis

Refer to [docs/analysis.md](./docs/analysis.md)