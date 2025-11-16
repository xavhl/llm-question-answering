# Demo

[HuggingFace App](https://huggingface.co/spaces/xavhl/aurora-question-answeirng) or via API endpoint according to `Use via API` option on Gradio interface


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
│  ├─ config.py             # constants, env-driven settings
│  └─ logger.py
├─ README.md
```

# Analyses

## Design Notes

I started designing in terms of robustness for varying scope of complexity,
1. Naive rule-based keyword matching 
   1. Rejected given our need for smart assistant question answering
2. TF-IDF (Term Frequency-Inverse Document Frequency) based retrieval
   1. I have implemented this method, though despite the low Complexity and high Efficiency, we could at best get back original message sentences that has high matching frequency score
   2. ref: [google colab](https://colab.research.google.com/drive/1Eqw8Zyxt1ND2adcGtOM-HUbmmqdiJxrp?usp=sharing)
3. Custom neural network with dataset training
   1. Rejected considering our current dataset scale; though fine-tuning on larger language models can be a feasible choice
4. Zero shot prompting on pretrained LLM with RAG
   1. Final proposed method

### Workflow 

(for method 4)

1. Build dataset by retrieving from given api
2. Construct vector database of dataset and corresponding RAG query function
   1. Selected `all-MiniLM-L6-v2` (~80MB) for its lightweight-ness with fast execution even on CPU (depending on our resource constraint, we can switch to larger model with stronger GPU)
3. Configure model with respective hyperparameters to create question answering pipeline to accept user input question
   1. Selected `Qwen/Qwen2-0.5B-Instruct` for efficiency 

### Future plans

This prototype is far from perfection, with following features that be further perfected:

- Validation and fact grounding
  - Given the stochastic generation by LLM, it is important to ensure that we have extracted and will provide the users near-perfectly precise, thus helpful information
    - fact checking shall be needed to verify LLM's answers against existing message data on the database
    - output validation will be needed depending on user request, such as making a time table consisting of recommended activities ahead, then the LLM output should be in the form of tabular format, for which several parser mechanisms by LangChain can be employed in the future
- More refined user targetted query
  - I observed that some RAG results contained messages that belong to different users, this noisy context wasted the context length and affected model performance
  - Hypothesis: it could be due to the limited model capacity in the first place for which we chose the lightweight model as our backbone 
  - Depending on resource constraint, an alternative workaround to ameliorate such situation can be 
    - using the conventional NLP method (TF-IDF) to filter out irrelevant user messages from the RAG results (TF-IDF results can be seen in *conventional NLP algorithm* -> *question answer retrieval* section of [google colab](https://colab.research.google.com/drive/1Eqw8Zyxt1ND2adcGtOM-HUbmmqdiJxrp?usp=sharing) )
- Vector database update functionality 
  - Add admin page to allow updating and re-build of embeddings and index (other concerns include, permission and protection of env key)
- Request caching 
  - Add short cache for transformer pipeline generator responses to avoid repeated LLM calls for identical queries

## Data Insights

Analyses results can be seen in *conventional NLP algorithm* -> *data analysis* section of [google colab](https://colab.research.google.com/drive/1Eqw8Zyxt1ND2adcGtOM-HUbmmqdiJxrp?usp=sharing) 

### User message lengths
```
mean       68.0
std         8.7
min         9.0
25%        63.0
50%        68.0
75%        73.0
max       105.0
Message length outliers (Z-score): 25
Message length outliers (IQR): 84
```

Given the above, we can see that the general message length is roughly between 60 and 80, which gives us a rough expectation of context content length. This helps us with the estimation of hyperparameter configuration such as `top_k` (i.e. number of matching messages as LLM prompt context) for RAG querying.

### Similarity

(along with sentiment, and spam detection)

```
Near-duplicate message pairs:       23
Messages with extreme sentiment:    49
Potential spam messages:            0
```

Example of duplicate:
| id  | user_id                              | user_name                            | timestamp       | message                          | message_length                                    | z_score | time_delta | sentiment   | is_spam |       |
| --- | ------------------------------------ | ------------------------------------ | --------------- | -------------------------------- | ------------------------------------------------- | ------- | ---------- | ----------- | ------- | ----- |
| 791 | 96e0b9a0-8471-4f2f-9d2b-db7520e52765 | 130f1fb9-2ddf-4049-ad0e-9a270f0cb561 | Vikram Desai    | 2024-11-11 14:24:40.167673+00:00 | Please book me a private jet to Paris for the ... | 54      | 1.606233   | 1133.266557 | 0       | FALSE |
| 609 | e60bc3f2-631d-472d-8a99-6281c702b6a3 | 6b6dc782-f40c-4224-b5d8-198a9070b097 | Thiago Monteiro | 2024-12-12 19:42:22.173546+00:00 | Can you book me a private jet to Paris for nex... | 56      | 1.376825   | 37.566794   | 0       | FALSE |

Given the above results,
- duplicate detection brings our attention to the issue that, some users have have similar message content that may cause confusion to model prediction
  - in the future, we may need to perform user-specific querying e.g. 
    - using simple TF-IDF for filtering as mentioned above, or 
    - another round of LLM analysis for filtering, or
    - refinement on RAG query procedure to enhance retrieval accuracy
- potentially speaking, sentiment information may be useful for our other future developments
- spam detection as anomaly detection serves as safeguard to enhance data quality, this can also be applied to other LLM safety aspects such as illegal content detection