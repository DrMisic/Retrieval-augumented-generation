
# Retrieval-Augmented Generation (RAG) System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer natural language questions using a hybrid document retrieval and generation approach. It incorporates techniques like semantic chunking, query expansion, hybrid retrieval, reranking, and evaluation to maximize accuracy and relevance.

---

## Features

- **Hybrid Retrieval:** Combines dense vector search (FAISS) with lexical search (BM25) to increase recall.
- **Query Expansion:** Generates multiple query formulations using a large language model (LLM) to improve retrieval diversity.
- **Cross-Encoder Reranking:** Refines retrieved documents to select the most relevant ones.
- **Semantic Chunking:** Splits documents based on semantic boundaries to preserve coherence.
- **Evaluation Metrics:** Measures performance using Exact Match, ROUGE-L, and METEOR.
- **User Interface:** Gradio-powered web interface for interactive question answering.

---

## Architecture

```
User Question
     │
     ▼
Query Generator (FLAN-T5)
     │
     ▼
Expanded Queries
     │
     ▼
Ensemble Retriever (FAISS + BM25)
     │
     ▼
Reranker (Cross-Encoder)
     │
     ▼
Selected Context
     │
     ▼
Answer Generator (FLAN-T5)
     │
     ▼
Final Answer
```

---

## Components

### 1. Model Initialization

```python
def initialize_llm_and_embeddings()
```

- **LLM:** `google/flan-t5-base` is used for query generation and final answer generation.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` used for semantic search.

---

### 2. Document Loading

```python
def load_documents()
```

- Loads the "rag-mini-wikipedia" dataset from Hugging Face.
- Wraps passages into LangChain `Document` objects.

---

### 3. Semantic Chunking

```python
def split_documents(docs, embeddings)
```

- Splits documents into chunks at points of high semantic change.
- Improves context granularity and relevance.

---

### 4. Hybrid Retrieval

```python
def create_retrievers(splits, embeddings)
```

- Uses FAISS for dense retrieval.
- Uses BM25 for traditional keyword matching.
- Combines both via an `EnsembleRetriever`.

---

### 5. Query Generation

```python
def create_query_generator(llm)
```

- Produces ten query reformulations using the LLM to enhance retrieval diversity.

---

### 6. Document Deduplication

```python
def get_unique_union(documents)
```

- Deduplicates documents retrieved across multiple queries.

---

### 7. Cross-Encoder Reranking

```python
def create_reranking_retriever(...)
```

- Uses a cross-encoder to rerank documents based on relevance to the original query.
- Selects the top `k` documents for context compression.

---

### 8. RAG Chain Construction

```python
def build_rag_chain(llm, retrieval_chain)
```

- Combines the retriever and LLM into a complete RAG pipeline.
- Provides answers based on compressed context and question input.

---

### 9. Evaluation

```python
def evaluate(rag_chain, test_data)
```

- Evaluates the system using a test set of QA pairs.
- Computes:
  - **Exact Match**
  - **ROUGE-L**
  - **METEOR**

---

## Metrics

| Metric       | Description                              |
|--------------|------------------------------------------|
| Exact Match  | Checks for exact string match            |
| ROUGE-L      | Measures longest common subsequence      |
| METEOR       | Evaluates based on semantic similarity   |

---

## How to Run

## Dependencies

Install the required libraries using pip:

```bash
pip install langchain transformers datasets sentence-transformers faiss-cpu evaluate nltk gradio
```

---


### Command-Line Execution

```bash
python retrival_augumented_generation.py
```

This will:
- Load documents and split them semantically.
- Initialize all models and retrievers.
- Evaluate the RAG system on a 10% sample of the dataset.
- Output average performance metrics.

---

## Model Overview

| Function             | Model                                  |
|----------------------|----------------------------------------|
| Language Generation  | `google/flan-t5-base`                  |
| Embeddings           | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-Encoder Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---

## Configuration Parameters

| Parameter                | Value  | Description                                     |
|--------------------------|--------|-------------------------------------------------|
| `max_new_tokens`         | 256    | Max tokens the LLM generates                    |
| `temperature`            | 0.1    | Lower temperature ensures more deterministic output |
| `k`                      | 50     | Number of top documents retrieved               |
| `ensemble_weights`       | 0.5/0.5| Weighting of FAISS and BM25 retrieval methods   |
| `top_k` reranker         | 5      | Top-k documents used after reranking            |
| `breakpoint_threshold`   | 95     | Percentile used for chunking boundaries         |
| `test_size`              | 0.1    | Fraction of dataset used for evaluation         |

---

## Example Output

```
Question: Who discovered gravity?
Expected: Isaac Newton
Predicted: Isaac Newton


Question: Who suggested Lincoln grow a beard?
Expected: 11-year-old Grace Bedell
Predicted: Grace Bedell

Question: What did The Legal Tender Act of 1862 establish?
Expected: The United States Note, the first paper currency in United States history
Predicted: United States Note

Question: What is Finland's economy like?
Expected: A highly industrialised, free-market economy
Predicted: Highly industrialised, free-market economy with a per capita output equal to that of other western economies such as Sweden, the UK, France and Germany. The largest sector of the economy is services at 65.7 percent, followed by manufacturing and refining at 31.4 percent. Primary production is low at 2.9 percent, reflecting the fact that Finland is a resource-poor country. With respect to foreign trade, the key economic sector is manufacturing.


```

---


