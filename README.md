# RAG Application

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using FastAPI and a custom dataset. The application loads a dataset, splits and embeds the documents, builds a vector index, and exposes an API endpoint for querying with natural language questions.

## Features
- Loads and processes a dataset from `dataset/dataset.json`
- Splits documents into manageable chunks
- Embeds text using a language model
- Builds a vector index for efficient retrieval
- Exposes a FastAPI endpoint for querying

## File Structure
```
main.py                # Main FastAPI app with RAG pipeline
RagFlanT5.ipynb        # Notebook for RAG with Flan-T5 model
RagGemini.ipynb        # Notebook for RAG with Gemini model
requirements.txt       # Python dependencies
/dataset/dataset.json  # Source dataset
```

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API server
```bash
uvicorn main:app --reload
```

### 3. Query the API
Send a POST request to `/ask` with a JSON body:
```json
{
  "question": "Your question here"
}
```

Example using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What is RAG?"}'
```

## Notebooks
- `RagFlanT5.ipynb`: RAG pipeline with Flan-T5 model
- `RagGemini.ipynb`: RAG pipeline with Gemini model

## Dataset
Place your dataset in `dataset/dataset.json` in the expected format.

## License
MIT License
