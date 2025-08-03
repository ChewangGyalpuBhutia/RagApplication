from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.base import Embeddings
from typing import Any, List

import os
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 

# Load environment variables
load_dotenv()

# --- Custom Embeddings Class ---
class PyTorchEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# --- Data and RAG pipeline setup ---
def load_and_process_data(file_path):
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["title"] = record.get("title", "")
        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="text",
        metadata_func=metadata_func,
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=125, chunk_overlap=25)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_store(texts):
    embeddings = PyTorchEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

def initialize_qa_chain(db):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.5, max_output_tokens=320
    )
    retriever = db.as_retriever(search_kwargs={"k": 9})
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

# --- FastAPI setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Load everything at startup
texts = load_and_process_data("dataset/dataset.json")
db = create_vector_store(texts)
qa_chain = initialize_qa_chain(db)
print("Server ready!")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    result = qa_chain.invoke({"query": request.query})
    sources = [
        doc.metadata.get("title", "") for doc in result.get("source_documents", [])
    ]
    return QueryResponse(answer=result["result"], sources=sources)
