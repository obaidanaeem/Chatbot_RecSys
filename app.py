from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "<YOUR_GEMINI_API_KEY>")
if not GEMINI_API_KEY or "<YOUR_GEMINI_API_KEY>" in GEMINI_API_KEY:
    raise ValueError("Gemini API Key not set. Replace it.")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
volunteer_vectorstore = FAISS.load_local("volunteer_faiss_index", embedding_model, allow_dangerous_deserialization=True)
event_vectorstore = FAISS.load_local("event_faiss_index", embedding_model, allow_dangerous_deserialization=True)

app = FastAPI(title="VolunteerBazaar API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

def generate_response(query: str, vectorstore, top_k: int = 3):
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are VolunteerBazaar AI Assistant.
Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question: 
{query}

Answer:
"""
    response = llm.generate_content(prompt)
    return response.text.strip()

@app.post("/volunteer-query")
def volunteer_query(req: QueryRequest):
    try:
        result = generate_response(req.query, event_vectorstore, req.top_k)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/organization-query")
def organization_query(req: QueryRequest):
    try:
        result = generate_response(req.query, volunteer_vectorstore, req.top_k)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
