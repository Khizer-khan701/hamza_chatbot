from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from .rag_pipeline import run_rag, build_conversational_chain, load_vectorstore


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup_event():
    vector_store = load_vectorstore()
    app.state.rag_chain = build_conversational_chain(vector_store)

class Chatbot(BaseModel):
    message: str

@app.on_event("startup")
def startup_event():
    app.state.vectorstores={}
    app.state.rag_chains={}

def get_chain():
    vector_store=load_vectorstore()
    chain=build_conversational_chain(vector_store)
    app.state.vectorstores=vector_store
    app.state.rag_chains=chain
    return chain

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/chatbot")
def chatbot_api(payload: Chatbot, request: Request):
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="User Query Not Found")  
    try:
        chat_history = []       
        chain = get_chain()
        result, _ = run_rag(message, chat_history, chain)     
        return JSONResponse(status_code=200, content={"response": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
