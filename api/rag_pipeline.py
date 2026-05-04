
import os
from dotenv import load_dotenv
from fastapi import HTTPException

# FAISS aur Community imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# RAG Chains
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_groq import ChatGroq
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Local FAISS index ka path
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_vectorstore():
    try:
        # Local folder se FAISS index load ho raha hai
        if os.path.exists(DB_FAISS_PATH):
            vector_store = FAISS.load_local(
                DB_FAISS_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store
        else:
            raise Exception("FAISS local index not found at the specified path.")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail={"message": str(e)})

def build_conversational_chain(vector_store):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=250,
    )
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        create_history_retriever = create_history_aware_retriever(
            model, retriever, contextualize_q_prompt
        )
        
        # Updated System Prompt for Portfolio Website
        system_prompt = """
        You are a professional AI Assistant for a portfolio website. Your goal is to introduce the developer and answer questions about their expertise, projects, and experience based ONLY on the provided context.
        
        Guidelines:
        1. Be polite, professional, and concise (limit response to 4-5 lines).
        2. If the user asks about the developer's skills or projects, use the context to provide accurate details.
        3. If the information is not in the context, politely state that you don't have that specific information.
        4. Include code snippets only if they are directly relevant to the user's technical query and present in the context.
        
        Context:
        {context}
        """
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        document_chain = create_stuff_documents_chain(model, qa_prompt)
        return create_retrieval_chain(create_history_retriever, document_chain)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail={"message": str(e)})

def run_rag(query, chat_history=None, chain=None):
    try:
        if chain is None:
            vector_store = load_vectorstore()
            chain = build_conversational_chain(vector_store)
        
        if chat_history is None:
            chat_history = []
            
        chatbot_messages = []
        for msg in chat_history:
            if msg.get("role") == "user":
                chatbot_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                chatbot_messages.append(AIMessage(content=msg.get('content', "")))
        
        response = chain.invoke({"input": query, "chat_history": chatbot_messages})
        return response["answer"], chain
        
    except Exception as e:
        # 'return' ki jagah 'raise' use karein taake unpacking error na aaye
        raise HTTPException(status_code=400, detail={"message": str(e)})