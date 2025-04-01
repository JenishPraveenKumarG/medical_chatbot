import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    return PromptTemplate(template="""
        Use the provided context to answer the user's question concisely. 
        If you don't know the answer, just say so. Do not generate information beyond the given context.
        
        Context: {context}
        Question: {question}
    """, input_variables=["context", "question"])

def load_llm():
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.getenv("HF_TOKEN")
    return HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID, temperature=0.5, model_kwargs={"token": HF_TOKEN, "max_length": 512})

def main():
    st.title("Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Enter your question")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            sources = [doc.metadata.get("source", "Unknown source") for doc in response["source_documents"]]
            
            result_to_show = f"{result}\n\n**Sources:**\n" + "\n".join(sources)
            
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
if __name__ == "__main__":
    main()