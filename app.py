import os
import json
import sys
import boto3
import streamlit as st 
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from src.ingestion import data_ingestion ,get_vector_store
from src.retrival import get_llama2 ,get_response_llm
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")


def main():
    st.set_page_config("QA and Doc")
    st.header("QA with LLMOPs")

    user_input = st.text_input("User_input...")

    with st.sidebar:
        st.title("Update or create vector store...")
        if st.button("Action"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs,embedding)
                st.success("Done")
        
    if st.button("Llama_model"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",embedding,allow_dangerous_deserialization=True)

            llm = get_llama2()

            result = get_response_llm(llm=llm, vectorstore_faiss=faiss_index,query=user_input)

            st.write(result)
            st.success("Done")

if __name__ =="__main__":
    main()
