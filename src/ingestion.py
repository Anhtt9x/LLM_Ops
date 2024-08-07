from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
import boto3
import json
import os
import sys
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# bedrock_client=boto3.client(service_name="bedrock-runtime")
# embedding = BedrockEmbeddings(client=bedrock_client,model_id="")


embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

def data_ingestion():
    documents = PyPDFDirectoryLoader("data").load()

    text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

    docs = text_spliter.split_documents(documents)

    return docs


def get_vector_store(docs,embedding):


    doc_search =  FAISS.from_documents(docs,embedding)

    doc_search.save_local("faiss_index")

    

if __name__ == "__main__":
    docs=data_ingestion()
    get_vector_store(docs,embedding)