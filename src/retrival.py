from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM ,pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_aws import Bedrock
import boto3
import torch
from langchain_core.prompts.prompt import PromptTemplate
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
# bedrock_client = boto3.client(service_name="bedrock-runtime")

# def get_bedrock_llm():
#     llm = Bedrock(client=bedrock_client,model_id="",model_kwargs={"max_new_tokens":512})
#     return llm


load_dotenv()

from huggingface_hub import login

login(os.getenv("HUGGING_FACE_API_TOKEN"))

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])


tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", device_map='auto')
pipeline = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                   trust_remote_code=True,
                   device_map='auto',
                   max_length=1000,
                   do_sample=True,
                   top_k=10,
                   num_return_sequences=1,
                    truncation=True,
                   eos_token_id=tokenizer.eos_token_id,)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature":0})
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

def get_response_llm(llm, vectorstore_faiss, query):
    qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                retriever=vectorstore_faiss.as_retriever(search_type="similarity",
                                                                        search_kwargs={"k":3}),
                                                                        return_source_documents=True,
                                                                        chain_type_kwargs={"prompt":prompt})
    
    response = qa.invoke({"query":query})
    
    return response['result']


if __name__ =="__main__":
    get_response_llm(llm=llm, vectorstore_faiss=FAISS.load_local("faiss_index",embedding,allow_dangerous_deserialization=True),
                     query="What is RAG token ?")