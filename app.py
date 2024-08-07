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



