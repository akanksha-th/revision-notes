from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANCGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an AI assistant. Answer the user queries."),
        ("user", "Question:{question}")
    ]
)

st.title("Langchain Demo ChatBot with Ollama")
input_text = st.text_input("ask Your Query")


# NOTE: to run this, first run the command "ollama run gemma:2b" on the command prompt
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
