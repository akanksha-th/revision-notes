import dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.messages import HumanMessage, SystemMessage
import streamlit as st

dotenv.load_dotenv()
model_name = "google/flan-t5-small"
hf_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    device=-1 # CPU
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

st.title("Basic LLM Chain")
question = st.text_input("Ask your Query")

messages = [
    SystemMessage(
        content='''You are an assistant knowledgeable about laws in India. 
        Only answer legal-awareness related questions.'''
    ),
    HumanMessage(content=question)
]

final_prompt = messages[0].content + "\nUser:" + messages[1].content

response = llm(final_prompt)
print("Assistant: ", response)