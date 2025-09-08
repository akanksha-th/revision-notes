import dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

dotenv.load_dotenv()
model_name = "google/flan-t5-small"
hf_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    device=-1, # CPU
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate(
    input_variable=["question"],
    template="""
    You are a friendly teacher for 3-5 year old kids.
    Answer their questions in a simple and fun way.
    Always give short, clear answers in 2-3 sentences.
    Use easy words and sometimes emojis.

    Question: {question}
    Answer:
    """
)
chain = LLMChain(prompt=prompt, llm=llm)

st.title("Play School Chatbot (LLMChain + HuggingFace)")
st.info("The llm used being small, the model hallucinates a lottt!")
question = st.text_input("Ask your query:")

if question:
    response = chain.invoke({"question": question})  # ✅ modern API
    st.write("Assistant:", response["text"])