from langchain import LLMMathChain
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from flask import Flask, request, render_template_string

model_name = "google/flan-ts-small"
hf_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    device=-1,
    max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

app = Flask(__name__)
HTML_PAGE = """
<!DOCTYPE HTML>
<title>Math ChatBot</title>
<h2>Ask a Question
<form method="post>
    <input type="text" name="question" style="width:400px"; required>
    <input type="submit" value="Ask">
</form>
{% if response %}
    <h3> Assistant Response:
    <p>{{ response }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    if request.method == "POST":
        question = request.form["question"]

        result = math_chain.run(question)
        response = f"Math result: {result}"

    return render_template_string(HTML_PAGE, response=response)

if __name__ == "__main__":
    app.run(debug=True)