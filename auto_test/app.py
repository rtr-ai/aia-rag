from flask import Flask, render_template, request
import json

app = Flask(__name__)

# Load results
with open("auto_test/evaluated_results.json", "r") as f:
    evaluated_results = json.load(f)

with open("auto_test/true_results.json", "r") as f:
    true_results = json.load(f)

@app.route("/")
def index():
    questions = list(evaluated_results.keys())
    return render_template("index.html", questions=questions)

@app.route("/compare/<question>")
def compare(question):
    evaluated = set(evaluated_results.get(question, []))
    true = set(true_results.get(question, []))
    
    matched = evaluated & true
    non_matched = evaluated - true
    missing = true - evaluated
    
    return render_template(
        "compare.html",
        question=question,
        matched=list(matched),
        non_matched=list(non_matched),
        missing=list(missing),
        evaluated=list(evaluated),
        true=list(true),
    )

if __name__ == "__main__":
    app.run(debug=True)
