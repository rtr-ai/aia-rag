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
    question_text = question.replace("_", " ")  # Handle spaces in URLs if necessary
    
    # Load the evaluated and true JSONs
    evaluated = set([normalize_string(chunk) for chunk in evaluated_results.get(question_text, [])])
    true = set([normalize_string(chunk) for chunk in true_results.get(question_text, [])])
    
    # Calculate matched, non-matched, and missing chunks
    matched = evaluated & true
    non_matched = evaluated - true
    missing = true - evaluated
    
    return render_template(
        "compare.html",
        question=question_text,
        matched=list(matched),
        non_matched=list(non_matched),
        missing=list(missing),
        evaluated=list(evaluated),
        true=list(true),
    )

def normalize_string(s):
    """Normalize strings by removing special characters, extra spaces, and making them lowercase."""
    return " ".join(s.lower().split()).replace("„", "").replace("“", "")

if __name__ == "__main__":
    app.run(debug=True)
