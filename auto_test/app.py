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
    evaluated = evaluated_results.get(question, [])
    true = true_results.get(question, [])
    
    # Convert both lists to sets for easier comparison
    evaluated_set = set(evaluated)
    true_set = set(true)
    
    # Calculate matched, non-matched, and missing chunks
    matched = evaluated_set.intersection(true_set)
    non_matched = evaluated_set.difference(true_set)
    missing = true_set.difference(evaluated_set)
    
    return render_template(
        "compare.html",
        question=question,
        matched=list(matched),
        non_matched=list(non_matched),
        missing=list(missing),
        evaluated=evaluated,
        true=true,
    )

if __name__ == "__main__":
    app.run(debug=True)
