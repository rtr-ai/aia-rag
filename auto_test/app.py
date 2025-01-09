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
@app.route("/compare/<question>")
def compare(question):
    # Get evaluated and true results for the question
    evaluated = evaluated_results.get(question, [])
    true = true_results.get(question, [])

    # Normalize: Extract only unique identifiers (e.g., titles)
    evaluated_normalized = {item.split(" (")[0] for item in evaluated}
    true_normalized = {item.split(" (")[0] for item in true}

    # Perform set operations
    matched = evaluated_normalized & true_normalized
    non_matched = evaluated_normalized - true_normalized
    missing = true_normalized - evaluated_normalized

    return render_template(
        "compare.html",
        question=question,
        matched=list(matched),
        non_matched=list(non_matched),
        missing=list(missing),
        evaluated=list(evaluated_normalized),
        true=list(true_normalized),
    )

if __name__ == "__main__":
    app.run(debug=True)
