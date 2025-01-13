from flask import Flask, render_template, request
import json
import urllib.parse

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

@app.route("/compare/<encoded_question>")
def compare(encoded_question):
    question = urllib.parse.unquote(encoded_question)
    evaluated_data = evaluated_results.get(question, [])
    
    # Extract articles and relevant from evaluated_results
    evaluated_articles = set()
    evaluated_relevant = set()
    
    for item in evaluated_data:
        if "articles" in item:
            evaluated_articles.update(item["articles"])
        if "relevant" in item:
            evaluated_relevant.update(item["relevant"])
    
    # Load true data
    true_data = set(true_results.get(question, []))
    
    # Compare relevant values
    matched_relevant = evaluated_relevant & true_data
    non_matched_relevant = evaluated_relevant - true_data
    missing_relevant = true_data - evaluated_relevant

    # Compare articles (separately, not present in true_results.json for direct comparison)
    matched_articles = evaluated_articles & true_data
    non_matched_articles = evaluated_articles - true_data
    
    print("Question:", question)
    print("Matched Relevant:", matched_relevant)  # Debugging print
    print("Non-Matched Relevant:", non_matched_relevant)  # Debugging print
    print("Missing Relevant:", missing_relevant)  # Debugging print
    print("Matched Articles:", matched_articles)  # Debugging print
    print("Non-Matched Articles:", non_matched_articles)  # Debugging print
    
    return render_template(
        "compare.html",
        question=question,
        matched_relevant=list(matched_relevant),
        non_matched_relevant=list(non_matched_relevant),
        missing_relevant=list(missing_relevant),
        matched_articles=list(matched_articles),
        non_matched_articles=list(non_matched_articles),
        evaluated_articles=list(evaluated_articles),
        evaluated_relevant=list(evaluated_relevant),
        true=list(true_data),
    )

if __name__ == "__main__":
    app.run(debug=True)
