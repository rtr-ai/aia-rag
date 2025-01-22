from flask import Flask, render_template, request
import json
import urllib.parse
import pandas as pd 
from io import BytesIO
import webbrowser
from threading import Timer
import os

app = Flask(__name__)

# Load results
with open("auto_test/evaluated_results.json", "r", encoding='utf-8') as f:
    evaluated_results = json.load(f)

with open("auto_test/true_results.json", "r", encoding='utf-8') as f:
    true_results = json.load(f)

def calculate_statistics():
    report_data = []
    total_matched_percentage = 0
    count = 0

    # Loop through each question and calculate statistics
    for question, evaluated_data in evaluated_results.items():
        evaluated_set = set()
        for item in evaluated_data:
            if "articles" in item:
                evaluated_set.update(item["articles"])
            if "relevant" in item:
                evaluated_set.update(item["relevant"])

        true_data = set(true_results.get(question, []))
        matched_total = evaluated_set & true_data
        total_count = len(true_data)

        # Calculate matched percentage
        matched_percentage = (len(matched_total) / total_count * 100) if total_count > 0 else 0
        total_matched_percentage += matched_percentage
        count += 1

        # Add the data for each question to the report
        report_data.append({
            "Question": question,
            "Matched Percentage": f"{matched_percentage:.2f}%",
            # Join matched and missing articles with '|' instead of ','
            "Matched Articles": " | ".join(matched_total),
            "Missing Articles": " | ".join(true_data - matched_total),
        })

    # Calculate overall matched percentage
    average_matched_percentage = (total_matched_percentage / count) if count > 0 else 0

    return report_data, average_matched_percentage


@app.route("/")
def index():
    report_data, average_matched_percentage = calculate_statistics()
    questions = list(evaluated_results.keys())
    return render_template("index.html", questions=questions, report_data=report_data, average_matched_percentage=average_matched_percentage)
    
    #questions = list(evaluated_results.keys())
    # Calculate the average matched percentage across all questions
    #total_matched_percentage = 0
    #count = 0
    #for question in questions:
        #evaluated_data = evaluated_results.get(question, []) 
        #evaluated_set = set()
        #for item in evaluated_data:
            #if "articles" in item:
                #evaluated_set.update(item["articles"])
            #if "relevant" in item:
                #evaluated_set.update(item["relevant"])
        #true_data = set(true_results.get(question, []))
        #matched_total = evaluated_set & true_data
        #total_count = len(true_data)  
        #if total_count > 0:
            #matched_percentage = (len(matched_total) / total_count) * 100
            #total_matched_percentage += matched_percentage
            #count += 1
    #average_matched_percentage = (total_matched_percentage / count) if count > 0 else 0
    #return render_template("index.html", questions=questions, average_matched_percentage=average_matched_percentage)

@app.route("/compare/<path:encoded_question>")
def compare(encoded_question):
    question = urllib.parse.unquote(encoded_question)
    evaluated_data = evaluated_results.get(question, [])

    evaluated_set = set()
    for item in evaluated_data:
        if "articles" in item:
            evaluated_set.update(item["articles"])
        if "relevant" in item:
            evaluated_set.update(item["relevant"])

    true_data = set(true_results.get(question, []))
    matched_total = evaluated_set & true_data
    missing_total = true_data - evaluated_set

    matched_percentage = (len(matched_total) / len(true_data) * 100) if len(true_data) > 0 else 0
    missing_percentage = (len(missing_total) / len(true_data) * 100) if len(true_data) > 0 else 0

    return render_template(
        "compare.html",
        question=question,
        matched_total=matched_total,
        missing_total=missing_total,
        matched_percentage=matched_percentage,
        missing_percentage=missing_percentage
    )
    # Extract articles and relevant from evaluated_results
    #evaluated_articles = set()
    #evaluated_relevant = set()

    #evaluated_set = set()
    #for item in evaluated_data:
        #if "articles" in item:
            #evaluated_set.update(item["articles"])
        #if "relevant" in item:
            #evaluated_set.update(item["relevant"])
    
    #for item in evaluated_data:
        #if "articles" in item:
            #evaluated_articles.update(item["articles"])
        #if "relevant" in item:
            #evaluated_relevant.update(item["relevant"])
    
    # Load true data
    #true_data = set(true_results.get(question, []))
    
    # Compare relevant values
    #matched_relevant = evaluated_relevant & true_data
    #non_matched_relevant = evaluated_relevant - true_data
    #missing_relevant = true_data - evaluated_relevant

    # Compare articles (separately, not present in true_results.json for direct comparison)
    #matched_articles = evaluated_articles & true_data
    #non_matched_articles = evaluated_articles - true_data
    #missing_articles = true_data - evaluated_articles

    #matched_total = evaluated_set & true_data
    #missing_total = true_data - evaluated_set

    # Calculate percentages
    #whole = len(true_data) #evaluated_set
    #matched_percentage = (len(matched_total) / whole * 100) if whole > 0 else 0
    #missing_percentage = (len(missing_total) / whole * 100) if whole > 0 else 0
    
    #print("Question:", question)
    #print("Matched Relevant:", matched_relevant)  # Debugging print
    #print("Non-Matched Relevant:", non_matched_relevant)  # Debugging print
    #print("Missing Relevant:", missing_relevant)  # Debugging print
    #print("Matched Articles:", matched_articles)  # Debugging print
    #print("Non-Matched Articles:", non_matched_articles)  # Debugging print
    #print("Missing Articles:", missing_articles)  # Debugging print
    #print("Matched Total:", matched_total)  # Debugging print
    #print("Missing Total:", missing_total)  # Debugging print
    #print("Whole:", evaluated_set)
    
    #return render_template(
        #"compare.html",
        #question=question,
        #matched_relevant=list(matched_relevant),
        #non_matched_relevant=list(non_matched_relevant),
        #missing_relevant=list(missing_relevant),
        #matched_articles=list(matched_articles),
        #non_matched_articles=list(non_matched_articles),
        #missing_articles=list(missing_articles),
        #evaluated_articles=list(evaluated_articles),
        #evaluated_relevant=list(evaluated_relevant),
        #true=list(true_data),
        #matched_total=matched_total,
        #missing_total=missing_total,
        #matched_percentage=matched_percentage,
        #missing_percentage=missing_percentage
    #)

@app.route("/generate-report")
def generate_report():
    report_data, average_matched_percentage = calculate_statistics()

    df = pd.DataFrame(report_data)
    df.loc[len(df)] = ["Overall Average", f"{average_matched_percentage:.2f}%", "", ""]
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Match Report")
        worksheet = writer.sheets["Match Report"]
        worksheet.set_column("A:D", 30)

    output.seek(0)
    return send_file(output, as_attachment=True, download_name="match_report.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Prevent the browser from opening twice due to the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
