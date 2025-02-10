from flask import Flask, render_template, request, send_file
import json
import os
import re
from tqdm import tqdm
import requests
from openai import OpenAI
import pandas as pd
from io import BytesIO
import webbrowser
from threading import Timer
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Read the markdown file
with open("data/fragen-qa.md", 'r', encoding='utf-8') as file: 
    markdown_content = file.read()

def excel_data():
    # Prepare data for the Excel file
    data = []
    for item in tqdm(faq, desc="Fetching data for excel report", unit="question"):
        question = item['question']
        existing_answer = item['answer'].split('Quellen:')[0].strip()
        existing_quellen = item['answer'].split('Quellen:')[1].strip()
        new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == question), "").split("Quellen:")[0].strip()
        quellen = next((n['answer'] for n in new_faq_responses if n['question'] == question), "").split("Quellen:")[-1].strip()
        comparison_result = compare_answers(question, existing_answer, new_answer).split("Zusammenfassung:")[-1].strip()

        #print(quellen)
        data.append({
            'Frage': question,
            'Bestehende Antwort': existing_answer,
            'Bestehende Quellen': existing_quellen,
            'Neue Antwort': new_answer,
            'Quellen (Neue)': quellen,
            'Vergleichsergebnis': comparison_result,
        })
    df = pd.DataFrame(data)
    return df
  
def parse_markdown(md_text):
    pattern = r'#\s\d+\.\s(.*?)\n\n(.*?)(?=\n#\s\d+|\Z)'
    matches = re.findall(pattern, md_text, re.DOTALL)
    result = []
    for question, content in matches:
        parts = content.strip().split("\n\nQuellen:\n")
        answer = content.strip()  # Remove leading and trailing whitespace
        answer = answer.split("Antwort:\n", 1)[-1].strip()  # Split and remove leading whitespace after "Antwort:"
        quellen = parts[1].replace("\n", ", ") if len(parts) > 1 else ""
        result.append({
            'question': question.strip(),
            'answer': answer,
            'quellen': quellen
        })
    return result

faq = parse_markdown(markdown_content)

OPENAI_API_KEY = os.getenv("RTR_OPENAI_API")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_new_answers(questions, max_retries=5, timeout=500):
    url = "https://rag.ki.rtr.at/llm-service/chat"
    headers = {
        'Authorization': os.getenv("RTR_BASIC_TOKEN"),
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json'
    }
    
    results = []
    
    for question in tqdm(questions, desc="Fetching new answers", unit="question"):
        payload = json.dumps({"prompt": question})
        attempt = 0
        
        while attempt < max_retries:
            try:
                response = requests.request(
                    "POST", url, headers=headers, data=payload, stream=True, timeout=timeout
                )
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data:"):
                            json_data = json.loads(decoded_line[5:].strip())
                            if json_data.get("type") == "assistant":
                                full_response += json_data.get("content", "")
                
                results.append({
                    "question": question,
                    "answer": full_response.encode('utf-8').decode('utf-8')
                })
                break  # Success, exit retry loop
            
            except requests.exceptions.ChunkedEncodingError as e:
                attempt += 1
                print(f"ChunkedEncodingError: {e}. Retrying {attempt}/{max_retries}...")
                
                if attempt == max_retries:
                    print(f"Failed to fetch answer for question: {question} after {max_retries} retries.")
                    results.append({"question": question, "answer": "Error retrieving response."})
    
    return results

questions = [item['question'] for item in faq]#[:1]
new_faq_responses = get_new_answers(questions)

def compare_answers(question, existing_answer, new_answer):
    prompt = f"""
    Untenstehend folgt eine Frage zum AI Act sowie zwei Antworten auf diese Frage. Deine Aufgabe ist es, eine Einschätzung zur Gleichwertigkeit der Antworten zu treffen.

    Frage: 
    {question}

    Antwort A:
    {existing_answer}

    Antwort B:
    {new_answer}

    Aufgabe: Obenstehend wurde eine rechtliche Frage zum AI Act zweimal beantwortet als "Antwort A" und "Antwort B".
    Vergleiche die beiden Texte und bestimme, wie stark die Antwort B von Antwort A abweicht. 
    Berücksichtige Fälle, wo Antwort B alle Aspekte von Antwort A abdeckt, aber darüber hinausgehend beantwortet. 
    Es handelt sich um ein Lehrbuch über das AI-Gesetz in der EU. 
    Schätze den Unterschied in verschiedenen Kategorien ein, wie zum Beispiel:
    * Kein Unterschied
    * Kein Unterschied, mehr Information
    * Mäßiger Unterschied
    * Mäßiger Unterschied, mehr Information
    * Großer Unterschied
    * Widersprüchliche Antwort

    Bitte fasse den Vergleich zwischen zwei Fragen unter dem Titel 'Zusammenfassung:' zusammen und starte immer mit der eingeschätzten Kategorie.
    Entferne bitte alle ** vor und nach den Wörtern. Zum Beispiel sollte **Frage:** zu Frage: geändert werden.
    Bitte halte diese Struktur in deiner Antwort ein: 'Frage:', 'Bestehende Antwort:', 'Quellen:', 'Neue Antwort:', 'Quellen:' und 'Zusammenfassung:'.
    Wenn Sie mehrere Quellen in „Neue Antwort“ finden, speichern Sie diese alle gemeinsam in einem einzigen gemeinsamen Abschnitt namens „Quellen“. 
    Alle Texte, die sich außerhalb von „Quellen“ in „Neue Antwort“ befinden, sollten in einen einzigen Textblock zusammengeführt werden
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein Experte für KI-Recht und -Regulierungen."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html', questions=[item['question'] for item in faq])#[:1])

#print(new_faq_responses)
# Store precomputed comparison results
comparison_cache = {}

def precompute_comparisons():
    global comparison_cache
    for item in tqdm(faq, desc="Fetching data for precompute_comparisons", unit="question"):
        new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == item['question']), "")
        comparison_result = compare_answers(item['question'], item['answer'], new_answer)
        comparison_cache[item['question']] = comparison_result

precompute_comparisons()

@app.route('/compare', methods=['GET'])
def compare():
    question_text = request.args.get('question')
    selected_item = next((item for item in faq if item['question'] == question_text), None)
    if selected_item:
        # Fetch the precomputed comparison result
        comparison_result = comparison_cache.get(selected_item['question'], "No comparison available")
        return render_template('compare.html', question=selected_item['question'], answer=selected_item['answer'], quellen=selected_item['quellen'], comparison=comparison_result)
    return "Question not found", 404

df = excel_data() 

@app.route('/download')
def download_excel():
    # Create an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Vergleiche')
    output.seek(0)

    # Send the file as a response
    return send_file(output, as_attachment=True, download_name='Vergleichsbericht.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
  
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Prevent the browser from opening twice due to the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
