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

    category_counts  = {
    "1: Kein Unterschied": 0,
    "2: Überschießende Antwort": 0,
    "3: Überschießende Antwort": 0,
    "4: Widersprüchliche Antwort": 0
    }
    for item in tqdm(faq, desc="Fetching data for excel report", unit="question"): # [:5]
        question = item['question']
        existing_answer = item['answer'].split('Quellen:')[0].strip()
        #existing_quellen = item['answer'].split('Quellen:')[1].strip()
        new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == question), "").split("Quellen:")[0].strip()
        #quellen = next((n['answer'] for n in new_faq_responses if n['question'] == question), "").split("Quellen:")[-1].strip()
        #comparison_result = compare_answers(question, existing_answer, new_answer).split("Zusammenfassung:")[1].strip()
        comparison_result = comparison_cache.get(question, "No comparison available").split("Zusammenfassung:")[-1].strip()

        # Count the occurrences of each category
        for category in category_counts.keys():
            if category in comparison_result:
                category_counts[category] += 1
                break  # Ensure only one category is counted per question

        data.append({
            'Frage': question,
            'Bestehende Antwort': existing_answer,
            #'Bestehende Quellen': existing_quellen,
            'Neue Antwort': new_answer,
            #'Quellen (Neue)': quellen,
            'Vergleichsergebnis': comparison_result,
        })
    df = pd.DataFrame(data)

    # Append summary rows directly to df
    summary_rows = [{"Frage": f"Anzahl {category}", "Vergleichsergebnis": count} for category, count in category_counts.items()]
    summary_df = pd.DataFrame(summary_rows)

    df = pd.concat([df, summary_df], ignore_index=True)

    return df
  
def parse_markdown(md_text):
    pattern = r'#\s\d+\.\s(.*?)\n\n(.*?)(?=\n#\s\d+|\Z)'
    matches = re.findall(pattern, md_text, re.DOTALL)
    result = []
    for question, content in matches:
        parts = content.strip().split("\n\nQuellen:\n")
        answer = content.strip()  # Remove leading and trailing whitespace
        answer = answer.split("Antwort:\n", 1)[-1].split("Quellen:\n", 1)[0].strip()  # Split and remove leading whitespace after "Antwort:"
        #quellen = parts[1].replace("\n", ", ") if len(parts) > 1 else ""
        result.append({
            'question': question.strip(),
            'answer': answer,
            #'quellen': quellen
        })
    return result

faq = parse_markdown(markdown_content)

OPENAI_API_KEY = os.getenv("RTR_OPENAI_API")
client = OpenAI(api_key=OPENAI_API_KEY)

def clean_answer(answer):
    import re
    """Removes the 'Quellen:' section and everything after it from the answer."""
    return re.split(r'\nQuellen:', answer, maxsplit=1)[0]

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

                cleaned_response = clean_answer(full_response)

                results.append({
                    "question": question,
                    "answer": cleaned_response.encode('utf-8').decode('utf-8')
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

def enforce_response_structure(response):
    required_sections = ["Frage:", "Bestehende Antwort:", "Neue Antwort:", "Zusammenfassung:"]
    for section in required_sections:
        if section not in response:
            response += f"\n\n{section} (keine Angabe)"
    return response

def compare_answers(question, existing_answer, new_answer):
    prompt = f"""
    Untenstehend folgt eine Frage zum AI Act sowie zwei Antworten auf diese Frage. Deine Aufgabe ist es, eine Einschätzung zur Gleichwertigkeit aus juristischer Sicht der Antworten zu treffen.
    Frage: 
    {question}

    Antwort A:
    {existing_answer}

    Antwort B:
    {new_answer}

    Aufgabe zur Beurteilung: Obenstehend wurde eine rechtliche Frage zum AI Act zweimal beantwortet als "Antwort A" und "Antwort B".
    Vergleiche die beiden Texte und bestimme, wie stark die Antwort B von Antwort A abweicht. 
    Berücksichtige Fälle, wo Antwort B alle Aspekte von Antwort A abdeckt, aber darüber hinausgehend beantwortet. 
    Es handelt sich um ein juristisches Lehrbuch über das AI-Gesetz in der EU. Die Fragen werden dort abgedruckt. Relevant sind juristische Unterschiede. Relevant sind, ob andere rechtliche Schlüsse und Subsumptionen gezogen werden. Nicht relevant sind reine Änderungen in der Formulierung. 

    Wie unterscheiden sich die beiden Antworten? Bitte halte dich strikt an die folgende Struktur und bewerte in:

    1: Kein Unterschied: Beide Antworten enthalten nur dieselbe Information. Beide Antworten kommen zu den selben juristischen Schlüssen.
    2: Überschießende Antwort, kein Widerspruch: Beide Antworten enthalten dieselben juristischen Schlüsse. Eine Antwort behandelt mehrere juristische Aspekte als die andere Antwort.
    3: Überschießende Antwort, Widerspruch: Beide Antworten enthalten die selben juristischen Schlüsse. Eine Antwort behandelt noch mehr Aspekte, die aber potentiell einen Widerspruch auslösen. Das ist insbesondere dann der Fall, wenn unterschiedliche Schlüsse zur Kategorisierung eines KI-Systems getroffen werden.
    4: Widersprüchliche Antwort: Beide Antworten enthalten juristische Schlüsse, die zueinander im Gegensatz stehen.

    Zusätzlich bewerte den Stil der Antwort B:
    gut: Professionelle Antwort, sie schafft juristische Klarheit.
    schlecht: Unprofessionelle Antwort, sie ist im juristischen Kontext und für ein Lehrbuch zum AI Act nicht professionell. Etwa wird der Leser gedutzt, oder es wird auf weitere Gespräche und Hilfestellungen verwiesen. Oder es wird auf die DSGVO oder andere Gesetze verwiesen, die nicht der AI Act sind.

    Beginne deine Antwort mit der Schulnote, direkt mit der Ziffer. Erläutere dann in zwei Sätzen, wie sich der Schluss zusammensetzt. Danach bewerte den Stil der Antwort mit "gut" oder "schlecht" und erläutere deine Entscheidung in einem Satz. Beginne mit "gut" oder "schlecht".

    Bitte fasse den Vergleich zwischen zwei Fragen unter dem Titel 'Zusammenfassung:' zusammen.
    Entferne bitte alle ** vor und nach den Wörtern. Zum Beispiel sollte **Frage:** zu Frage: geändert werden.
    Bitte halte dich strikt an die folgende Struktur und gib IMMER alle Abschnitte aus, auch wenn sie leer sind:
    Frage:
    Bestehende Antwort:
    Neue Antwort:
    Zusammenfassung:
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein Experte für KI-Recht und -Regulierungen."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return enforce_response_structure(completion.choices[0].message.content)

@app.route('/')
def index():
    return render_template('index.html', questions=[item['question'] for item in faq], comparison_counts=comparison_counts) #[:5]

#print(new_faq_responses)
# Store precomputed comparison results
comparison_cache = {}

comparison_counts = {
    "1: Kein Unterschied": 0,
    "2: Überschießende Antwort": 0,
    "3: Überschießende Antwort": 0,
    "4: Widersprüchliche Antwort": 0
}

def precompute_comparisons():
    global comparison_cache, comparison_counts
    for item in tqdm(faq, desc="Fetching data for precompute_comparisons", unit="question"): # [:5]
        new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == item['question']), "")
        comparison_result = compare_answers(item['question'], item['answer'], new_answer)

        # Store result in cache
        comparison_cache[item['question']] = comparison_result

        # Extract the first category from the comparison result
        for category in comparison_counts.keys():
            if category in comparison_result:
                comparison_counts[category] += 1
                break

#def precompute_comparisons():
    #global comparison_cache
    #for item in tqdm(faq, desc="Fetching data for precompute_comparisons", unit="question"): # [:5]
        #new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == item['question']), "")
        #comparison_result = compare_answers(item['question'], item['answer'], new_answer)
        #comparison_cache[item['question']] = comparison_result

precompute_comparisons()

@app.route('/compare', methods=['GET'])
def compare():
    question_text = request.args.get('question')
    selected_item = next((item for item in faq if item['question'] == question_text), None)
    if selected_item:
        # Fetch the precomputed comparison result
        comparison_result = comparison_cache.get(selected_item['question'], "No comparison available")
        #print(f"Comparison content: {comparison_result}")
        return render_template('compare.html', question=selected_item['question'], answer=selected_item['answer'],  comparison=comparison_result) #quellen=selected_item['quellen'],
    return "Question not found", 404

df = excel_data() 

@app.route('/download')
def download_excel():
    #df = excel_data()  # Get both dataframes
    # Create an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Vergleiche')
        #summary_df.to_excel(writer, index=False, sheet_name='Übersicht')
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
