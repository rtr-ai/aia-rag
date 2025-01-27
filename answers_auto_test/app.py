from flask import Flask, render_template, request
import json
import os
import re
from tqdm import tqdm
import requests
from openai import OpenAI
from io import BytesIO
import webbrowser
from threading import Timer
import os

app = Flask(__name__)

# Read the markdown file
with open("data/fragen-qa.md", 'r', encoding='utf-8') as file: 
    markdown_content = file.read()

def parse_markdown(md_text):
    pattern = r'#\s\d+\.\s(.*?)\n\n(.*?)(?=\n#\s\d+|\Z)'
    matches = re.findall(pattern, md_text, re.DOTALL)
    result = []
    for question, content in matches:
        parts = content.strip().split("\n\nQuellen:\n")
        answer = content.strip().split("\n\nAntwort:\n")
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

def get_new_answers(questions):
    url = "https://rag.ki.rtr.at/llm-service/chat"
    headers = {
        'Authorization': os.getenv("RTR_BASIC_TOKEN"),
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json'
    }
    results = []
    for question in tqdm(questions, desc="Fetching new answers", unit="question"):
        payload = json.dumps({"prompt": question})
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data:"):
                        json_data = json.loads(decoded_line[5:].strip())
                        if json_data.get("type") == "assistant":
                            full_response += json_data.get("content", "")
                except Exception as e:
                    print(f"Error processing line: {e}")
        results.append({
            "question": question,
            "answer": full_response.encode('utf-8').decode('utf-8')
        })
    return results

questions = [item['question'] for item in faq][:6]
new_faq_responses = get_new_answers(questions)

def compare_answers(question, existing_answer, new_answer):
    prompt = f"""
    Frage: {question}
    
    Bestehende Antwort:
    {existing_answer}
    
    Neue Antwort:
    {new_answer}
    
    Aufgabe: Untenstehend wird eine rechtliche Frage zum AI Act zweimal beantwortet.
    Stimmen die beiden Antworten überein? Antworte entweder mit "ja", oder mit "nein" und einer Liste von Bullet Points mit Widersprüchen. Bitte schreibe neben der Frage auch beide Antworten für alle Fragen.
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein Experte für KI-Recht und -Regulierungen."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html', questions=[item['question'] for item in faq][:6])

@app.route('/compare', methods=['GET'])
def compare():
    question_text = request.args.get('question')
    selected_item = next((item for item in faq if item['question'] == question_text), None)
    if selected_item:
        new_answer = next((n['answer'] for n in new_faq_responses if n['question'] == question_text), "")
        comparison_result = compare_answers(selected_item['question'], selected_item['answer'], new_answer)
        return render_template('compare.html', question=selected_item['question'], answer=selected_item['answer'], quellen=selected_item['quellen'], comparison=comparison_result)
    return "Question not found", 404

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Prevent the browser from opening twice due to the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)