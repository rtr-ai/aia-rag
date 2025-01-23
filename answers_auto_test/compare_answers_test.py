from openai import OpenAI
import requests
import json
import os
from tqdm import tqdm

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("RTR_OPENAI_API")
client = OpenAI(api_key=OPENAI_API_KEY)

# Existing FAQ data
faq = [
    {
        'question': 'Ich möchte meinen Mitarbeitern den Einsatz von ChatGPT ermöglichen. Welche Rolle im AI Act nehme ich als Unternehmen ein?',
        'answer': '''ChatGPT wird als KI-System im Sinne des AI Acts einzustufen sein [1].  
Da Sie als Unternehmen das KI-System nicht selbst entwickeln, sondern dieses im Rahmen der beruflichen Tätigkeit in eigener Verantwortung einsetzen, wird Ihr Unternehmen die Rolle eines Betreibers einnehmen [2], [3], [4]. 
[1] Art 3: Z 1 Definition KI-System
[2] Art 3: Z3-Z11, Z68 Akteure
[3] KI-Servicestelle: Akteure, KI-Wertschöpfungskette
[4] KI-Servicestelle: Akteure, Betreiber'''
    },
    {
        'question': 'Ich möchte Werbeaussendungen mit einem Large Language Model automatisch generieren und individuell angepasst an Kunden versenden. Muss ich das offenlegen?',
        'answer': '''Das Einsatz von Large Language Models bei der automatischen Generierung von Werbeaussendungen unterliegt nicht den Transparenzpflichten des AI-Acts, weil kein öffentliches Interesse gegeben ist, [1],[2], [3]
[1] Art 50 Abs 1 AIA
[2] Art 50 Abs 4 UA 2 AIA
[3] KI-Servicestelle: Transparenzpflichten, Kennzeichnungspflicht
[4] KI-Servicestelle: Transparenzpflichten, Generierung synthetischer Inhalte
[5] KI-Servicestelle: Transparenzpflichten, Textgenerierung'''
    }
]

# Function to fetch new answers from the API
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

# Fetch new answers from API
questions = [item['question'] for item in faq]
new_faq_responses = get_new_answers(questions)

# Function to compare answers using OpenAI
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

# Compare and print results
for old, new in tqdm(zip(faq, new_faq_responses), total=len(faq), desc="Comparing answers", unit="comparison"):
    comparison_result = compare_answers(old['question'], old['answer'], new['answer'])
    print(f"Vergleich für Frage: {old['question']}\n")
    print(comparison_result)
    print("=" * 80)

