from openai import OpenAI
import requests
import json
import os
import re
from tqdm import tqdm

# Read the markdown file
with open("L:/System/Downloads/fragen-qa.md", 'r', encoding='utf-8') as file: 
    markdown_content = file.read()

def parse_markdown(md_text):
    # Regex pattern to capture questions, answers, and sources (Quellen)
    pattern = r'#\s\d+\.\s(.*?)\n\n(.*?)(?=\n#\s\d+|\Z)'
    matches = re.findall(pattern, md_text, re.DOTALL)

    result = []
    for question, content in matches:
        # Separate answer from sources
        parts = content.strip().split("\n\nQuellen:\n")
        answer = content.strip().split("\n\nAntwort:\n")
        quellen = parts[1].replace("\n", ", ") if len(parts) > 1 else ""

        result.append({
            'question': question.strip(),
            'answer': answer,
            'quellen': quellen
        })
    
    return result

# Parse markdown content
parsed_data = parse_markdown(markdown_content)
faq = parsed_data

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("RTR_OPENAI_API")
client = OpenAI(api_key=OPENAI_API_KEY)

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

