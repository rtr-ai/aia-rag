import requests
import json
import time
from tqdm import tqdm

# URL of the service
url = "https://rag.ki.rtr.at/llm-service/chat"

# Load the questions from the JSON file NORMAL
with open("auto_test/true_results.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract keys as a list
faq = list(data.keys())

# Container for the final JSON data
output_data = {}

for q in tqdm(faq, desc="Processing questions", unit="question"):
    # Data payload
    payload = {"prompt": q}

    # Headers
    headers = {
        "Accept": "text/event-stream",
        "Authorization": os.getenv("RTR_BASIC_TOKEN"),
        "Content-Type": "application/json"
    }

    # Send the POST request
    s = requests.Session()
    response = s.post(url, json=payload, headers=headers, stream=True)

    # Check for a successful response
    if response.status_code == 200:
        events = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_data = decoded_line[6:]
                    event_data = json.loads(json_data)
                    events.append(event_data)
                    if len(events) >= 3:
                        break
            time.sleep(1)

        # Process the second event
        main_event = events[2]
        content = json.loads(main_event["content"])

        # Prepare data for JSON in the new structure
        articles = []
        relevant_chunks = []

        for item in content:
            if item.get("skip") == True and item.get("skip_reason") == "context_window":
                continue  # Skip this object

            main_title = item.get('title', '<No Title>')
            articles.append(main_title)
            for chunk in item.get('relevantChunks', []):
                chunk_title = chunk.get('title', '<No Title>')
                relevant_chunks.append(chunk_title)

        # Assign to the question key if relevant data exists
        if articles or relevant_chunks:
            output_data[q] = [
                {"articles": articles},
                {"relevant": relevant_chunks}
            ]

    else:
        print('Request failed with status code:', response.status_code)

# Write the collected data to a JSON file NORAML
with open("auto_test/evaluated_results.json", "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

# Write the collected data to a JSON file GEHEIM
#with open("auto_test/evaluated_results_geheim.json", "w", encoding="utf-8") as json_file:
    #json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print("JSON file 'evaluated_results_geheim.json' created successfully.")
