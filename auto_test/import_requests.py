import requests
import json
import time
import subprocess
import os
from tqdm import tqdm

# URL of the service
url = "https://rag.ki.rtr.at/llm-service/chat"

# fetch the questions rom true_ersults.json file
with open("true_results.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract keys as a list
faq = list(data.keys())

# Container for the final JSON data
output_data = {}

for q in tqdm(faq, desc="Processing questions", unit="question"):
    # Data payload
    payload = {
        "prompt": q
    }

    # Headers, based on your request
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
                    # Strip 'data: ' part from the line
                    json_data = decoded_line[6:]
                    event_data = json.loads(json_data)
                    events.append(event_data)
                    
                    # Check to see if we have enough events
                    if len(events) >= 3:
                        break

            # Introduce a delay, as subsequent responses may take time
            time.sleep(1)

        # Process the second event
        main_event = events[2]
        content = json.loads(main_event["content"])

        # Prepare data for JSON in the new structure
        articles = []
        relevant_chunks = []

        for item in content:
            main_title = item.get('title', '<No Title>')
            articles.append(main_title)
            for chunk in item.get('relevantChunks', []):
                chunk_title = chunk.get('title', '<No Title>')
                relevant_chunks.append(chunk_title)

        # Assign to the question key
        output_data[q] = [
            {"articles": articles},
            {"relevant": relevant_chunks}
        ]
        
        
    else:
        print('Request failed with status code:', response.status_code)

# Write the collected data to a JSON file
with open("auto_test/evaluated_results.json", "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print("JSON file 'evaluated_results.json' created successfully.")

def git_push():
    try:
        # Add the updated file to the staging area
        subprocess.run(["git", "add", "auto_test/evaluated_results.json"], check=True)

        # Commit the changes with a message
        subprocess.run(["git", "commit", "-m", "Update FAQ JSON file"], check=True)

        # Push the changes to the remote repository
        subprocess.run(["git", "push"], check=True)

        print("File successfully pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Git operations: {e}")

# Call the function to push changes
git_push()
