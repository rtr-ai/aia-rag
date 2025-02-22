import json
import os
from time import sleep

import requests

MAX_RETRIES = 10
TIMEOUT = 200

URL = "https://rag.ki.rtr.at/llm-service/chat"
HEADERS = {
    'Authorization': os.getenv("RTR_BASIC_TOKEN"),
    'Accept': 'text/event-stream',
    'Content-Type': 'application/json'
}

results = []

for question in ["Ist der AIA ausserhalb der EU anwendbar?"]:
    payload = json.dumps({"prompt": question})
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            response = requests.request(
                "POST", URL, headers=HEADERS, data=payload, stream=True, timeout=TIMEOUT
            )
            if response.status_code != 200:
                raise requests.exceptions.ConnectionError

            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data:"):
                        json_data = json.loads(decoded_line[5:].strip())
                        print("got " + json_data.get("type"))
                        if json_data.get("type") == "assistant":
                            exit(0)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            attempt += 1
            print("retrying " + str(attempt))
            if attempt > MAX_RETRIES:
                exit(1)
            sleep(10)