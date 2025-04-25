import json
import os
import time
from time import sleep

import requests
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

MAX_RETRIES = 10
TIMEOUT = 90
MAX_TIME = 240

URL = "https://rag.ki.rtr.at/llm-service/chat"
HEADERS = {
    'Authorization': os.getenv("HTTP_AUTHORIZATION"),
    'Accept': 'text/event-stream',
    'Content-Type': 'application/json'
}

captcha_override = os.getenv('CAPTCHA_OVERRIDE_SECRET')
begin = time.time()

results = []

for question in ["Ist der AIA ausserhalb der EU anwendbar?"]:
    obj = {"prompt": question}
    if captcha_override is not None and len(captcha_override) > 10:
        obj["frc_captcha_solution"] = captcha_override
    payload = json.dumps(obj)
    attempt = 0
    while attempt < MAX_RETRIES and (begin + MAX_TIME > time.time()):
        try:
            response = requests.request(
                "POST", URL, headers=HEADERS, data=payload, stream=True, timeout=TIMEOUT
            )
            if response.status_code != 200:
                logging.info("return code " + str(response.status_code))
                raise requests.exceptions.ConnectionError

            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data:"):
                        json_data = json.loads(decoded_line[5:].strip())
                        logging.info("got " + json_data.get("type"))
                        if json_data.get("type") == "assistant":
                            exit(0)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            attempt += 1
            logging.info("retrying " + str(attempt))
            sleep(10)

logging.warning("maximum retries reached, exiting")
exit(1)