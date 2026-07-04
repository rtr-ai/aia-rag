import json
import os
import time
from time import sleep

import requests
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

MAX_RETRIES = 10
TIMEOUT = 1300
MAX_TIME = 1480

URL = "https://rag.ki.rtr.at/llm-service/chat"
HEADERS = {
    'Authorization': os.getenv("HTTP_AUTHORIZATION"),
    'Accept': 'text/event-stream',
    'Content-Type': 'application/json'
}

captcha_override = os.getenv('CAPTCHA_OVERRIDE_SECRET')

# Validate environment variables
logger.info(f"Environment check:")
logger.info(f"  HTTP_AUTHORIZATION set: {'Yes' if os.getenv('HTTP_AUTHORIZATION') else 'NO - MISSING'}")
logger.info(f"  CAPTCHA_OVERRIDE_SECRET set: {'Yes' if os.getenv('CAPTCHA_OVERRIDE_SECRET') else 'Not set (optional)'}")
logger.info(f"Target URL: {URL}")
logger.info(f"Max retries: {MAX_RETRIES}, Timeout per request: {TIMEOUT}s, Max total time: {MAX_TIME}s")

begin = time.time()

results = []

for question in ["Ist der AIA ausserhalb der EU anwendbar?"]:
    logger.info(f"Starting validation with question: '{question}'")
    obj = {"prompt": question}
    if captcha_override is not None and len(captcha_override) > 10:
        obj["frc_captcha_solution"] = captcha_override
        logger.debug("Captcha override included in request")
    payload = json.dumps(obj)
    logger.debug(f"Request payload: {payload}")
    
    attempt = 0
    while attempt < MAX_RETRIES and (begin + MAX_TIME > time.time()):
        elapsed = time.time() - begin
        logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} (elapsed: {elapsed:.1f}s)")
        try:
            logger.debug(f"Sending POST request to {URL}")
            response = requests.request(
                "POST", URL, headers=HEADERS, data=payload, stream=True, timeout=TIMEOUT
            )
            
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"HTTP error - Expected 200, got {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                logger.debug(f"Response text: {response.text[:500]}")  # Log first 500 chars
                raise requests.exceptions.ConnectionError(f"HTTP {response.status_code}")

            logger.info("Successfully received 200 response, processing stream...")
            full_response = ""
            line_count = 0
            
            for line in response.iter_lines():
                line_count += 1
                if line:
                    try:
                        decoded_line = line.decode('utf-8').strip()
                        logger.debug(f"Line {line_count}: {decoded_line[:100]}")  # Log first 100 chars
                        
                        if decoded_line.startswith("data:"):
                            json_data = json.loads(decoded_line[5:].strip())
                            msg_type = json_data.get("type")
                            logger.info(f"Received message type: {msg_type}")
                            
                            if msg_type == "assistant":
                                logger.info("SUCCESS: Received 'assistant' type message. Validation passed!")
                                exit(0)
                    except json.JSONDecodeError as je:
                        logger.warning(f"Failed to parse JSON on line {line_count}: {je}")
                        logger.debug(f"Raw line: {line}")
                        continue
                    except Exception as ex:
                        logger.warning(f"Error processing line {line_count}: {ex}")
                        continue
            
            logger.warning(f"Stream ended after {line_count} lines without receiving 'assistant' message")
            attempt += 1
            if attempt < MAX_RETRIES:
                logger.info(f"Will retry after 10 seconds...")
                sleep(10)
            
        except requests.exceptions.Timeout:
            attempt += 1
            logger.error(f"Request timeout (>{TIMEOUT}s)")
            logger.info(f"Retry {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                sleep(10)
                
        except requests.exceptions.ConnectionError as e:
            attempt += 1
            logger.error(f"Connection error: {e}")
            logger.info(f"Retry {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                sleep(10)
                
        except Exception as e:
            attempt += 1
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            logger.info(f"Retry {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                sleep(10)
    
    elapsed = time.time() - begin
    logger.error(f"Failed after {attempt} attempts and {elapsed:.1f}s")

logger.error("VALIDATION FAILED: Maximum retries reached, exiting")
exit(1)
