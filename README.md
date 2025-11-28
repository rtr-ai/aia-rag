# RTR AI-RAG Project Documentation  

This document provides an overview of the **RTR AI-RAG Project**, including its **Docker Compose setup**, **environment variables**, and **server setup** for **Ubuntu servers with NVIDIA graphics cards**.  

---
1. [Project Overview](#project-overview)
   * [Services Overview](#services-overview)
     + [1. Ollama](#1-ollama)
     + [2. Annotation Tool](#2-annotation-tool)
     + [3. LLM Service](#3-llm-service)
     + [4. User Interface](#4-user-interface)
2. [Docker Compose Setup](#docker-compose-setup)
   * [Building and starting the project](#building-and-starting-the-project)
   * [Services and Configuration](#services-and-configuration)
     + [1. Ollama](#1-ollama-1)
       - [Environment Variables](#environment-variables)
     + [2. Annotation Tool](#2-annotation-tool-1)
       - [Environment Variables](#environment-variables-1)
       - [Using the Annotation Tool](#using-the-annotation-tool)
     + [3. LLM Service](#3-llm-service-1)
       - [Chunk retrieval](#chunk-retrieval)
       - [Environment Variables](#environment-variables-2)
       - [Volumes Mapping](#volumes-mapping)
     + [4. User Interface](#4-user-interface-1)
       - [Environment Variables](#environment-variables-3)
3. [Server Setup for Ubuntu (NVIDIA GPU Support)](#server-setup-for-ubuntu-nvidia-gpu-support)
   * [1. Install Docker](#1-install-docker)
   * [2. Install NVIDIA Drivers](#2-install-nvidia-drivers)
   * [3. Install NVIDIA Container Toolkit](#3-install-nvidia-container-toolkit)
4. [Matomo Configuration](#matomo-configuration)
   * [Environment Configuration](#environment-configuration)
     + [Secure Token Management](#secure-token-management)
   * [Tracking Behavior](#tracking-behavior)
   * [Usage Example](#usage-example)
5. [Friendly Captcha Integration](#friendly-captcha-integration)
6. [Hardware requirements](#hardware-requirements)
---

## **Project Overview**  

The RTR AI-RAG project consists of multiple services working together to facilitate **text annotation, chunk indexing, and interaction with large language models (LLMs)**. These services are deployed using **Docker Compose** and rely on NVIDIA GPUs for accelerated computing.

### **Services Overview**  

#### 1. **Ollama**  
- A backend service responsible for managing **LLMs and embedding models**.  
- Downloads and caches models as required.  
- Manages parallel execution of models.  
- Uses NVIDIA GPUs for inference acceleration.  

#### 2. **Annotation Tool**  
- A web-based tool for manuelly segmenting text into chunks.  
- Allows users to **edit chunk texts, define interconnected (related) chunks**, and **add metadata**.  
- The processed chunks are stored and later indexed into a **vector store**.  

#### 3. **LLM Service**  
- A **FastAPI-based** service for indexing text chunks and prompting LLMs.  
- Uses **Server-Sent Events (SSE)** to stream responses.  
- On startup, **downloads all required LLMs and embedding models** via **Ollama API**, creates a **Vector Store Index**.  
- Supports **retrieval-augmented generation (RAG)**.  
- Measures approximate power consumption for indexing, retrieval and response generation.
- Optionally sends data to a remote **Matomo REST API**.

#### 4. **User Interface**  
- A web application that interacts with the **LLM service**.  
- Allows users to **query LLMs** and receive structured responses.  
- Users are able to see chunks used as sources for specific responses and power consumption for each step.

---

## **Docker Compose Setup**  

The whole project is available as a single **docker-compose.yml** file, consisting of four services.

### **Building and starting the project**  

To build the project, go to the project directory where ```docker-compose.yml``` is located and run
```console
docker compose build
```
This will download the latest version of Ollama and build the required images


For starting the project run
```
docker compose up -d
```
The `-d` option starts all the required containers in detached mode.

For deleting the downloaded LLM and embeddings models, simple remove all files in directory, which is mapped as a volume to ollama container (by default it's ```/app/ollama/data```).
This will force Ollama to download the latest versions on startup.



### **Services and Configuration**  

#### **1. Ollama**  
Responsible for LLM and embedding models interference.  

```yaml
  ollama:
    volumes:
      - /app/ollama/data:/root/.ollama
    container_name: ollama
    pull_policy: always
    tty: true
    restart: unless-stopped
    image: ollama/ollama:latest
    ports:
      - 127.0.0.1:11434:11434
    environment:
      - OLLAMA_KEEP_ALIVE=5m
      - OLLAMA_NUM_PARALLEL=1
    networks:
      - ollama-docker
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

##### **Environment Variables**  
For information about Ollama configuration please see [Ollama FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md).
- `OLLAMA_KEEP_ALIVE=5m` → Keeps models loaded for **5 minutes** before unloading.  
- `OLLAMA_NUM_PARALLEL=1` → Limits the number of **parallel running models**.  

---

#### **2. Annotation Tool**  
A React frontend application for **text chunking and metadata annotation**. 
Allows cutting the input into seperate chunks, name them, edit the text, add metadata, add keywords (suggestions can be generated by an LLM) and also define interconnected (related) chunks.

```yaml
  annotation-tool:
    build:
      context: ./annotation-tool 
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:3000"
    environment:
      - HOST=0.0.0.0
      - PUBLIC_URL=/annotation-tool
    command: npm start
    networks:
      - ollama-docker
    depends_on:
      - ollama
```

##### **Environment Variables**  
- `HOST=0.0.0.0` → Binds the service to all network interfaces.  
- `PUBLIC_URL=/annotation-tool` → Specifies the **public path** for accessing the tool.  

##### **Using the Annotation Tool**
In order to use the annotation tool open it under the corresponding URL in your browser. There are two options for inserting text in the very beginning: 
- Copy text and insert it into the text field. Click on "Weiter".
- Insert an existing index in the correct json-format (for this purpose use the button "Index importieren [JSON]")

On the next page you will see the inserted text. When marking text with the cursor the button "Textpassage schneiden" will become active and you can convert the marked text into a chunk. For each chunk you can add a title and a URL. Furthermore, you can add keywords and negative keywords (negative keywords not in use for this specific project). If useful, you can create the keywords using an LLM by clicking on the button "Keywords via LLM generieren" (Ollama instance has to be up and running for this feature to be used). In the field "Querverweise" you will find a drop down containing all other chunks of this index; choose from the list if you want to interconnect another chunk with this chunk. 

In the end, export the file by clicking on the button "Index herunterladen (JSON)". 

---

#### **3. LLM Service**  
Python FastAPI application which handles **index creation, LLM interactions, and prompt streaming**. 
Privileged mode is required for accessing the power usage data, which is used to measure approximate power consumuption for each step.


Upon application startup, the service first uses Ollama to pull all required LLMs and embedding models, then creates a vectore store index using the chunks cut beforehand and stored in the ./data/combined.json file.

To answer a user prompt, the LLM Service vectorizes the user prompt, calculates the **Cosine Similarity** between the user prompt and each of the indexed chunks, and uses the number of chunks defined in the **TOP_N_CHUNKS** variable to retrieve the top N chunks with the highest score. Afterwards, the text of these chunks, together with user and system prompt are sent to the LLM for generating a response.

##### **Chunk retrieval**
When retrieving chunks for a specific user prompt, the LLM Service checks the following:
- **Interconnected chunks** For each chunk, the interconnected (related) chunks are added.
- **Context Window** Chunks that would cause the LLM's context window to be exceeded, are skipped. This check starts at the top of the list (ranking based on Cosine Similarity) and counts till the last chunk that will still fit into the context window. 
- **Duplicates** Duplicated chunks are skipped.

```yaml
  llm-service:
    privileged: true
    build:
      context: ./llm-service
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "127.0.0.1:8085:8000"
    environment:
      - HOST=0.0.0.0
      - OLLAMA_HOST=ollama
      - EMBEDDING_MODELS=snowflake-arctic-embed2
      - LLM_MODELS=mistral-small,mistral-nemo,granite3.1-dense,phi4,llama3.1:8b-instruct-fp16,mixtral:8x7b-instruct-v0.1-q2_K
      - ALLOWED_ORIGINS=http://rag.ki.rtr.at,https://rag.ki.rtr.at
      - TEMPERATURE=0.1
      - CONTEXT_WINDOW=20000
      - PROMPT_BUFFER=4000
      - ROOT_PATH=/llm-service
      - TOP_N_CHUNKS=25
      - ORDER_CHUNKS_FROM_SOURCE=true
    volumes:
      - ./data/combined.json:/app/data/chunks.json
      - /sys/class/powercap:/sys/class/powercap:ro
      - /proc/stat:/proc/stat:ro
      - /proc/cpuinfo:/proc/cpuinfo:ro
      - /sys/devices/system/cpu:/sys/devices/system/cpu:ro
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-modeset:/dev/nvidia-modeset
      - /dev/nvidia-uvm:/dev/nvidia-uvm
    networks:
      - ollama-docker
    depends_on:
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

##### **Environment Variables**  
- `EMBEDDING_MODELS` → Comma-separated list of embedding models. First model is used by default. The available embedding models are listed [here](https://ollama.com/search?c=embedding)
- `LLM_MODELS` → Comma-separated list of LLM models.  First model is used by default. The available models are listed [here](https://ollama.com/search)
- `ALLOWED_ORIGINS` → Defines allowed CORS origins.  
- `TEMPERATURE=0.1` → Controls randomness in LLM responses.  
- `CONTEXT_WINDOW=20000` → Defines the LLM **context window size (tokens)**.  
- `PROMPT_BUFFER=4000` → Reserves tokens for the **user prompt** during prompt construction.  
- `TOP_N_CHUNKS=25` → Maximum number of text chunks used for RAG. The number corresponds to top level chunks; interconnected (related) chunks are not counted.  
- `ORDER_CHUNKS_FROM_SOURCE=true` → Maintains original order of text chunks instead of sorting them by score calculated using cosine similarity. The choice of chunks is still based on cosine similarity, the order only affects how the chosen chunks are inserted into the prompt.  


##### **Volumes Mapping**  
- `./data/combined.json:/app/data/chunks.json` → Output from the annotation-tool, which should be stored in the file **combined.json** which is used for index creation.
- `./app/logs:/app/logs/` → Logs from llm-service.
- `The rest of volumes` → Used for power consumption metrics.


---

#### **4. User Interface**  
 **Angular web-based frontend** for interacting with the LLM service.  Allows users to prompt the LLM, see responses as they are streamed, but also the sources used for generating the response and approximate power usage.

```yaml
  user-interface:
    build:
      context: ./user-interface
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "127.0.0.1:8090:80"
    environment:
      - HOST=0.0.0.0
      - LLM_ENDPOINT=https://rag.ki.rtr.at/llm-service
    networks:
      - ollama-docker
    depends_on:
      - llm-service
```

##### **Environment Variables**  
- `LLM_ENDPOINT` → Specifies the **LLM service URL** for the UI. Currently, a relative path "/llm-service" is used. 

---

## **Server Setup for Ubuntu (NVIDIA GPU Support)**  

### **1. Install Docker**  
Follow the **official Docker installation guide**:  

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### **2. Install NVIDIA Drivers**  

```bash
sudo apt update
sudo ubuntu-drivers devices  # Lists available drivers
sudo apt install nvidia-driver-535  # Install the recommended driver
sudo reboot
nvidia-smi  # Verify installation
```

### **3. Install NVIDIA Container Toolkit**  

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---



## Matomo Configuration

Matomo can be configured using a .env environment variable.

### Environment Configuration

#### Secure Token Management


1. Create an `.env`using matomo.env.example as reference file:
```
MATOMO_ENDPOINT=https://your-matomo-instance.com/matomo.php
MATOMO_TOKEN=your_tracking_token
MATOMO_SITE_ID=your_site_id
```

### Tracking Behavior
- If Matomo configuration is missing, tracking silently skips.
- Supports tracking events with optional JSON  or string metadata.
- Uses singleton service for consistent tracking across application.
- The event data is sent using the Matomo REST API.

### Usage Example
```python
from services.matomo_tracking_service import matomo_service

# Track event with optional data
matomo_service.track_event(
    action='user_prompt', 
    data={'May I use AI in my organization?'}
)
```
 

## Friendly Captcha Integration  

This describes how **Friendly Captcha** is integrated into the system, how it can be disabled and how it interacts with the **LLM service** and **user interface**.

### Environment Variables  

The following environment variables must be set for Friendly Captcha to function correctly:  

| Variable Name                 | Description |
|--------------------------------|-------------|
| **FRIENDLY_CAPTCHA_SITEKEY**   | The Sitekey for Friendly Captcha, used in the user interface. If this variable is not set, captcha verification is disabled in the **user interface**. |
| **CAPTCHA_API_KEY**            | The API key for verifying captchas in the LLM service. If this variable is not set or empty, captcha verification is disabled in the **LLM service**. |
| **CAPTCHA_OVERRIDE_SECRET**    | A hardcoded captcha response for automated tests. If this key is provided in requests, captcha verification is skipped. |

### Captcha Verification Flow  

1. The **user interface** includes a Friendly Captcha challenge using the **FRIENDLY_CAPTCHA_SITEKEY**.  
2. Once solved, the **captcha response** is sent to the **LLM service** in field `frc_captcha_solution` along with the request payload.  
3. The **LLM service** validates the captcha by sending a request to Friendly Cpathca API using the **CAPTCHA_API_KEY**:  
   - If **CAPTCHA_API_KEY** is not set or empty, captcha verification is **disabled** in the LLM service.  
   - If the value of `frc_captcha_solution` exactly matches the value of  **CAPTCHA_OVERRIDE_SECRET** environment variable, captcha verification is skipped.

### Automated Testing  

For automated tests, the `frc_captcha_solution` field should contain the value of **CAPTCHA_OVERRIDE_SECRET**. If this matches the configured environment variable, captcha verification is bypassed.  

### Disabling Captcha  

- If **FRIENDLY_CAPTCHA_SITEKEY** is **not configured**, captcha verification is **disabled in the user interface**.  
- If **CAPTCHA_API_KEY** is **not configured or empty**, captcha verification is **disabled in the LLM service**.  

---


## Hardware requirements

The current configuration has been been tested on a server running **Ubuntu 24.04.2 LTS** with an **NVIDIA RTX 4000 SFF Ada** graphics card with **20GB VRAM**, **Intel® Core™ i5-13500** processor and **64 GB DDR4** RAM. This allows Ollama to run the mistral-small LLM with a context window of 25.000 tokens (with a decent token per second speed for a context window of 15.000 tokens).
Generally, the actual hardware requirements will depend on the chosen model and context window.


The software can also run on a consumer graded PC or Laptop if the VRAM requirements are met, but in such case, the power usage metrics accuracy will depend on the hardware used. In some scenarios, the power usage should be disabled, if device compatibility issue arise. 

