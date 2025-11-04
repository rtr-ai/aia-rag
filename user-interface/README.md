# KI Servicestelle AI-Act RAG-Chat

## Known issues

Docker does not correctly replace the environment variable LLM_ENDPOINT defined in docker-compose.yml.
Temporarily, the environment variable default value is set to the actual endpoint in `/src/environments/environment.ts`

## Starting the User Interface

Please start the user interface using `docker-compose.yml` provided
