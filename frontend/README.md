# KiServicestelleAiabot

## Know issues

Docker does not correctly replace the environment variable LLM_ENDPOINT defined in docker-compose.yml.
Temporarity, the environment variable default value is set to the actual endpoint in `/src/environments/environment.ts`

## Starting the Frontend

Please start the frontend using `docker-compose.yml` provided