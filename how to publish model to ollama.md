Yes. You can publish it on **ollama.com**, similar to Hugging Face. Ollama requires the converted GGUF model first.

1. Create an account at [ollama.com](https://ollama.com).

2. Register the GGUF in an Ollama installation:

```bash
ollama create snowflake-arctic-embed2-ai-act -f Modelfile
```

3. Sign in:

```bash
ollama signin
```

4. Add your Ollama username to the model name:

```bash
ollama cp snowflake-arctic-embed2-ai-act YOUR_USERNAME/snowflake-arctic-embed2-ai-act
```

5. Publish it:

```bash
ollama push YOUR_USERNAME/snowflake-arctic-embed2-ai-act
```

Then your colleague only needs:

```bash
docker exec ollama-embedding ollama pull YOUR_USERNAME/snowflake-arctic-embed2-ai-act
```

To keep the current name in `docker-compose.dev.yml`:

```bash
docker exec ollama-embedding ollama cp YOUR_USERNAME/snowflake-arctic-embed2-ai-act snowflake-arctic-embed2-ai-act
```

This removes the manual GGUF file transfer to the server. Ollama documents this publishing workflow in its [official importing and sharing guide](https://docs.ollama.com/import).