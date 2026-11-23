import ollama

reponse = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": "Réponds en une phrase : qu'est-ce que Python ?"}],
)
print(reponse["message"]["content"])