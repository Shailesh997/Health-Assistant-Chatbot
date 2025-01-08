import ollama

model_name = "mistral:latest"

def generate_response(input_text):
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": input_text}])
    return response['message']['content']
