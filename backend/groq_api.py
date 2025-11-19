import requests
import os

def groq_chat_api(prompt, dialog_history=[]):
    api_url = os.getenv("GROQ_URL")
    api_key = os.getenv("GROQ_API_KEY")


    # Make sure messages list is not empty and roles are correct
    messages = []
    for item in dialog_history:
        if item.get("role") == "user" and item.get("content"):
            messages.append({"role": "user", "content": item["content"]})
        elif item.get("role") == "bot" and item.get("content"):
            messages.append({"role": "assistant", "content": item["content"]})
    messages.append({"role": "user", "content": prompt})


    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


    payload = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",   # use the available model
    "messages": messages,
    "temperature": 0.85
}


    response = requests.post(api_url, headers=headers, json=payload)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("Groq API Error Response:")
        print(response.text)  # <-- See Groq's error details!
        raise e


    reply = response.json()["choices"][0]["message"]["content"]
    return reply