import requests
from utils.parser import get_thinking_content

url = "http://127.0.0.1:8000/generate"
prompt = "Hello, who are you?"
payload = {"prompt": prompt, "args": {"max_new_tokens": 500}}

response = requests.post(url, json=payload)
print(
    f"thinking: {get_thinking_content(response.json()[0])[0]}\nresponse: {get_thinking_content(response.json()[0])[1]}"
)
