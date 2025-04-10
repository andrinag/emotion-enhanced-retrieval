import requests

async def send_query_to_llama(query: str):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama2",
        "prompt": query,
        "stream": False
    })

    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise Exception(f"Failed to get response from LLaMA. Status code: {response.status_code}")


if __name__ == "__main__":
    send_query_to_llama("tell me a joke")