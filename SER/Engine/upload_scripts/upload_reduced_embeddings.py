import os
import requests

"""
Simple script that lets you upload the desired reduced embeddings to the already full database 
"""

if __name__ == "__main__":
    url = "http://127.0.0.1:8000/generate_low_dim_embeddings/"
    response = requests.post(url)
    print(response)