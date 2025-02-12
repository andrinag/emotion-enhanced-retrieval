import os
import requests
"""
Simple script that lets you upload the caltec pictures dataset into a postgres (pgvector) database
"""

url = "http://127.0.0.1:8001/search_image_to_image/"
file_path = "./images/img_1.png"  # folder path to the images


if __name__ == "__main__":
    with open(file_path, "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
        print(response)