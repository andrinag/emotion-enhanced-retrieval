import os
import requests
from fastapi.responses import FileResponse

"""
Simple script that lets you upload the pokemon pictures dataset into a postgres (pgvector) database
"""

folder_path = "images/archive/images"
url = "http://127.0.0.1:8000/upload/"
url_query = "http://127.0.0.1:8000/search/?query={query}"
url_image = "http://127.0.0.1:8000/images/?image={image}"

folder_path = "images/archive/images"  # the downloaded pokemon dataset

query = "girl"


def get_file_list():
    """
    creates a list of files at file location
    :return: list of image names
    """
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith((".png")):
            files.append(("files", (filename, open(file_path, "rb"), "image/jpeg")))

    return files



if __name__ == "__main__":
    response = requests.get(url_query)
    response = response.json()
    filenames = [item["filename"] for item in response["results"]]
    print(filenames)  # Output: ['ekans.png', 'bonsly.png']
    for filename in filenames:
        print(filename)
        image = filename
        requests.get(url_image)
