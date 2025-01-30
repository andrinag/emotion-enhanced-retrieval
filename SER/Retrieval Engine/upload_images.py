import os
import requests

"""
Simple script that lets you upload the pokemon pictures dataset into a postgres (pgvector) database
"""

folder_path = "./images/archive(11)/images"
url = "http://127.0.0.1:8000/upload/"


def get_file_list():
    """
    creates a list of files at file location
    :return: list of image names
    """
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            files.append(("files", (filename, open(file_path, "rb"), "image/jpeg")))

    return files


if __name__ == "__main__":
    files = get_file_list()

    if not files:
        print("No images found in the folder!")
    else:
        try:
            response = requests.post(url, files=files)
            print(response.text)
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
        finally:
            for _, f in files:
                f[1].close()
