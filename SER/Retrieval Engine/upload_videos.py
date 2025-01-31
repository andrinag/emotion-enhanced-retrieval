import os
import requests
import kagglehub
import shutil

"""
Simple script that lets you upload the caltect video dataset into a postgres (pgvector) database via engine
"""

def download_kaggle_videos():
    path = kagglehub.dataset_download("example/example")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    # uploading videos, still needs to be changed for a better dataset
    video_list = []
    video_list.append(os.path.abspath("videos/butterflies_960p.mp4"))
    video_list.append(os.path.abspath("videos/butterflies_1280.mp4"))
    video_list.append(os.path.abspath("videos/elefant_1280p.mp4"))
    video_list.append(os.path.abspath("videos/giraffes_1280p.mp4"))
    video_list.append(os.path.abspath("videos/seafood_1280p.mp4"))

    url = "http://127.0.0.1:8000/upload_videos/"
    video_files = []
    for i in range(5):
        video_files.append(("files", open(video_list[i], "rb")))

    response = requests.post(url, files=video_files)
    print(response.json())
