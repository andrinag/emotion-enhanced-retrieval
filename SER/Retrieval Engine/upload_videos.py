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


def get_video_list_from_folder(folder_path):
    video_list = [
        os.path.abspath(os.path.join(folder_path, file))
        for file in os.listdir(folder_path)
        if file.endswith(".mp4")
    ]
    return video_list


if __name__ == "__main__":
    # uploading videos, still needs to be changed for a better dataset
    folder_path = "/media/V3C/V3C1/video-480p"
    video_list = get_video_list_from_folder(folder_path)

    url = "http://127.0.0.1:8000/upload_videos/"
    video_files = []
    for i in range(len(video_list)):
        video_files.append(("files", open(video_list[i], "rb")))

    response = requests.post(url, files=video_files)
    print(response.json())
