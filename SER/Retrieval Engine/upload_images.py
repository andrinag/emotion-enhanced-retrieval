import os
import requests
import kagglehub
import shutil

"""
Simple script that lets you upload the pokemon pictures dataset into a postgres (pgvector) database
"""

url = "http://127.0.0.1:8000/upload_image/"
url_query = "http://127.0.0.1:8000/search/?query={query}"
url_image = "http://127.0.0.1:8000/images/?image={image}"

folder_path = "images"  # folder path to the images
query = "girl"


def get_file_list():
    """
    creates a list of files at file location
    :return: list of image names
    """
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith((".jpg")):
            files.append(("files", (filename, open(file_path, "rb"), "image/jpeg")))

    return files

def download_kaggle_videos():
    path = kagglehub.dataset_download("example/example")
    print("Path to dataset files:", path)


def copy_all_images(source_dir, dest_dir="images"):
    """
    method to extract all of the images in the caltec dataset into one single folder
    :param source_dir: source directory of the images
    :param dest_dir: where the images should be stores
    """
    os.makedirs(dest_dir, exist_ok=True)

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    src_path = os.path.join(category_path, file)
                    dest_path = os.path.join(dest_dir, file)

                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(file)
                        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {src_path} -> {dest_path}")



if __name__ == "__main__":
    # uploading videos, still needs to be changed for a better dataset
    """
    video1 = os.path.abspath("videos/butterflies_960p.mp4")
    video2 = os.path.abspath("videos/butterflies_1280.mp4")
    video3 = os.path.abspath("videos/elefant_1280p.mp4")
    video4 = os.path.abspath("videos/giraffes_1280p.mp4")
    video5 = os.path.abspath("videos/seafood_1280p.mp4")

    if not os.path.exists(video1) or not os.path.exists(video2):
        print("Error: One or both video files do not exist!")
    else:
        url = "http://127.0.0.1:8000/upload_videos/"
        video_files = [("files", open(video1, "rb")), ("files", open(video2, "rb"))]

        response = requests.post(url, files=video_files)
        print(response.json())
    """
    # uploading images from caltech dataset
    # copy_all_images("256_ObjectCategories")
    files = get_file_list()
    print(len(files))
    for i in range (0,len(files), 1000):
        print(f"currently working on files {i} of {len(files) / 1000}")
        response = requests.post(url, files=[files[i]])
    # response = requests.post(url, files=files)
    """
    response = requests.get(url_query)
    response = response.json()
    filenames = [item["filename"] for item in response["results"]]
    print(filenames)
    for filename in filenames:
        print(filename)
        image = filename
        requests.get(url_image)
    """