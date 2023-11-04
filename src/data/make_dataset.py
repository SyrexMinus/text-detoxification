import os
from zipfile import ZipFile

import wget


URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
ZIP_SAVE_PATH = "./data/raw/filtered.zip"
TSV_SAVE_DIR = "./data/raw"

wget.download(URL, ZIP_SAVE_PATH)
with ZipFile(ZIP_SAVE_PATH, "r") as zip_file:
    zip_file.extractall(TSV_SAVE_DIR)
os.remove(ZIP_SAVE_PATH)
