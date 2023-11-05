import os
from zipfile import ZipFile

import wget
import pandas as pd
from sklearn.model_selection import train_test_split


TSV_PATH = "./data/raw/filtered.tsv"
URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
ZIP_SAVE_PATH = "./data/raw/filtered.zip"
TSV_SAVE_DIR = "./data/raw"
TRAIN_DATA_PATH = "./data/internal/train.tsv"
VAL_DATA_PATH = "./data/internal/validatation.tsv"
VALIDATION_RATIO = 0.01


def download_dataset():
    """Downloads the zipped dataset and unpacks it on the disk"""
    if not os.path.isfile(TSV_PATH):
        wget.download(URL, ZIP_SAVE_PATH)
        with ZipFile(ZIP_SAVE_PATH, "r") as zip_file:
            zip_file.extractall(TSV_SAVE_DIR)
        os.remove(ZIP_SAVE_PATH)


def preprocess_dataset():
    """Preprocesses the filtered ParaNMT-detox corpus and stores train/test sets on the disk as .tsv.

    The preprocessing includes removal of entries with more toxic translation than reference and entries with
    non-toxic reference or toxic translation."""
    dataset = pd.read_csv(TSV_PATH, sep="\t", index_col=0)
    dataset.drop_duplicates(["reference", "translation"], inplace=True)
    dataset.drop(dataset[dataset["trn_tox"] > dataset["ref_tox"]].index, inplace=True)
    dataset.drop(dataset[dataset["ref_tox"] < 0.5].index, inplace=True)
    dataset.drop(dataset[dataset["trn_tox"] > 0.5].index, inplace=True)

    train_split, val_split = train_test_split(dataset.index, test_size=VALIDATION_RATIO, random_state=420)
    train = dataset[dataset.index.isin(train_split)]
    val = dataset[dataset.index.isin(val_split)]

    train.to_csv(TRAIN_DATA_PATH, sep="\t")
    val.to_csv(VAL_DATA_PATH, sep="\t")


if __name__ == "__main__":
    download_dataset()
    preprocess_dataset()
