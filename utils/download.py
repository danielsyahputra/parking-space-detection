from genericpath import isdir
import os
import gdown
import zipfile
import logging

def check_dir(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def download_data(dir_name: str = "data") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    logging.info("Downloading data....")
    gdown.download(
        "https://drive.google.com/uc?id=1xhYtY6NQaY7sB-bZtmcxGd_grgfJnjrY", quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile("rois_gopro.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    os.remove("rois_gopro.zip")
    os.chdir("..")

def download_mlruns() -> None:
    logging.info("Downloading data....")
    gdown.download(
        "https://drive.google.com/uc?id=1D87knvaUwQLpZxLBVwPqnwsl8sNZLS4y", quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile("mlruns.zip", 'r') as zip_ref:
        zip_ref.extractall("mlruns")
    os.remove("mlruns.zip")