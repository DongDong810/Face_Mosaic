import os
import wget
import zipfile


def download_file(url, destination):
    folder = os.path.dirname(destination)
    if not os.path.exists(folder):
        os.makedirs(folder)
    wget.download(url, destination)
    print(f'\nDownload completed: {destination}')


def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f'Extraction completed: {extract_path}')


# Settings (WIDER FACE Dataset)

urls = ['https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_train.zip'
        'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip',
        'https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_test.zip',
        'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip']

zip_paths = ['data/widerface/zip/train.zip',
             'data/widerface/zip/val.zip',
             'data/widerface/zip/test.zip',
             'data/widerface/zip/annot.zip']

extract_paths = ['data/widerface/',
                 'data/widerface/',
                 'data/widerface/',
                 'data/widerface/']

for i in range(4):
    # download_file(urls[i], zip_paths[i])
    extract_zip(zip_paths[i], extract_paths[i])
