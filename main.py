import requests
import os
import gzip
import numpy as np

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",   # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",        # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",   # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz"         # 10,000 test labels.
}

data_dir = '../_data'
os.makedirs(data_dir, exist_ok=True)

base_url = "https://github.com/rossbar/numpy-tutorial-data-mirror/blob/main/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status() # 다운로드가 성공했는지 확인
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

mnist_dataset = {}

# 이미지
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), 'rb') as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=16).reshape(-1, 28*28)
# 라벨
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), 'rb') as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)
