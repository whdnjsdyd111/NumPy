import requests
import os
import gzip
import numpy as np
import matplotlib.pylab as plt

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",   # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",        # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",   # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz"         # 10,000 test labels.
}

base_url = "https://github.com/rossbar/numpy-tutorial-data-mirror/blob/main/"
data_dir = os.getcwd()

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status()  # 다운로드가 성공했는지 확인
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

mnist_dataset = {}

# 이미지
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), 'rb') as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=16).reshape(-1, 28, 28)
# 라벨
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), 'rb') as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

x_train, y_train, x_test, y_test = (mnist_dataset["training_images"],
                                    mnist_dataset["training_labels"],
                                    mnist_dataset["test_images"],
                                    mnist_dataset["test_labels"])

print('The shape of training images: {} and training labels: {}'.format(x_train.shape, y_train.shape))
print('The shape of test images: {} and test labels: {}'.format(x_test.shape, y_test.shape))

# 훈련셋으로부터의 60,000 번째 이미지 (인덱스 59,999)를
# (784,) 를 (28, 28) 로 적합한 모양으로 만든다.
mnist_image = x_train[59999, :].reshape(28, 28)
# 검은색 배경을 사용하려면 색상 매핑을 그레이스케일로 설정한다.
plt.imshow(mnist_image, cmap='gray')
plt.show()


