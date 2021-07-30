import requests
import os
import gzip
import numpy as np
import matplotlib.pylab as plt

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz"  # 10,000 test labels.
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


training_sample, test_sample = 1000, 1000
training_images = x_train[0:training_sample] / 255
test_images = x_test[0:test_sample] / 255


def one_hot_encoding(labels, dimension=10):
    # 10개의 치수(0 ~ 9) 0 벡터에 대한 단일 원핫 변수를 정의
    one_hot_labels = (labels[..., None] == np.arange(dimension)[None])
    # 인코딩된 원핫 라벨 반환
    return one_hot_labels.astype(np.float64)


training_labels = one_hot_encoding(y_train[:training_sample])
test_labels = one_hot_encoding(y_test[:test_sample])

seed = 884736743
rng = np.random.default_rng(seed)


# 입력이 양이면 반환하고 그렇지 않으면 0이면 반환하는 ReLU를 정의
def relu(x):
    return (x >= 0) * x


# 양의 입력에 대해 1을, 그렇지 않으면 0을 반환하는 ReLU 기능의 파생 모델을 설정
def relu2deriv(output):
    return output >= 0


learning_rate = 0.005
epochs = 20
hidden_size = 100
pixels_per_image = 784
num_labels = 10

weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1
weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1

# 테스트셋 손실과 시각화를 위한 정확한 예측을 저장
store_training_loss = []
store_training_accurate_pred = []
store_test_loss = []
store_test_accurate_pred = []

# 훈련 루프 시작
# 정의된 에포크(반복)의 수에 대한 학습 실험을 실행
for j in range(epochs):

    #################
    #    훈련 단계   #
    #################

    # 손실/오차와 예측 수를 0으로 초기화
    training_loss = 0.0
    training_accurate_predictions = 0

    # 훈련셋의 모든 이미지에 대한 순전파와 역전파를 수행하고 그에 따른 가중치 조정
    for i in range(len(training_images)):
        # 순/역전파 :
        # 1. 입력 레이어:
        #    훈련용 이미지 데이터를 입력으로 초기화
        layer_0 = training_images[i]
        # 2. 히든 레이어:
        #    훈련용 이미지를 랜덤하게 초기화된 가중치를 곱함으로써, 중간 레이어로 가져온다.
        layer_1 = np.dot(layer_0, weights_1)
        # 3. ReLU 활성화 함수를 통해 히든 레이어의 출력을 전달
        layer_1 = relu(layer_1)
        # 4. 정규화를 위한 드롭아웃 함수를 정의
        dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
        # 5. 히든 레이어의 출력에 드롭아웃을 적용
        layer_1 *= dropout_mask * 2
        # 6. 출력 레이어:
        #    중간 레이어의 출력을 랜덤하게 초기화된 가중치를 곱하여 최종 레이어로 수집
        #    점수가 10점인 10차원 벡터를 생성
        layer_2 = np.dot(layer_1, weights_2)

        # 역전파:
        # 1. 실제 이미지 레이블과 예측 사이의 오류(손실 함수)를 측정
        training_loss += np.sum((training_labels[i] - layer_2) ** 2)
        # 2. 정확한 예측 카운터 증가시킨다.
        training_accurate_predictions += int(np.argmax(layer_2) == np.argmax(training_labels[i]))
        # 3. 손실/오차를 구분한다.
        layer_2_delta = (training_labels[i] - layer_2)
        # 4. 손실 함수의 경사를 히든 레이어에 다시 전파
        layer_1_delta = np.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)
        # 5. 드롭아웃에 경사 적용
        layer_1_delta *= dropout_mask
        # 6. 학습률과 경사를 곱하여 중간 및 입력 계층에 가중치를 업데이트함
        weights_1 += learning_rate * np.outer(layer_0, layer_1_delta)
        weights_2 += learning_rate * np.outer(layer_1, layer_2_delta)

    # 훈련셋의 손실 및 정확한 예측을 저장
    store_training_loss.append(training_loss)
    store_training_accurate_pred.append(training_accurate_predictions)

    ###################
    #    측정 단계     #
    ###################

    # 각각의 에포크의 테스트셋 성능을 측정한다.

    # 휸련 단계와 달리, 각 이미지(배치)에 대한 수정이 되지 않는다.
    # 그러므로 개별적으로 각 이미지에 루프할 필요가 없는 벡터화된 방법으로 테스트용 이미지에 모델을 적용할 수 있다.

    results = relu(test_images @ weights_1) @ weights_2

    # 실제 레이블과 예측 값 사이의 오차를 측정한다.
    test_loss = np.sum((test_labels - results) ** 2)

    # 테스트셋의 예측 정확도 측정
    test_accurate_predictions = np.sum(
        np.argmax(results, axis=1) == np.argmax(test_labels, axis=1)
    )

    # 테스트셋의 손실 및 정확한 예측을 저장
    store_test_loss.append(test_loss)
    store_test_accurate_pred.append(test_accurate_predictions)

    # 각 에포크에서 오류 및 정확도 메트릭 요약
    print("\n" + \
          "Epoch: " + str(j) + \
          " Training set error:" + str(training_loss / float(len(training_images)))[0:5] + \
          " Training set accuracy:" + str(training_accurate_predictions / float(len(training_images))) + \
          " Test set error:" + str(test_loss / float(len(test_images)))[0:5] + \
          " Test set accuracy:" + str(test_accurate_predictions / float(len(test_images))))
