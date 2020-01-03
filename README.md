# tiny SSD 

Object detection 모델인 SSD를 축소시킨 tiny ssd는 base network인 VGG16을 VGG11로 대체하고 모델이 예측하는 8732개의 bounding box를 6792개를 예측하도록 만들었다. 즉, 모델의 parameter 수를 줄여서 모델을 경량화시키고 예측하는 bounding box 개수를 줄여서 연산량을 감소시키고자 했다.

이렇게 감소된 parameter 수는 아래와 같다

- base network based on VGG16 : 20483904

- base network based on VGG11 : 3409408

또 tiny ssd는 다시 2개의 모델로 나뉘어 진다. aspect ration가 1일 때, 보통 크기와 작은 크기의 default box 중 하나를 선택하는 모델을 만들어서 서로의 성능 비교를 하고자 했으며 아래와 같이 3개의 모델에 대해서 성능 비교를 실시했다.

1. Original SSD
2. tiny SSD(aspect ratio, normal size)
3. tiny SSD(aspect ratio, min size)

library 및 version
- Python 3.6.5, Pytorch 1.1.0

# Datasets

사용한 데이터셋은 VOC2007과 VOC2012, BDD100K로 VOC를 통해서 학습시킨 모델을 BDD100K에서 전이학습을 시키고자 하였다.

# Experiments

2020.01.03에 업로딩 예정..

# References

작성한 코드의 대부분의 틀은 github 사이트에서 참조를 하였다
https://github.com/amdegroot/ssd.pytorch

BDD100K(Berkeley DeepDrive, self car driving Datasets)
https://github.com/ucbdrive/bdd-data
