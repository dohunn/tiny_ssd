# tiny ssd 

####Object detection 모델인 SSD를 축소시킨 tiny ssd는 base network인 VGG16을 VGG11로 대체하고 모델이 예측하는 8732개의 bounding box를 6792개를 예측하도록 만들었다. 즉, 모델의 parameter 수를 줄여서 모델을 경량화시키고 예측하는 bounding box 개수를 줄여서 연산량을 감소시키고자 했다.

####이렇게 감소된 parameter 수는 아래와 같다
## base network based on VGG16 : 20483904
## base network based on VGG11 : 20483904
