# LSTM_image_caption with Pytorch


I think Image Caption project is good for study Encoder-Decoder Structure.


So, I made LSTM Image Caption train code with `Flickr8k Dataset`.

# What is Image Caption?

Image Caption is a computer vision and natural language processing task that automatically generates a descriptive, human-readable sentence for an input image.

If you giva a image to model, then model creates description of Image.

For example,
| Input Image | Description(Example) |
|-------------|-------------|
|<img width="772" height="512" alt="스크린샷 2026-05-13 오후 9 37 46" src="https://github.com/user-attachments/assets/3279dfba-fca8-436c-9dfb-f20fd499397f" />|A white dog with a green collar is lying on the floor.|

# How it works? (Structure)

First, Pre-trained MobileNet_V3 extract the feature-map of image.


Known as well, Convolution extract the feature of image.


<img width="944" height="397" alt="스크린샷 2026-05-13 오후 9 55 09" src="https://github.com/user-attachments/assets/2ebf1c10-b9f4-446e-b0ff-b55df43b5a6a" />


Originally, features pass through a soft-max and then argmaxed value is predicted to output in Classification.

우리는 다른거 다 필요없이, 이미지를 추출하는 Convolution과 Softmax로 향하기 위한 마지막 Linear 부분만 살려둘 것이다.


그렇게 나온 결과들은 Softmax로 향하는것이 아니라, going to Decoder that constructed with LSTM.


LSTM에 대해서 알아보기 전에, RNN이라는 순환 구조를 알아야한다.


<img width="885" height="329" alt="스크린샷 2026-05-13 오후 10 08 18" src="https://github.com/user-attachments/assets/44d94cdf-f0bb-4528-8be5-bf7eaa796107" />

RNN은 연속적인 시퀸스를 처리하기 위한 신경망이다.

사람은 글을 읽을 때, 이전 단어들에 대한 이해를 바탕으로 다음 단어를 예측한다. 이를 모방한 구조라고 볼 수 있다.

RNN은 시점에 따라서 입력을 받는데, 중간에 있는 Hidden이 연산을 위해서 직전 시점의 Hidden을 입력받는다.

이러한 점이 RNN이 정보를 기억하고 있는 비결이다.

하지만 RNN은 시점이 길어지면 길어질수록, 앞에 있던 정보가 소실되는 문제가 있다. (RNN에서 사용되는 Hyper-bolic tangent의 한계로 인해서)

이런 문제를 어느정도 해결할 수 있는 방법이 바로 LSTM(Long-Short Term Memory)이다.

LSTM은 Cell-State를 사용하여, RNN에 비해서 시점이 길어져도, 앞에 있던 정보의 손실이 덜 생긴다는 장점이 있다.

지금까지 나온 내용으로는 Encoder에서 특징을 추출하고, LSTM에 넣어서 을 배웠다.

그런데 문장을 생성하는 과정이 어떻게 이루어질까?

# About generates caption

우선 학습에서 단어 예측은 어떤 방식일까?

학습은 Loss를 줄이기 위해서 실제 이미지 캡션 == 예측 이미지 캡션이 되도록 해야한다.

현명하게 생각해보자. 특징이 주어졌을때, 예측 단어를 하나 뱉는 경우에 그 예측 단어를 다시 입력으로 넣어서 학습을 하면 Caption을 잘 만들어낼 것이다.

하지만 신경망은 생각보다 엄격하다.

위와 같은 방식으로 학습을 시키는 경우에는 생각보다 모델의 정확도가 떨어질 뿐만 아니라, 모델이 엉뚱한 결과를 학습하는 문제가 생길 수 있다.










# What is Beam-Search?




## Recommended Folder Structure

```
├── main.py
├── model/
│     ├──  encoder.py
│     ├── decoder.py
│     └── caption.py
├── get_data/
│     └── load_data.py
├── utils/
│     ├── valid.py
│     ├── parser.py
│     ├── device.py
│     └── seed.py
├── Images/
├── captions.txt
├── dog.jpg
├── horse.jpg
└── lstm_caption.pth
```

## CLI argparse

Key arguments (from `utils/parser.py`):
```
  • Reproducibility
    • --seed (default: 845)

  • Mode
    • --train-mode / --no-train-mode (default: True)
    • --test-mode / --no-test-mode (default: True)
    • --consistency / --no-consistency (default: True)
        └── fixes same samples for reconstruction comparison across epochs

  • Dataset
    • --data-set (default: mnist, choices: [mnist, cifar10, cifar100])
    • --root (default: ./data)

  • VAE
    • --latent-vector (default: 20)

  • Training
    • --batch-size (default: 16)
    • --lr (default: 1e-3)
    • --epoch (default: 50)

  • Testing
    • --nos (number of samples to generate, default: 64)

  • Save paths
    • --save-pt (default: ./pt_save)
    • --save-img (default: ./img_save)
```

## Expected Output
