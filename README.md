# LSTM_image_caption with Pytorch


I think Image Caption project is good for study Encoder-Decoder Structure.


So, I made LSTM Image Caption train code with `Flickr8k Dataset`.

# What is Image Caption?

Image Caption is a computer vision and natural language processing task that automatically generates a descriptive, human-readable sentence for an input image.

If you giva a image to model, then model creates description of Image.

For example,
| Input Image | Description(Example) |
|-------------|----------------------|
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

하지만 학습의 속도가 매우매우 느려지는 결과가 생긴다.

그래서 Teacher Forcing이라는 방식을 사용하여, 모델이 빠르게 학습한다.

<img width="215" height="192" alt="스크린샷 2026-05-13 오후 10 26 02" src="https://github.com/user-attachments/assets/ff82d257-1f71-48ff-9621-f2ad0ab328bf" />
Caption이 "What will the cat sit on"인 경우에, 다음 단어가 어떤것이 나와야 올바른 것인지를 학습한다.

이런 이유로 Teacher Forcing이라고 불린다.

테스트 과정에서는 결과의 정확도를 높이기 위해 Teacher Forcing 없이, 예측 단어를 입력으로 넣어서 올바른 Caption이 잘 생성되도록 해준다.
<img width="224" height="195" alt="스크린샷 2026-05-13 오후 10 27 29" src="https://github.com/user-attachments/assets/6364b76c-3e22-4ffe-aded-97ad63e5ade8" />

엄청 복잡하지만, model폴더에 있는 코드들을 보면서 이해하면 그다지 어렵지 않을 것이다.

그런데 모델의 예측 결과가 썩 좋지 않은 결과를 내는 것을 발견했다.

그래서 Beam-Search를 통해 문장을 생성하도록 만들었다.

# What is Beam-Search?
caption.py에 있는 inference를 보면 Beam-Search가 적용된 모습이다.

기존에는 디코더가 매 시점마다 가장 높은 확률을 가지는 단어를 선택하도록 하였다. (Greedy Decoding)

그러나 문장이 이어지면 이어질수록 이전에 골랐던 단어가 최적의 선택을 내지 못할 수 있고, 순간 잘못된 선택을 내리게 되어도 결정을 취소할 수 없다.

Beam-Search는 이런 단점을 보완하기 위해 만들어진 알고리즘으로, 확률이 가장 높은 단어 k개를 후보로 선택하고 나머지 단어는 제거한다.

예를 들면 다음과 같은 방식이다.

(k가 3이라면)
In Greedy Decoding => ["I"]
In Beam-Search => ["I", "Am", "The"]

여기서 I, Am, The 각각에 대해서 다음 단어를 예측하고 예측 단어들중 가장 확률이 높은 k개의 후보군을 다시 남긴다.

이런식으로 계속 문장을 만들어서 가장 확률이 높은 문장을 선택하는 것이 Beam-Search이다.



## Recommended Python Environment and Folder Structure
I run this code in:
``` Environment
python=3.9.25
pip install torch torchvision nltk pandas numpy
```

Folder Structure:
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
  • Model Setting
    • --embed_size (default = 256)
    • --hidden_size (default = 256)
    • --num_layers (default = 1)

  • Train
    • --lr (default = 0.0005)
    • --epoch (default = 100)
    • --batch_size (default = 128)
    • --threshold (default = 3) <--- (I use Count based Vocab, so if you want more larger vocab, then low that)

  • Data
    • --img_path
    • --txt_path

  • Else
    • --seed (default = 845)
    • --load_model (default = False)
    • --save_model (default = True)
```

## Expected Output
