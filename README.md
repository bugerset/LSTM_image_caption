# LSTM Image Captioning with PyTorch

> Image Captioning is a great project for studying the Encoder-Decoder structure — so I built an LSTM-based captioning model trained on the Flickr8k Datase

You can download Flickr8k Dataset in this link

https://www.kaggle.com/datasets/adityajn105/flickr8k

---

# What is Image Caption?

Image Caption is a **Computer Vision** and **Natural Language Processing** task that automatically generates a human-readable sentence describing an input image.

If you give an image to model, then model creates description of Image.

For example,
| Input Image | Description(Example) |
|-------------|----------------------|
|<img width="772" height="512" alt="스크린샷 2026-05-13 오후 9 37 46" src="https://github.com/user-attachments/assets/3279dfba-fca8-436c-9dfb-f20fd499397f" />|A white dog with a green collar is lying on the floor.|

# How it works? - Architecture Overview

**Encoder (MobileNetV3)**

A pre-trained MobileNet_V3 acts as the feature extractor. 

CNNs are well known for capturing visual features through convolution operations.

In a standard classification pipeline, features pass through a fully-connected layer and a Softmax to produce class probabilities. 

Here, we strip everything after the final Linear layer and instead feed those features directly into the Decoder.

<img width="944" height="397" alt="스크린샷 2026-05-13 오후 9 55 09" src="https://github.com/user-attachments/assets/2ebf1c10-b9f4-446e-b0ff-b55df43b5a6a" />

**Decoder (LSTM)**

Before diving into LSTM, it helps to understand RNN (Recurrent Neural Network).

RNNs are designed to process sequential data. Just as humans predict the next word based on prior context, RNNs maintain a hidden state that carries information from previous time steps — giving the model a form of memory.

<img width="885" height="329" alt="스크린샷 2026-05-13 오후 10 08 18" src="https://github.com/user-attachments/assets/44d94cdf-f0bb-4528-8be5-bf7eaa796107" /> |

However, RNNs suffer from the vanishing gradient problem: as sequences grow longer, earlier information gradually fades due to the limits of the hyperbolic tangent activation.

LSTM (Long Short-Term Memory) addresses this by introducing a Cell State, which acts as a long-term memory lane and significantly reduces information loss over long sequences.

---

# Generating Captions - Training vs Inference

**Training : Teacher Forcing**

A naive approach would be to feed each predicted word back as input for the next step. While this works, it is extremely slow to train.

Instead, we use Teacher Forcing: rather than using the model's own predictions as inputs, we feed the ground-truth words at each time step. This dramatically speeds up convergence.

<img width="215" height="192" alt="스크린샷 2026-05-13 오후 10 26 02" src="https://github.com/user-attachments/assets/ff82d257-1f71-48ff-9621-f2ad0ab328bf" />

But during inference, Teacher Forcing is disabled — instead, each predicted word is fed back as the next input, allowing the model to generate captions autoregressively.

Although this may look complex at first, walking through the code in the model/caption file makes it fairly straightforward to follow.

<img width="224" height="195" alt="스크린샷 2026-05-13 오후 10 27 29" src="https://github.com/user-attachments/assets/6364b76c-3e22-4ffe-aded-97ad63e5ade8" />

That said, I noticed the model's predictions weren't particularly great with plain greedy decoding — so I implemented Beam Search to improve caption quality.

## What is Beam-Search?

You can see Beam Search applied in the inference function inside caption.py.

Previously, the decoder selected the single highest-probability word at every time step — this is called Greedy Decoding. 

The problem is that as the sentence grows longer, an earlier word choice that seemed optimal may turn out to be suboptimal, and there is no way to undo that decision once it's made.

Beam Search addresses this weakness by keeping the top-k candidate words at each step and discarding the rest.

For example, with k = 3:

In Greedy Decoding => ["I"]

In Beam-Search => ["I", "Am", "The"]

From each of these candidates — "I", "Am", "The" — the model predicts the next word, then again retains only the top-k highest-scoring sequences. 

This process repeats until the sentence is complete, and the sequence with the highest overall probability is selected as the final caption.

# Environment & Setup
I run this code in:
```
python=3.9.25
pip install torch torchvision nltk pandas numpy
```

# Folder Structure:
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

# CLI argparse

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
    • --threshold (default = 3) <-- (Lower = large vocab, Higher = small vocab)

  • Data
    • --img_path
    • --txt_path

  • Else
    • --seed (default = 845)
    • --load_model (default = False) <-- (if you declare this, then load pth)
    • --save_model (default = True) <-- (if you declare this, then save pth)
```

# Expected Output

