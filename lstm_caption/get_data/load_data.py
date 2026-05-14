import torch
import torch.nn as nn
import nltk
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Vocab:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    @staticmethod
    def tokenizer_eng(text):
        return nltk.word_tokenize(text.lower())
    
    # Count vocab
    def build_vocab(self, sentence_list):
        freq = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                freq[word] = freq.get(word, 0) + 1

                if freq[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in self.tokenizer_eng(text)
            ]

class FlickrDataset(Dataset):
    def __init__(self, img_root, txt_root, transform = None, threshold = 5):
        self.img_root = img_root
        self.df = pd.read_csv(txt_root)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocab(threshold)
        self.vocab.build_vocab(self.captions.tolist())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img = self.imgs[index]
        img = Image.open(os.path.join(self.img_root, img)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

# Padding data
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def data_loader(img_path, txt_path, batch_size, threshold, shuffle = True, transform = None):
    flickrData = FlickrDataset(img_path, txt_path, transform = transform, threshold = threshold)
    pad_idx = flickrData.vocab.stoi["<PAD>"]
    dl = DataLoader(flickrData, batch_size = batch_size, shuffle = shuffle, collate_fn = MyCollate(pad_idx))

    return dl, flickrData