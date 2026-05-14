import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embedding = self.dropout(self.embed(captions))
        embedding = torch.cat((features.unsqueeze(0), embedding), dim = 0)
        outputs, _ = self.lstm(embedding)
        outputs = self.linear(outputs)
        return outputs

