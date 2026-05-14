import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.features[-1:].parameters():
            param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        feature = self.dropout(self.model(x))
        return feature