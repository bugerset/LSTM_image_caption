import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Caption(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Caption, self).__init__()

        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def inference(self, image, vocab, beam_size = 3, max_length = 50):
        with torch.no_grad():
            feature = self.encoder(image)
            beams = [(0.0, [], None, feature.unsqueeze(0))]
            completed = []

            for _ in range(max_length):
                candidates = []
                
                for score, tokens, states, x in beams:
                    output, new_states = self.decoder.lstm(x, states)
                    logits = self.decoder.linear(output.squeeze(0))
                    log_probs = torch.log_softmax(logits, dim=1)
                    
                    top_log_probs, top_idx = log_probs.topk(beam_size)
                    
                    for i in range(beam_size):
                        token_id = top_idx[0][i].item()
                        token = vocab.itos[token_id]
                        new_score = score + top_log_probs[0][i].item()
                        new_x = self.decoder.embed(top_idx[0][i].unsqueeze(0)).unsqueeze(0)
                        
                        if token == "<EOS>":
                            completed.append((new_score / (len(tokens) + 1), tokens))
                        elif token != "<PAD>" and token != "<SOS>":
                            candidates.append((new_score, tokens + [token], new_states, new_x))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]
            
            if not completed:
                completed = [(b[0] / max(len(b[1]), 1), b[1]) for b in beams]
            
            best = max(completed, key=lambda x: x[0])
            return best[1]