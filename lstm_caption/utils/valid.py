from PIL import Image
import torch
import numpy as np

def validation(model, ds, tf, device, max_length, beam_size):
    model.eval()
    with torch.no_grad():
        print("\n=== Trained Image Caption ===")
        for i in range(3):
            index = np.random.randint(0, len(ds))
            img, caption = ds[index]
            true = " ".join([ds.vocab.itos[idx.item()] for idx in caption[1:-1]])
            pred = model.inference(img.unsqueeze(0).to(device), ds.vocab, beam_size, max_length)
            print(f"True : {true}")
            print(f"Pred : {' '.join(pred)}")

        print("\n=== New Image Caption ===")
        true_caption = ["A white dog with a green collar is lying on the floor.", 
                        "A woman in a white top is riding a brown horse across a green field under a cloudy sky."]
        test_images = ["dog.jpg", "horse.jpg"]

        for i in range(2):
            img = Image.open(test_images[i]).convert("RGB")
            img_tensor = tf(img).unsqueeze(0).to(device)
            pred = model.inference(img_tensor, ds.vocab, beam_size, max_length)
            print(f"True : {true_caption[i]}")
            print(f"Pred : {' '.join(pred)}")

        print()
    model.train()