import torch
import torch.nn as nn
from utils.parser import parse_args
from torchvision import transforms
from get_data.load_data import data_loader
from utils.seed import set_seed
from utils.device import select_device
from model.caption import Caption
from utils.valid import validation
from tqdm import tqdm

def main():
    args = parse_args() 
    device = select_device()
    set_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness = 0.2,
            contrast = 0.2,
            saturation = 0.2,
            hue = 0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    ])

    dl, ds = data_loader(img_path = args.img_path,
                        txt_path = args.txt_path, 
                        batch_size = args.batch_size,
                        threshold = args.threshold,
                        shuffle = True, 
                        transform = train_transform)

    embed_size = args.embed_size
    hidden_size = args.hidden_size
    vocab_size = len(ds.vocab.stoi)
    num_layers = args.num_layers

    model = Caption(embed_size, hidden_size, vocab_size, num_layers).to(device)

    lr = args.lr
    epochs = args.epoch
    loss_fn = nn.CrossEntropyLoss(ignore_index = ds.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    if args.load_model:
        print("Loading Pre-trained model...")
        model.load_state_dict(torch.load("/Users/youngseo/Desktop/lstm_caption85.pth"))

    model.train()

    for epoch in range(epochs):
        print("====== Epoch {} ======".format(epoch + 1))

        rt_loss = 0

        for (imgs, captions) in tqdm(dl, total = len(dl), leave = False):
            imgs, captions = imgs.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, captions[:-1])
            loss = loss_fn(outputs[1:].reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))
            loss.backward()
            optimizer.step()
            rt_loss += loss.item()

        print("Loss = {:.4f}".format(rt_loss / len(dl)))
        validation(model, ds, test_transform, device, max_length = args.max_length, beam_size = args.beam_size)

        if args.save_model:
            torch.save(model.state_dict(), "lstm_caption.pth")

if __name__ == "__main__":
    main()