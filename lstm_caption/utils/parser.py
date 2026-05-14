import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Image Caption Argparser")

    # Model Setting
    parser.add_argument("--embed_size", type = int, default = 256)
    parser.add_argument("--hidden_size", type = int, default = 256)
    parser.add_argument("--num_layers", type = int, default = 1)

    # Train
    parser.add_argument("--lr", type = float, default = 0.0005)
    parser.add_argument("--epoch", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--threshold", type = int, default = 3)
    parser.add_argument("--beam_size", type = int, default = 3)
    parser.add_argument("--max_length", type = int, default = 50)

    # Data
    parser.add_argument("--img_path", type = str, default = "Images")
    parser.add_argument("--txt_path", type = str, default = "captions.txt")

    # Else
    parser.add_argument("--seed", type = int, default = 845)
    parser.add_argument("--load_model", action="store_true", default = False)
    parser.add_argument("--save_model", action="store_false", default = True)

    return parser.parse_args()