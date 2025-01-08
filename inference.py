import argparse
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from models import Encoder128, Decoder128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, default="out.png")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--attr_index", type=int, default=31)  # ex. Smiling ?
    parser.add_argument("--attr_value", type=float, default=1.0, 
                        help="à quel point on renforce l'attribut (+1, -1, etc.)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = Encoder128(latent_dim=args.latent_dim).to(device)
    decoder = Decoder128(latent_dim=args.latent_dim, attr_dim=40).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # Transform identique à l'entraînement
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Charger l'image
    image = Image.open(args.input_image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)  # [1,3,128,128]

    # On crée un vecteur d'attributs 'y'. Par simplification, on part de 0
    # (ou on pourrait détecter l'attribut via un classif).
    # On aura y in [-1,1], on force tout à 0 sauf l'index 'attr_index'.
    y = torch.zeros((1,40), device=device)
    # On met +1 ou -1 au besoin
    y[0, args.attr_index] = args.attr_value

    with torch.no_grad():
        z = encoder(x)            # [1,latent_dim]
        x_hat = decoder(z, y)     # [1,3,128,128]

    # Revenir en [0,1]
    out_tensor = 0.5*(x_hat[0] + 1.0)
    out_tensor = out_tensor.clamp(0,1)
    out_pil = transforms.ToPILImage()(out_tensor.cpu())
    out_pil.save(args.output_image)
    print(f"[INFO] Saved manipulated image in {args.output_image}")

if __name__ == "__main__":
    main()
