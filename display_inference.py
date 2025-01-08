#!/usr/bin/env python3

import argparse
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# On suppose que models.py contient Encoder128, Decoder128
from models import Encoder128, Decoder128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin du fichier .pth (ex: fader_epoch_10.pth)")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Chemin de l'image d'entrée (ex: Img/img_align_celeba/000001.jpg)")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Dimension latente (identique à l'entraînement)")
    parser.add_argument("--attr_index", type=int, default=31,
                        help="Index de l'attribut à manipuler (0..39)")
    parser.add_argument("--attr_value", type=float, default=1.0,
                        help="Valeur souhaitée pour cet attribut (ex: +1, -1, +2...)")
    parser.add_argument("--title_in", type=str, default="Input Image",
                        help="Titre à afficher sur l'image d'entrée")
    parser.add_argument("--title_out", type=str, default="Output Image",
                        help="Titre à afficher sur l'image de sortie")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Charger le checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Instancier l'encodeur/décodeur
    encoder = Encoder128(latent_dim=args.latent_dim).to(device)
    decoder = Decoder128(latent_dim=args.latent_dim, attr_dim=40).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    # 2) Définir la transformation d'entrée (même que l'entraînement)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    # 3) Charger et transformer l'image
    image_pil = Image.open(args.input_image).convert('RGB')
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # [1,3,128,128]

    # 4) Créer un vecteur d'attributs y (on part de 0 pour la démo)
    y = torch.zeros((1, 40), device=device)  # shape [1,40]
    y[0, args.attr_index] = args.attr_value  # ex: +1.0 ou +2.0

    # 5) Forward (inférence)
    with torch.no_grad():
        z = encoder(image_tensor)          # [1,latent_dim]
        x_hat = decoder(z, y)             # [1,3,128,128]

    # 6) Repasser l'image reconstruite en plage [0,1] (car Tanh → [-1,1])
    #    pour pouvoir l'afficher
    x_hat_0_1 = 0.5*(x_hat[0] + 1.0)
    x_hat_0_1 = x_hat_0_1.clamp(0,1)  # [3,128,128] en range [0,1]

    # 7) Préparer l'affichage matplotlib
    #    On va afficher côte à côte l'image d'entrée et l'image de sortie
    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    # Image d'entrée (non normalisée pour l'affichage)
    # On peut l'afficher directement à partir du PIL de départ
    axes[0].imshow(image_pil)
    axes[0].set_title(args.title_in)
    axes[0].axis("off")

    # Image de sortie
    # On convertit x_hat_0_1 en PIL
    out_pil = transforms.ToPILImage()(x_hat_0_1.cpu())
    axes[1].imshow(out_pil)
    axes[1].set_title(args.title_out)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("Che.png")

if __name__ == "__main__":
    main()
