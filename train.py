#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm  # pour la barre de progression

# On suppose que data.py et models.py se trouvent dans le même dossier
from data import get_celeba_dataloader
from model2 import Encoder128, Decoder128, DiscriminatorLatent


def train_one_epoch(loader, encoder, decoder, disc,
                    opt_encdec, opt_disc, device,
                    lambda_adv=0.01,
                    epoch=1,
                    total_epochs=10):
    """
    Entraîne le modèle sur un epoch,
    avec une barre de progression `tqdm` pour afficher la progression batch/batch.

    Arguments:
      loader (DataLoader)       : DataLoader du split (ex: train).
      encoder, decoder, disc    : modules PyTorch (FaderNets).
      opt_encdec, opt_disc      : optimiseurs (Adam ou autre).
      device                    : 'cuda' ou 'cpu'.
      lambda_adv (float)        : pondération de la perte adversariale latente.
      epoch, total_epochs (int) : pour affichage du numéro d’epoch.
    """
    encoder.train()
    decoder.train()
    disc.train()

    # Critères
    recon_criterion = nn.MSELoss()         
    bce_logits = nn.BCEWithLogitsLoss()    

    total_recon_loss = 0.
    total_disc_loss = 0.
    steps = 0

    # Barre de progression : on indique l'epoch en cours
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for batch_idx, (imgs, attrs) in enumerate(pbar, start=1):
        imgs = imgs.to(device)      # [B,3,128,128]
        attrs = attrs.to(device)    # [B,40], -1 ou +1
        attrs_01 = (attrs + 1.)/2.  # convertit en [0,1]

        # -----------------------------
        # 1) Entraîner le Discriminateur
        # -----------------------------
        with torch.no_grad():
            z_detach = encoder(imgs)  # pas de gradient sur l’encodeur
        z_detach = z_detach.detach()

        logits_disc = disc(z_detach)  # [B,40]
        disc_loss = bce_logits(logits_disc, attrs_01)

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # -----------------------------
        # 2) Entraîner (Encodeur + Décodeur)
        # -----------------------------
        z = encoder(imgs)              # [B,latent_dim]
        x_hat = decoder(z, attrs)      # [B,3,128,128]

        # (a) Reconstruction
        recon_loss = recon_criterion(x_hat, imgs)

        # (b) Adversarial : on veut tromper le discriminateur
        logits_enc = disc(z)
        adv_loss   = bce_logits(logits_enc, 1. - attrs_01)

        encdec_loss = 3*recon_loss + lambda_adv * adv_loss

        opt_encdec.zero_grad()
        encdec_loss.backward()
        opt_encdec.step()

        # Accumulateur de stats
        total_recon_loss += recon_loss.item()
        total_disc_loss  += disc_loss.item()
        steps += 1

        # Mettre à jour l'affichage sur tqdm
        pbar.set_postfix({
            "batch": f"{batch_idx}/{len(loader)}",
            "recon": f"{recon_loss.item():.4f}",
            "disc":  f"{disc_loss.item():.4f}"
        })

    mean_recon = total_recon_loss / steps
    mean_disc  = total_disc_loss  / steps
    return mean_recon, mean_disc

def validate_one_epoch(val_loader, encoder, decoder, disc, device):
    encoder.eval()
    decoder.eval()
    disc.eval()

    recon_criterion = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    total_recon_loss = 0.
    total_disc_loss = 0.
    steps = 0

    with torch.no_grad():
        for imgs, attrs in val_loader:
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            attrs_01 = (attrs + 1.)/2.

            # On encode et décode, comme en train
            z = encoder(imgs)
            x_hat = decoder(z, attrs)
            recon_loss = recon_criterion(x_hat, imgs)

            # Discriminateur (si on veut voir si z reste prédictif)
            logits_disc = disc(z)
            disc_loss = bce_logits(logits_disc, attrs_01)

            total_recon_loss += recon_loss.item()
            total_disc_loss  += disc_loss.item()
            steps += 1

    mean_recon = total_recon_loss / steps
    mean_disc  = total_disc_loss  / steps

    return mean_recon, mean_disc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--attr_file", type=str, required=True)
    parser.add_argument("--eval_partition", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--lambda_adv", type=float, default=0.01)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")

    # Ajout du paramètre fraction
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction du dataset à utiliser (ex. 0.05 = 5% pour un test).")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # On charge seulement fraction * 100% du dataset
    train_loader = get_celeba_dataloader(
        data_dir=args.data_dir,
        attr_file=args.attr_file,
        eval_partition=args.eval_partition,
        partition='train',
        batch_size=args.batch_size,
        shuffle=True,
        fraction=args.fraction
    )
    
    val_loader = get_celeba_dataloader(
    data_dir=args.data_dir,
    attr_file=args.attr_file,
    eval_partition=args.eval_partition,
    partition='val',         # <-- on met 'val'
    batch_size=args.batch_size,
    shuffle=False
    )


    # Instanciation des modèles, etc. (inchangé)
    encoder = Encoder128(latent_dim=args.latent_dim).to(device)
    decoder = Decoder128(latent_dim=args.latent_dim, attr_dim=40).to(device)
    disc    = DiscriminatorLatent(latent_dim=args.latent_dim, attr_dim=40).to(device)

    opt_encdec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, betas=(0.5,0.999))
    opt_disc   = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5,0.999))

    for epoch in range(1, args.epochs+1):
        recon_loss_avg, disc_loss_avg = train_one_epoch(
            loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            disc=disc,
            opt_encdec=opt_encdec,
            opt_disc=opt_disc,
            device=device,
            lambda_adv=args.lambda_adv,
            epoch=epoch,
            total_epochs=args.epochs
        )
        print(f"\nEpoch {epoch}/{args.epochs} terminé "
              f"| recon_loss={recon_loss_avg:.4f} | disc_loss={disc_loss_avg:.4f}")
        
        
        
        val_recon, val_disc = validate_one_epoch(val_loader, encoder, decoder, disc, device)
        print(f"Epoch {epoch} Val    | recon={val_recon:.4f}, disc={val_disc:.4f}")

        ckpt_path = os.path.join(args.out_dir, f"fader_epoch_{epoch}.pth")
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "disc": disc.state_dict()
        }, ckpt_path)
        print(f"[INFO] Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()