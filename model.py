import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder128(nn.Module):
    """
    Encodeur conçu pour des images 128x128 → un vecteur latent de dimension 'latent_dim'.
    Selon l'architecture du papier : 7 couches Conv-BatchNorm-ReLU, chacune divisant
    la résolution par 2, et en augmentant le nombre de canaux comme suit :

    3 (entrée) → 16 → 32 → 64 → 128 → 256 → 512 → latent_dim

    Avec une image 128x128, après 7 convolutions stride=2, la sortie est [B, latent_dim, 1, 1].
    On aplatit ensuite en [B, latent_dim].
    """
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # 7 blocs de conv (stride=2), qui font : 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
        self.conv = nn.Sequential(
            # bloc1: 128->64, C16
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc2: 64->32, C32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc3: 32->16, C64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc4: 16->8, C128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc5: 8->4, C256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc6: 4->2, C512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc7: 2->1, sort vers latent_dim + batchnorm + ReLU
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        z = self.conv(x)           # [B, latent_dim, 1, 1]
        z = z.view(z.size(0), -1)  # [B, latent_dim]
        return z


class Decoder128(nn.Module):
    """
    Décodeur pour images 128x128 ré-injectant (z, y) à chaque couche.
    On part de [B, latent_dim + attr_dim, 1,1] et on veut arriver à [B, 3, 128,128].
    
    Architecture (7 blocs de convT) :
       - Bloc1 : (latent_dim+attr_dim) -> 512
       - Bloc2 : (512 + latent_dim+attr_dim) -> 256
       - Bloc3 : (256 + latent_dim+attr_dim) -> 128
       - Bloc4 : (128 + latent_dim+attr_dim) -> 64
       - Bloc5 : (64  + latent_dim+attr_dim) -> 32
       - Bloc6 : (32  + latent_dim+attr_dim) -> 16
       - Bloc7 : (16  + latent_dAim+attr_dim) -> 3
       
    À la fin on applique un Tanh pour obtenir des pixels dans [-1,1].
    """
    def __init__(self, latent_dim=256, attr_dim=40):
        super().__init__()
        self.latent_dim = latent_dim
        self.attr_dim   = attr_dim
        in_dim          = latent_dim + attr_dim

        # --- Bloc 1 ---
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 2 ---
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512 + in_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 3 ---
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256 + in_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 4 ---
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128 + in_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 5 ---
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64 + in_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 6 ---
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(32 + in_dim, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # --- Bloc 7 ---
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(16 + in_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),      # on peut garder un BN ici si désiré
            nn.ReLU(inplace=True),  # ReLU avant la sortie
            nn.Tanh()               # pour avoir la sortie dans [-1, 1]
        )

    def forward(self, z, y):
        """
        Paramètres
        ----------
        z : [B, latent_dim]
        y : [B, attr_dim]
        Retour
        ------
        x_hat : [B, 3, 128, 128]
        """
        # Concatène (z, y) => code "in_dim"
        zy = torch.cat([z, y], dim=1)                  # [B, latent_dim+attr_dim]
        x  = zy.unsqueeze(-1).unsqueeze(-1)            # [B, in_dim, 1, 1]
        
        # --- Bloc 1 : pas besoin de concat, l'entrée est déjà (z,y) ---
        x = self.deconv1(x)                            # => [B, 512, 2, 2]
        
        # Pour chaque bloc suivant, on "ré-injecte" (z,y) en l'étirant
        def inject_and_deconv(x, deconv):
            # on étend [B, in_dim] en [B, in_dim, H, W] pour concat
            B, _, H, W = x.shape
            zy_broadcast = zy.unsqueeze(-1).unsqueeze(-1).expand(B, zy.shape[1], H, W)
            return deconv(torch.cat([x, zy_broadcast], dim=1))
        
        # --- Bloc 2 ---
        x = inject_and_deconv(x, self.deconv2)         # => [B, 256, 4, 4]
        # --- Bloc 3 ---
        x = inject_and_deconv(x, self.deconv3)         # => [B, 128, 8, 8]
        # --- Bloc 4 ---
        x = inject_and_deconv(x, self.deconv4)         # => [B, 64, 16, 16]
        # --- Bloc 5 ---
        x = inject_and_deconv(x, self.deconv5)         # => [B, 32, 32, 32]
        # --- Bloc 6 ---
        x = inject_and_deconv(x, self.deconv6)         # => [B, 16, 64, 64]
        # --- Bloc 7 ---
        x = inject_and_deconv(x, self.deconv7)         # => [B, 3, 128, 128] (Tanh)
        
        return x


import torch
import torch.nn as nn

class DiscriminatorLatent(nn.Module):
    def __init__(self, latent_dim=256, attr_dim=40, p=0.3):
        """
        Discriminateur prenant en entrée un vecteur latent [B, latent_dim]
        et prédisant [B, attr_dim].
        """
        super().__init__()
        # "C512 layer": linéaire (latent_dim->512) + LeakyReLU
        self.c512 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # fully-connected de deux couches: 512->512->attr_dim, avec dropout
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, attr_dim)
        )

    def forward(self, z):
        """
        Paramètres
        ----------
        z : [B, latent_dim]

        Retour
        ------
        logits : [B, attr_dim]
        """
        x = self.c512(z)   # [B,512]
        logits = self.fc(x) 
        return logits

