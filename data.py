import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms

def custom_collate(batch):
    """
    Filtre les éléments (None, None) qui peuvent survenir 
    si un fichier image est corrompu ou introuvable.
    Renvoie un batch valide ou (None, None) si tout est invalide.
    """
    # Filtrer les (None, None)
    batch_filtered = [(img, lab) for (img, lab) in batch if img is not None]

    if len(batch_filtered) == 0:
        # Tous les éléments du batch étaient None => on renvoie (None, None)
        # (PyTorch ignore alors ce batch et passe au suivant)
        return None, None

    images, labels = zip(*batch_filtered)  # sépare en deux listes
    images = torch.stack(images, dim=0)    # [B,3,128,128]
    labels = torch.stack(labels, dim=0)    # [B,40]
    return images, labels


class CelebADataset(Dataset):
    def __init__(self, data_dir, attr_file, eval_partition, partition='train', fraction=1.0):
        """
        data_dir     : dossier avec les images alignées (ex: "./Img/img_align_celeba")
        attr_file    : chemin vers "list_attr_celeba.txt"
        eval_partition : chemin vers "list_eval_partition.txt"
        partition    : 'train', 'val' ou 'test'
        fraction     : fraction du dataset à garder (0 < fraction <= 1).
                       ex: 0.05 -> 5% des données.
        """
        self.data_dir = data_dir
        self.partition = partition
        self.fraction = fraction

        # 1) Lecture de la répartition (train=0, val=1, test=2)
        partition_map = {}
        with open(eval_partition, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, part_id = line.split()
                partition_map[img_name] = int(part_id)

        if partition == 'train':
            desired_id = 0
        elif partition == 'val':
            desired_id = 1
        elif partition == 'test':
            desired_id = 2
        else:
            raise ValueError("partition must be 'train', 'val' or 'test'.")

        # 2) Lecture des attributs
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        # lines[0] = nb_images
        # lines[1] = noms des 40 attributs
        lines = lines[2:]  # le reste: "000001.jpg -1 1 ..."

        attr_map = {}
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0]
            vals = list(map(int, parts[1:]))  # 40 attributs
            attr_map[img_name] = vals

        # 3) Construction de la liste self.imgs et self.labels
        self.imgs = []
        self.labels = []
        for img_name, pid in partition_map.items():
            if pid == desired_id:
                if img_name in attr_map:
                    self.imgs.append(img_name)
                    self.labels.append(attr_map[img_name])

        # 4) Tronquer à 'fraction' (si < 1.0)
        if self.fraction < 1.0:
            combined = list(zip(self.imgs, self.labels))
            random.shuffle(combined)  # mélange aléatoire
            keep_count = int(len(combined) * self.fraction)
            combined = combined[:keep_count]
            self.imgs, self.labels = zip(*combined)
            self.imgs = list(self.imgs)
            self.labels = list(self.labels)

        # 5) Transformations
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Si on ne parvient pas à ouvrir une image (fichier introuvable, corrompu, etc.),
        on renvoie (None, None). Le 'custom_collate' filtrera ces éléments.
        """
        img_name = self.imgs[idx]
        label_40 = self.labels[idx]  # tableau de 40 valeurs +1/-1

        img_path = os.path.join(self.data_dir, img_name)

        try:
            # Tenter d'ouvrir le fichier
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            # Fichier introuvable, corrompu, ou non reconnu comme image
            print(f"[WARN] Impossible d'ouvrir '{img_path}' : {e}")
            return None, None

        # Appliquer la transformation
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label_40, dtype=torch.float)
        return image_tensor, label_tensor


def get_celeba_dataloader(data_dir, attr_file, eval_partition,
                          partition='train', batch_size=32, shuffle=True, fraction=1.0):
    """
    Crée un DataLoader pour CelebA,
    en filtrant les images illisibles via 'custom_collate'.

    fraction : si < 1.0, on garde seulement une fraction (ex: 0.05 => 5%)
               du dataset, pratique pour des tests rapides.
    """
    dataset = CelebADataset(
        data_dir=data_dir,
        attr_file=attr_file,
        eval_partition=eval_partition,
        partition=partition,
        fraction=fraction
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=custom_collate  # <-- on utilise notre collate perso
    )
    return loader
