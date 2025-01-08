
import os
import sys
from PIL import Image, UnidentifiedImageError

def remove_corrupted_images(directory):
    """
    Parcourt le dossier 'directory',
    tente d'ouvrir chaque fichier en image,
    et s'il est corrompu ou non reconnu, on le supprime.
    Retourne le nombre de fichiers supprimés.
    """
    count_removed = 0

    # Lister tous les fichiers du dossier
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        # Vérifier que c'est bien un fichier (pas un sous-dossier)
        if not os.path.isfile(file_path):
            continue

        # Tenter d'ouvrir le fichier en tant qu'image
        try:
            with Image.open(file_path) as im:
                im.load()  # force la lecture de l'image
        except (UnidentifiedImageError, OSError):
            # Si on a une erreur, c'est que le fichier n'est pas une image valide
            print(f"[INFO] Fichier corrompu ou non reconnu : {file_path}")
            os.remove(./Img/img_align_celeba/095025.jpg')  # on le supprime
            count_removed += 1

    return count_removed


if __name__ == '__main__':
    # On s'attend à recevoir un argument : le chemin du dossier d'images
   
    directory = "./Img/img_align_celeba"

    # Vérifier que le dossier existe
    if not os.path.isdir(directory):
        print(f"Erreur : {directory} n'est pas un dossier valide.")
        sys.exit(1)

    print(f"[INFO] Vérification du dossier : {directory}")
    removed = remove_corrupted_images(directory)
    print(f"[INFO] Nombre de fichiers supprimés : {removed}")

