import os
import shutil
import random

# Ce script organise les images d'un dataset en deux dossiers : 'train' et 'test'. 
# Les donnees doivent etre dans le dossier 'data/15-Scene' avec une structure de sous-dossiers pour chaque classe (ex: 'data/15-Scene/bedroom', 'data/15-Scene/kitchen', etc.).
def organize_dataset(source_root, train_ratio=0.2):
    train_dir = os.path.join(source_root, 'train')
    test_dir = os.path.join(source_root, 'test')
    
    # Vérification de l'existence des dossiers 'train' et 'test'
    if (os.path.exists(train_dir) and os.path.exists(test_dir)):
        if (len(os.listdir(train_dir)) > 0 or len(os.listdir(test_dir)) > 0):
            return
    # Verficiation de l'existence du dossier source
    if not os.path.exists(source_root):
        print(f"Le dossier source '{source_root}' n'existe pas. Veuillez vérifier le chemin.")
        return

    classes = [d for d in os.listdir(source_root) 
               if os.path.isdir(os.path.join(source_root, d)) and d not in ['train', 'test']]

    for cls in classes:
        cls_path = os.path.join(source_root, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in test_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
        
        print(f"Classe {cls} traitée : {len(train_images)} images vers train, {len(test_images)} vers test.")

    print("\nOrganisation terminée !")

source_path = 'data/15-Scene'
organize_dataset(source_path)