import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def visualize_first_layer_activations(image_path, save_path):
    """Passe l'image dans la 1ère couche de VGG16 et sauvegarde une grille d'activations."""
    
    img = Image.open(image_path)
    img_resized = img.resize((224, 224), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_arr = img_arr.transpose((2, 0, 1))

    mu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    sigma = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_arr = (img_arr - mu) / sigma

    x = torch.Tensor(np.expand_dims(img_arr, 0))

    # Chargement du modèle
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.eval()
    
    # On isole uniquement la toute première couche de convolution (Conv2d)
    first_conv_layer = vgg16.features[0]

    # Forward pass uniquement sur cette première couche
    with torch.no_grad():
        activations = first_conv_layer(x) # Format de sortie : [1, 64, 224, 224]
    
    # On retire la dimension du batch pour obtenir [64, 224, 224]
    activations = activations[0].numpy()

    # Visualisation des 6 premières cartes d'activation
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Cartes d'activation de la 1ère couche de VGG16", fontsize=16)

    for i, ax in enumerate(axes.flat):
        # On affiche la i-ème carte de caractéristiques
        feature_map = activations[i]
        
        # Affichage en nuances de gris (standard pour les activations)
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Filtre n°{i+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure sauvegardée sous : {save_path}")
    plt.show()

if __name__ == "__main__":
    # Test sur le chat et le chien
    visualize_first_layer_activations("../data/cat.jpg", "../data/first_layer_activations/activations_cat.png")
    visualize_first_layer_activations("../data/dog.jpg", "../data/first_layer_activations/activations_dog.png")