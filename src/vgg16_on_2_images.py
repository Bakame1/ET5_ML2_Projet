import torch
import torchvision
import pickle
import numpy as np
from PIL import Image

def predict_image_vgg16(image_path, classes_dict_path):
    """Charge une image, la normalise, et prédit sa classe avec VGG16."""

    # Chargement de l'image et du dictionnaire
    try:
        img = Image.open(image_path)
        with open(classes_dict_path, 'rb') as f:
            imagenet_classes = pickle.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {image_path} ou {classes_dict_path} est introuvable.")
        return

    # Prétraitement et Normalisation ImageNet
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose((2, 0, 1))

    mu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    sigma = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # Application de la normalisation Z-score
    img = (img - mu) / sigma

    # Chargement de VGG16 en mode évaluation
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.eval()

    # Forward pass
    img = np.expand_dims(img, 0)
    x = torch.Tensor(img)

    with torch.no_grad():
        y = vgg16(x)
        
    y = y.numpy()

    # Résultat
    predicted_class_idx = np.argmax(y)
    predicted_label = imagenet_classes[predicted_class_idx]
    
    print(f"---> L'image '{image_path}' a été classifiée comme : {predicted_label}")

if __name__ == "__main__":
    # On teste sur plusieurs images
    images_a_tester = ["../data/cat.jpg", "../data/dog.jpg"]
    
    for image in images_a_tester:
        predict_image_vgg16(image, "../data/imagenet_classes.pkl")