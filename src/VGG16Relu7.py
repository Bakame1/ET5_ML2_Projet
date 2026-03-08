from torch import nn
import torch
from sklearn.svm import LinearSVC
from torchvision import transforms, datasets, models 

vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
vgg16_base = models.vgg16(weights=vgg_weights)
vgg16_base.eval()

transform_15scene = transforms.Compose([
    transforms.Resize((224, 224)),      # Redimensionnement
    transforms.Grayscale(num_output_channels=3), # Adaptation Noir et Blanc (duplication sur les 3 canaux)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation 
])


def loadData():
    from torchvision import datasets
    from torch.utils.data import DataLoader

    #Chargement des datasets avec les transformations
    train_dataset = datasets.ImageFolder(root='data/15-Scene/train', transform=transform_15scene)
    test_dataset = datasets.ImageFolder(root='data/15-Scene/test', transform=transform_15scene)

    # Création des loaders 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, test_loader

def extract_features(loader, model, device):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device) 
            outputs = model(inputs)
            outputsL2 = torch.nn.functional.normalize(outputs, p=2, dim=1)  # Normalisation L2
            features.append(outputsL2.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)

class VGG16relu7(nn.Module):
    def __init__(self):
        super(VGG16relu7, self).__init__()
        # Copy the entire convolutional part
        self.features = nn.Sequential( *list(vgg16_base.features.children()))
        # Keep a piece of the classifier: -2 to stop at relu7
        self.classifier = nn.Sequential(*list(vgg16_base.classifier.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")
    # Charger les données
    train_loader, test_loader = loadData()
    

    # Initialiser le modèle
    model = VGG16relu7().to(device) # Envoyer le modèle sur GPU
    model.eval()
    # Le modele SVM
    svm = LinearSVC(C=1.0, multi_class='ovr') # ovr pour One-vs-Rest

    # Extraire les caractéristiques pour les ensembles d'entraînement et de test
    X_train, y_train = extract_features(train_loader, model, device)
    X_test, y_test = extract_features(test_loader, model, device)

    # Entraîner le SVM
    svm.fit(X_train.numpy(), y_train.numpy())
    accuracy = svm.score(X_test.numpy(), y_test.numpy())
    print(f'Accuracy: {accuracy:.4f}')

    # Sauvegarder les caractéristiques et les étiquettes
    torch.save((X_train, y_train), 'train_features_labels.pt')
    torch.save((X_test, y_test), 'test_features_labels.pt')