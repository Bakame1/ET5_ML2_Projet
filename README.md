# Projet d’apprentissage par transfert via l’extraction de caractéristiques à partir d’un CNN
**Polytech Paris-Saclay - ET5 - Machine Learning II**
**Auteurs :** Titouan BEAUVERGER, Marko BABIC, Morgan PHILIPPE


Ce projet met en œuvre l'apprentissage par transfert (Transfer Learning) en utilisant le réseau de neurones pré-entraîné VGG16 comme extracteur de caractéristiques. Ces caractéristiques sont ensuite utilisées par un classifieur SVM linéaire pour catégoriser les images du jeu de données 15-Scene.

### Pour activer l'environnement si besoin
```
.env\Scripts\activate
```

### Pour installer les librairies requises
```
pip install -r requirements.txt
```

## 🗂️ Arborescence du projet
```text
ET5_ML2_Projet/
│
├── data/                   # Dossier contenant les données manipulées (images de test, dataset 15-Scene)
├── src/                    # Dossier contenant le code source des scripts Python
│   ├── vgg16_on_2_images.py   # Test de VGG16 sur 2 images
│   └── ...                     
├── requirements.txt        # Liste des bibliothèques Python nécessaires
├── README.md               
└── Beauverger-Babic-Philippe.pdf # Rapport final du projet
```

