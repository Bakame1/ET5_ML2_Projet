import torch
import time
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


def main():
    print("--- Expérience 1 : Optimisation du paramètre C ---")

    # 1. Chargement des features extraites et sauvegardées précédemment
    print("Chargement des données...")
    try:
        X_train, y_train = torch.load('train_features_labels.pt')
        X_test, y_test = torch.load('test_features_labels.pt')
    except FileNotFoundError:
        print("Erreur : Fichiers .pt introuvables. Lancez d'abord VGG16Relu7.py")
        return

    # Conversion en tableaux NumPy pour scikit-learn
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    print(f"Dimension des features d'origine : {X_train_np.shape[1]}")

    # 2. Définition de la grille de paramètres à tester
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    svm_base = LinearSVC(multi_class='ovr', max_iter=5000)

    # Ajout de return_train_score=True pour pouvoir tracer la courbe d'entraînement
    grid_search = GridSearchCV(svm_base, param_grid, cv=3, n_jobs=-1, verbose=1, return_train_score=True)

    print("Recherche du meilleur paramètre C en cours...")
    t0 = time.time()
    grid_search.fit(X_train_np, y_train_np)
    t_grid = time.time() - t0

    # 3. Résultats texte
    best_c = grid_search.best_params_['C']
    print(f"\nTemps de recherche : {t_grid:.2f} secondes")
    print(f"Meilleur paramètre C trouvé : {best_c}")

    best_svm = grid_search.best_estimator_
    acc_best_c = best_svm.score(X_test_np, y_test_np)
    print(f"Accuracy sur le test set (final) avec le meilleur C : {acc_best_c:.4f}")

    # 4. Tracé des courbes d'apprentissage
    results = grid_search.cv_results_
    c_values = results['param_C'].data.astype(float)
    train_scores = results['mean_train_score']
    test_scores = results['mean_test_score']  # Correspond aux scores de validation croisée

    plt.figure(figsize=(10, 6))

    # On utilise semilogx car les valeurs de C sont espacées de manière logarithmique
    plt.semilogx(c_values, train_scores, label="Score d'entraînement", marker='o', linestyle='--', color='blue')
    plt.semilogx(c_values, test_scores, label="Score de validation (CV)", marker='s', linestyle='-', color='orange')

    plt.xlabel("Paramètre de régularisation C (échelle log)", fontsize=12)
    plt.ylabel("Précision (Accuracy)", fontsize=12)
    plt.title("Évolution des performances du SVM en fonction de C", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Sauvegarde de l'image pour l'intégrer dans le rapport LaTeX
    plt.tight_layout()
    save_path = "svm_c_tuning_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nCourbe sauvegardée sous : {save_path}")

    plt.show()


if __name__ == "__main__":
    main()