import torch
import time
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


def main():
    print("--- Expérience 2 : Réduction de dimension avec PCA ---")

    # 1. Chargement des données
    print("Chargement des données...")
    try:
        X_train, y_train = torch.load('train_features_labels.pt')
        X_test, y_test = torch.load('test_features_labels.pt')
    except FileNotFoundError:
        print("Erreur : Fichiers .pt introuvables. Lancez d'abord VGG16Relu7.py")
        return

    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    print(f"\nDimensions d'origine : {X_train_np.shape[1]}")

    # ==========================================
    # ÉTAPE A : Évaluation SANS PCA (Baseline)
    # ==========================================
    print("\n--- 1. Évaluation SANS PCA (Modèle de base) ---")
    svm_baseline = LinearSVC(C=1.0, multi_class='ovr', max_iter=2000)

    t0_baseline = time.time()
    svm_baseline.fit(X_train_np, y_train_np)
    t_train_baseline = time.time() - t0_baseline

    acc_baseline = svm_baseline.score(X_test_np, y_test_np)

    print(f"Temps d'entraînement (sans PCA) : {t_train_baseline:.2f} secondes")
    print(f"Accuracy sur le test set (sans PCA) : {acc_baseline:.4f}")

    # ==========================================
    # ÉTAPE B : Évaluation AVEC PCA
    # ==========================================
    print("\n--- 2. Évaluation AVEC PCA ---")
    # On réduit de 4096 à 512 composantes
    n_components = 512
    pca = PCA(n_components=n_components)

    print(f"Calcul de la PCA pour réduire à {n_components} composantes...")
    t0_pca = time.time()
    X_train_pca = pca.fit_transform(X_train_np)
    X_test_pca = pca.transform(X_test_np)
    t_fit_pca = time.time() - t0_pca

    print(f"Temps d'exécution de la PCA : {t_fit_pca:.2f} secondes")
    variance_retenue = sum(pca.explained_variance_ratio_)
    print(f"Variance expliquée conservée : {variance_retenue:.2%}")

    svm_pca = LinearSVC(C=1.0, multi_class='ovr', max_iter=2000)

    t0_train_pca = time.time()
    svm_pca.fit(X_train_pca, y_train_np)
    t_train_pca = time.time() - t0_train_pca

    acc_pca = svm_pca.score(X_test_pca, y_test_np)

    print(f"Temps d'entraînement du SVM (avec PCA) : {t_train_pca:.2f} secondes")
    print(f"Accuracy sur le test set (avec PCA) : {acc_pca:.4f}")

    # ==========================================
    # ÉTAPE C : Bilan comparatif
    # ==========================================
    print("\n--- Bilan de la comparaison ---")
    temps_total_pca = t_fit_pca + t_train_pca
    gain_temps = t_train_baseline - temps_total_pca
    diff_acc = (acc_pca - acc_baseline) * 100

    print(f"Temps total (Sans PCA)  : {t_train_baseline:.2f} s")
    print(f"Temps total (Avec PCA)  : {temps_total_pca:.2f} s (dont {t_fit_pca:.2f}s de calcul PCA)")

    if gain_temps > 0:
        print(f"-> Gain de temps        : {gain_temps:.2f} s")
    else:
        print(f"-> Perte de temps       : {abs(gain_temps):.2f} s")

    print(f"Différence d'accuracy   : {diff_acc:+.2f} %")


if __name__ == "__main__":
    main()