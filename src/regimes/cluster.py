import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    os.makedirs("outputs/regimes", exist_ok=True)

    embeddings = np.load("outputs/regimes/test_embeddings.npy")

    n_regimes = 3
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(embeddings)

    sil_score = silhouette_score(embeddings, regime_labels)

    np.save("outputs/regimes/test_regime_labels.npy", regime_labels)
    np.save("outputs/regimes/test_cluster_centers.npy", kmeans.cluster_centers_)

    print("Clustering complete.")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of regimes: {n_regimes}")
    print(f"Silhouette Score: {sil_score:.4f}")

    unique, counts = np.unique(regime_labels, return_counts=True)
    print("Regime counts:")
    for u, c in zip(unique, counts):
        print(f"  Regime {u}: {c}")


if __name__ == "__main__":
    main()