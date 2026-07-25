import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

df = pd.read_csv("../ruang_data/hasil_preprocess_data.csv")

# tentukan grid parameter yang akan di uji
param_grid = {
    "n_components": [3, 4, 5],
    "covariance_type": ["full", "tied", "diag", "spherical"],
    "init_params": ["kmeans", "k-means++", "random"],
    "n_init": [10]
}

result = []

for n in param_grid['n_components']:
    for cov in param_grid["covariance_type"]:
        for init in param_grid["init_params"]:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=cov,
                init_params=init,
                n_init=10,
                random_state=42
            )

            labels = gmm.fit_predict(df)

            bic = gmm.bic(df)
            sil = silhouette_score(df, labels)

            result.append(
                {'n_components': n,
                'covariance_type': cov,
                'init_params': init,
                'BIC_Score': round(bic, 2),
                'Silhouette_Score': round(sil, 4)}
            )

# tampilkan 10 kombinasi terbaik dengan bic terendah
result_df = pd.DataFrame(result).sort_values(by="BIC_Score", ascending=True)

print('=== 10 KOMBINASI PARAMETER GMM TERBAIK ===')
print(result_df.head(10).to_string(index=False))