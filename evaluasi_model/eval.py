import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

df = pd.read_csv("../ruang_data/hasil_preprocess_data.csv")

gmm = GaussianMixture(
    n_components=4, random_state=42, covariance_type="full", n_init=10, init_params='kmeans'
)
labels = gmm.fit_predict(df)

bic = gmm.bic(df)
aic = gmm.aic(df)
sil = silhouette_score(df, labels)
ch = calinski_harabasz_score(df, labels)
db = davies_bouldin_score(df, labels)

print('=== HASIL EVALUASI MODEL GMM (CLUSTERING) ===')
print(f'BIC Score               : {bic:,.2f}')
print(f'AIC Score               : {aic:,.2f}')
print(f'Silhouette Score        : {sil:.4f}')
print(f'Calinski-Harabasz Index : {ch:.2f}')
print(f'Davies-Bouldin Index    : {db:.4f}')