import pandas as pd
from sklearn.mixture import GaussianMixture

# baca file datasets
df = pd.read_csv("../ruang_data/hasil_preprocess_data.csv")

# inisiasi model dengan 4 parameter yakni jumlah klaster, penentuan seed awal, covaraince full memungkinkan setiap klaster memiliki bentuk yang berbeda, dan inisialisasi 10 kali
gmm = GaussianMixture(
    n_components=4, random_state=42, covariance_type="full", n_init=10
)
df["Hasil_Segmentasi"] = gmm.fit_predict(df)

# mapping untuk klaster segmentasi
cluster_map = {0: "A", 1:"B", 2:"C", 3:"D"}
df["Hasil_Segmentasi"] = df["Hasil_Segmentasi"].map(cluster_map)

# hitung probabilitas sebuah data masuk ke dalam klaster
probabilitas = gmm.predict_proba(df.drop(columns=["Hasil_Segmentasi"]))
for i in range(4):
    df[f"Klaster_Probabilitas{i}"] = probabilitas[:, i].round(2)

df.to_csv("Hasil_Segmentasi_GMM.csv", index=False)
print("HASIL BERHASIL DISIMPAN")