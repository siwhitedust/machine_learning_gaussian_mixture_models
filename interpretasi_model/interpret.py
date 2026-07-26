import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("../pembuatan_model/Hasil_Segmentasi_GMM.csv")

# pisahkan fitur input dan target
drop_cols = [
    c for c in df.columns if "Hasil_Segmentasi" in c or "Klaster_Probabilitas" in c or "Segmen" in c
]
x = df.drop(columns=drop_cols)
y = df["Hasil_Segmentasi"]

# membuat random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x, y)

# hitung feature importance
feat_imp = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)

# hitung permutation importance
perm_imp = permutation_importance(rf, x, y, n_repeats=10, random_state=42)
perm_imp_series = pd.Series(perm_imp.importances_mean, index=x.columns).sort_values(ascending=False)

# visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[0])
axes[0].set_title("Feature Importance")
axes[0].set_xlabel("Nilai Importance")

sns.barplot(x=perm_imp_series.values, y=perm_imp_series.index, ax=axes[1])
axes[1].set_title("Permutation Importance")
axes[1].set_xlabel("Nilai Penurunan Akurasi")

plt.tight_layout()
plt.show()