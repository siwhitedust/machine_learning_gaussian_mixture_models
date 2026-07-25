import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sumber_data = pd.read_csv("Datasets_Customer_Segmentation.csv")
df = pd.DataFrame(sumber_data)

# menghapus duplikasi pada datasets
df_hapus_duplikasi = df.drop_duplicates().reset_index(drop=True)
# print(f"Sebelum hapus duplikasi: {len(df)}", f"Setelah hapus duplikasi {len(df_hapus_duplikasi)}")

# pengisian (median) data numerik (Work_Experience dan Family_Size) yang hilang
for col in ["Work_Experience", "Family_Size"]:
    median_val = df_hapus_duplikasi[col].median()
    df_hapus_duplikasi[col] =  df_hapus_duplikasi[col].fillna(median_val)

# pengisian (modus) data kategorikal (Graduated, Ever_Married, Profession) yang hilang
for col in ["Graduated", "Ever_Married", "Profession"]:
    mode_val = df_hapus_duplikasi[col].mode()[0]
    df_hapus_duplikasi[col] = df_hapus_duplikasi[col].fillna(mode_val)

# menangani outlier "Work_Experience"
Q1 = df_hapus_duplikasi["Work_Experience"].quantile(0.25)
Q3 = df_hapus_duplikasi["Work_Experience"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

df_hapus_duplikasi["Work_Experience"] = np.where(
    df_hapus_duplikasi["Work_Experience"] > upper_bound,
    upper_bound,
    df_hapus_duplikasi["Work_Experience"]
)

# proses encoding akan dilakukan dengan 3 poin yaitu binary mapping, ordinal mapping, dan one-hot encoding
# binary mapping
df_hapus_duplikasi["Gender"] = df_hapus_duplikasi["Gender"].map({"Female": 0, "Male": 1})
df_hapus_duplikasi["Graduated"] = df_hapus_duplikasi["Graduated"].map({"No": 0, "Yes": 1})
df_hapus_duplikasi["Ever_Married"] = df_hapus_duplikasi["Ever_Married"].map({"No": 0, "Yes": 1})

# nominal mapping
penyebaran_map = {"Low": 0, "Average": 1, "High": 2}
df_hapus_duplikasi["Spending_Score"] = df_hapus_duplikasi["Spending_Score"].map(penyebaran_map)

# one-hot encoding
df_encoded = pd.get_dummies(df_hapus_duplikasi, columns=["Profession"], drop_first=False, dtype=int)

# standarisasi kolom numerik
kolom = ["Age", "Work_Experience", "Family_Size"]
scaler_std = StandardScaler()
df_encoded[kolom] = scaler_std.fit_transform(df_encoded[kolom])
df_encoded = df_encoded.drop_duplicates().reset_index(drop=True)

df_encoded.to_csv("Hasil_1.csv", index=False)
print("csv berhasil disimpan")