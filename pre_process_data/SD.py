import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Datasets_Customer_Segmentation.csv")
df = pd.DataFrame(data)

# melihat informasi detail dari datasets
info_datasets = df.info()

# mendeteksi data hilang
data_hilang = df.isnull().sum()
# print(data_hilang)

# mendeteksi duplikasi
data_duplikat = df.duplicated().sum()
# print("Data duplikat berjumlah: ", data_duplikat)

# statitik deskriptif data numerik
sd_numerik = df.describe().T
kemiringan_numerik = df[["Age", "Work_Experience", "Family_Size"]].skew()
print(sd_numerik, "\n\nKemiringan\n", kemiringan_numerik)

# statistik deskriptif data kategorikal
sd_kategorikal = df.describe(include=["str"]).T
distribusi_frekuensi_spending_score = df["Spending_Score"].value_counts()
distribusi_frekuensi_profesi = df["Profession"].value_counts()
# print(sd_kategorikal, "\n"*2, distribusi_frekuensi_profesi, "\n"*2, distribusi_frekuensi_spending_score)

# group by spending
group_by_spending = df.groupby("Spending_Score")[["Age", "Work_Experience", "Family_Size"]].mean()
# print(group_by_spending)