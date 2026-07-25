import pandas as pd

sumber = pd.read_csv("Hasil_1.csv")
df = pd.DataFrame(sumber)

print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())