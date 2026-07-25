import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Datasets_Customer_Segmentation.csv")
df = pd.DataFrame(data)

# VISUALISASI
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.boxplot(data=df, y="Age", ax=axes[0], color="skyblue")
axes[0].set_title("Boxplot_Age")

sns.boxplot(data=df, y="Work_Experience", ax=axes[1], color="lightgreen")
axes[1].set_title("Boxplot_Work_Experience")

sns.boxplot(data=df, y="Family_Size", ax=axes[2], color="salmon")
axes[2].set_title("Boxplot_Family_Size")

plt.tight_layout()
plt.show()