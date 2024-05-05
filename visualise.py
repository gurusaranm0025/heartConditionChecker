from config import DATASET_PATH_FOR_LINUX
import os
import pandas as pd
script_dir = os.path.dirname(__file__)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(f"{script_dir}/{DATASET_PATH_FOR_LINUX}")

disease_counts = data["HeartDisease"].value_counts()
temp_df = pd.DataFrame({
	"Disease": disease_counts.index,
	"Counts": disease_counts.values
})

plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()