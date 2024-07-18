import pandas as pd
from sklearn.model_selection import train_test_split

features = [
    "pbo_sum",
    "pbo_mean",
    "tob_mean",
    "sob_sum",
    "tob_sum",
    "sob_mean",
    "fe_mean",
    "fe_sum",
    "so_sum",
    "po4_mean",
    "target",
]

non_feature = ["zone_id", "ts"]

df = pd.concat([
  pd.read_parquet("../dataset/zonal/africa.parquet")[features], #.drop(columns=non_feature),
  pd.read_parquet("../dataset/zonal/australia.parquet")[features], #.drop(columns=non_feature),
], ignore_index=True)

df.info()

df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
df_test, df_validation = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp["target"])

print("Train:", len(df_train), "Validation:", len(df_validation), "Test:", len(df_test))
print("Train stats", df_train["target"].value_counts())
print("Validation stats", df_validation["target"].value_counts())
print("Test stats", df_test["target"].value_counts())

df_train.to_parquet("../dataset/train-sel-10.parquet")
df_validation.to_parquet("../dataset/validation-sel-10.parquet")
df_test.to_parquet("../dataset/test-sel-10.parquet")
