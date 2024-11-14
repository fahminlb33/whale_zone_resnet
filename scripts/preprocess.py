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

UNDERSAMPLE = False
SELECT_TOP = False

# load all
all_dfs = pd.concat(
    [
        pd.read_parquet("../dataset/africa.parquet").assign(country="africa"),
        pd.read_parquet("../dataset/australia.parquet").assign(country="australia"),
    ],
    ignore_index=True,
)

if UNDERSAMPLE:
    # determine lowest class
    min_samples = min(
        all_dfs[(all_dfs["target"] == 0) & (all_dfs["country"] == "africa")].shape[0],
        all_dfs[(all_dfs["target"] == 1) & (all_dfs["country"] == "africa")].shape[0],
        all_dfs[(all_dfs["target"] == 0) & (all_dfs["country"] == "australia")].shape[
            0
        ],
        all_dfs[(all_dfs["target"] == 1) & (all_dfs["country"] == "australia")].shape[
            0
        ],
    )

    # undersample
    sampled_dfs = [
        all_dfs[(all_dfs["target"] == 0) & (all_dfs["country"] == "africa")].sample(
            min_samples, random_state=42
        ),
        all_dfs[(all_dfs["target"] == 1) & (all_dfs["country"] == "africa")].sample(
            min_samples, random_state=42
        ),
        all_dfs[(all_dfs["target"] == 0) & (all_dfs["country"] == "australia")].sample(
            min_samples, random_state=42
        ),
        all_dfs[(all_dfs["target"] == 1) & (all_dfs["country"] == "australia")].sample(
            min_samples, random_state=42
        ),
    ]

    df_final = pd.concat(sampled_dfs, ignore_index=True).drop(columns=["country"])
else:
    df_final = all_dfs

# split train-test-validation
df_train, df_temp = train_test_split(
    df_final, test_size=0.4, random_state=42, stratify=df_final["target"]
)
df_test, df_validation = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp["target"]
)

print("Train:", len(df_train), "Validation:", len(df_validation), "Test:", len(df_test))
print("Train stats", df_train["target"].value_counts())
print("Validation stats", df_validation["target"].value_counts())
print("Test stats", df_test["target"].value_counts())

if UNDERSAMPLE:
    fsuffix = "-undersampled"
else:
    fsuffix = ""

if SELECT_TOP:
    df_train[features].to_parquet(f"../dataset/train-sel-10{fsuffix}.parquet")
    df_validation[features].to_parquet(f"../dataset/validation-sel-10{fsuffix}.parquet")
    df_test[features].to_parquet(f"../dataset/test-sel-10{fsuffix}.parquet")
else:
    df_train.to_parquet(f"../dataset/train-all{fsuffix}.parquet")
    df_validation.to_parquet(f"../dataset/validation-all{fsuffix}.parquet")
    df_test.to_parquet(f"../dataset/test-all{fsuffix}.parquet")
