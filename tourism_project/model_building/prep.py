import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop columns that are not useful for modeling: the row-export index and
# the unique customer identifier
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Fix a known data-entry typo in Gender ("Fe Male" -> "Female")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Define target variable
target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform a stratified train-test split so both classes are represented
# proportionally in train and test (ProdTaken is an imbalanced target)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
