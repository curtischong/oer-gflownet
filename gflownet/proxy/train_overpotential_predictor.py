from sklearn.model_selection import train_test_split
import polars as pl
import lightgbm as lgb
import numpy as np
from gflownet.utils.seed import seed_everything


# TODO: do kfold
seed_everything(81293)

# Load CSV data using Polars
file_path = "data.csv"
data = pl.read_csv(file_path)

# Convert Polars DataFrame to NumPy arrays
X = data.drop("overpotential").to_numpy()
y = data["overpotential"].to_numpy()

# Split into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.shape)

# Convert datasets into LightGBM format
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# these params were modified from a kaggle competition I participated in
params = {
    # 'boosting_type': 'gbdt',
    "metric": "rmse",
    # 'reg_alpha': 0.003188447814669599,
    # 'reg_lambda': 0.0010228604507564066,
    # 'colsample_bytree': 0.5420247656839267,
    # 'subsample': 0.9778252382803456,
    # 'feature_fraction': 0.8,
    # 'bagging_freq': 1,
    # 'bagging_fraction': 0.75,
    "learning_rate": 0.01716485155812008,
    # "num_leaves": 8,
    "min_data_in_leaf": 2,
    # "min_child_samples": 46,
    # "verbosity": -1,
    # "random_state": SEED,
    # "n_estimators": 500,
    # "device_type": "cpu",
    "objective": "regression",
}


# Train the model
num_round = 100
bst = lgb.train(
    params,
    train_data,
    num_boost_round=num_round,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
)

# Predict on validation data
y_pred = bst.predict(X_val)
err = np.mean(np.square(y_val - y_pred))
print(f"Validation err: {err:.8f}")

# Save the model
bst.save_model("lightgbm_model.txt")
