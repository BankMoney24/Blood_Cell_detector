import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlp_classifier_own import MLPClassifierOwn

def train_nn_own(X_train, X_val, X_test, y_train, y_val, y_test):
    np.random.seed(24)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    print("\nTraining model from scratch...")
    model = MLPClassifierOwn(hidden_layer_sizes=(16,), alpha=0, max_iter=5, random_state=42)
    model.fit(X_train_std, y_train)

    acc_train = model.score(X_train_std, y_train)
    acc_val = model.score(X_val_std, y_val)
    acc_test = model.score(X_test_std, y_test)

    print(f"Train: {acc_train:.4f} | Val: {acc_val:.4f} | Test: {acc_test:.4f}")

    alpha_vals = [0.01, 0.1]
    for alpha in alpha_vals:
        print(f"\nTraining with L2 regularization (alpha={alpha})")
        model_reg = MLPClassifierOwn(hidden_layer_sizes=(16,), alpha=alpha, max_iter=5, random_state=42)
        model_reg.fit(X_train_std, y_train)

        acc_train = model_reg.score(X_train_std, y_train)
        acc_val = model_reg.score(X_val_std, y_val)
        acc_test = model_reg.score(X_test_std, y_test)

        print(f"Alpha {alpha} - Train: {acc_train:.4f} | Val: {acc_val:.4f} | Test: {acc_test:.4f}")

    print("\nBinary classification (classes 0 and 1 only)")
    binary_mask_train = (y_train == 0) | (y_train == 1)
    binary_mask_val = (y_val == 0) | (y_val == 1)
    binary_mask_test = (y_test == 0) | (y_test == 1)

    X_train_bin = X_train_std[binary_mask_train]
    y_train_bin = y_train[binary_mask_train]
    X_val_bin = X_val_std[binary_mask_val]
    y_val_bin = y_val[binary_mask_val]
    X_test_bin = X_test_std[binary_mask_test]
    y_test_bin = y_test[binary_mask_test]

    model_bin = MLPClassifierOwn(hidden_layer_sizes=(16,), alpha=0.01, max_iter=5, random_state=42)
    model_bin.fit(X_train_bin, y_train_bin)

    acc_train = model_bin.score(X_train_bin, y_train_bin)
    acc_val = model_bin.score(X_val_bin, y_val_bin)
    acc_test = model_bin.score(X_test_bin, y_test_bin)

    print(f"Binary - Train: {acc_train:.4f} | Val: {acc_val:.4f} | Test: {acc_test:.4f}")

    return model
