import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn_classification_sklearn import train_nn_sklearn
from nn_classification_from_scratch import train_nn_own

def main():
    np.random.seed(24)
    
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=24)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pca_full = PCA(n_components=128, random_state=42)
    X_train_pca = pca_full.fit_transform(X_train_scaled)
    X_val_pca = pca_full.transform(X_val_scaled)
    X_test_pca = pca_full.transform(X_test_scaled)
    
    print("Running sklearn-based neural network...")
    train_nn_sklearn(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)
    
    print("Running custom neural network from scratch...")
    pca_small = PCA(n_components=16, random_state=42)
    X_train_small = pca_small.fit_transform(X_train_scaled)
    X_val_small = pca_small.transform(X_val_scaled)
    X_test_small = pca_small.transform(X_test_scaled)

    train_nn_own(X_train_small, X_val_small, X_test_small, y_train, y_val, y_test)

if __name__ == "__main__":
    main()
