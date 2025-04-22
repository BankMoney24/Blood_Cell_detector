import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our modules
from nn_classification_sklearn import train_nn_sklearn
from nn_classification_from_scratch import train_nn_own

def main():
    # Set random seed for reproducibility
    np.random.seed(24)
    
    # Load brain scan dataset - replaced with digits for this implementation
    # In a real scenario, you would load your actual dataset
    X, y = load_digits(return_X_y=True)
    
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=24
    )
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=128, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)
    
    # Task 1: Train neural network using sklearn
    print("Training neural network using sklearn...")
    train_nn_sklearn(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)
    
    # Task 2: Train neural network from scratch
    print("Training neural network from scratch...")
    # Use fewer PCA components for the from-scratch implementation to speed up training
    pca_small = PCA(n_components=16, random_state=42)
    X_train_pca_small = pca_small.fit_transform(X_train_std)
    X_val_pca_small = pca_small.transform(X_val_std)
    X_test_pca_small = pca_small.transform(X_test_std)
    
    train_nn_own(X_train_pca_small, X_val_pca_small, X_test_pca_small, y_train, y_val, y_test)

if __name__ == "__main__":
    main()