//nn_classification_from_scratch.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlp_classifier_own import MLPClassifierOwn

def train_nn_own(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train neural networks using our own implementation.
    
    Args:
        X_train: PCA-transformed training data
        X_val: PCA-transformed validation data
        X_test: PCA-transformed test data
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
    """
    # Set random seed for reproducibility
    # as in the project requirement
    np.random.seed(24)
    
    # Standardize the PCA-projected data
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    # Task 2.5: Create and train an MLPClassifierOwn
    print("\nTraining neural network from scratch...")
    
    # Create a classifier with specified parameters
    clf_own = MLPClassifierOwn(
        hidden_layer_sizes=(16,),  # Single hidden layer with 16 neurons
        alpha=0,  # No L2 regularization
        max_iter=5,  # 5 epochs
        random_state=42
    )
    
    # Train the classifier
    clf_own.fit(X_train_std, y_train)
    
    # Evaluate the model
    train_acc = clf_own.score(X_train_std, y_train)
    val_acc = clf_own.score(X_val_std, y_val)
    test_acc = clf_own.score(X_test_std, y_test)
    
    print(f"From scratch - Train accuracy: {train_acc:.4f}")
    print(f"From scratch - Validation accuracy: {val_acc:.4f}")
    print(f"From scratch - Test accuracy: {test_acc:.4f}")
    
    # Task 2.6: Train the network with L2 regularization
    # Try two different alpha values
    #necessary so as to be on point
    alpha_values = [0.01, 0.1]
    
    for alpha in alpha_values:
        print(f"\nTraining with L2 regularization (alpha={alpha})...")
        
        # Create a classifier with L2 regularization
        clf_own_reg = MLPClassifierOwn(
            hidden_layer_sizes=(16,),
            alpha=alpha,
            max_iter=5,
            random_state=42
        )
        
        # Train the classifier
        clf_own_reg.fit(X_train_std, y_train)
        
        # Evaluate the model
        train_acc = clf_own_reg.score(X_train_std, y_train)
        val_acc = clf_own_reg.score(X_val_std, y_val)
        test_acc = clf_own_reg.score(X_test_std, y_test)
        
        print(f"From scratch (alpha={alpha}) - Train accuracy: {train_acc:.4f}")
        print(f"From scratch (alpha={alpha}) - Validation accuracy: {val_acc:.4f}")
        print(f"From scratch (alpha={alpha}) - Test accuracy: {test_acc:.4f}")
    
    # Task 3: Binary Classification
    print("\nPerforming Binary Classification...")
    
    # Extract classes 0 and 1 from the dataset
    binary_mask_train = (y_train == 0) | (y_train == 1)
    binary_mask_val = (y_val == 0) | (y_val == 1)
    binary_mask_test = (y_test == 0) | (y_test == 1)
    
    X_train_binary = X_train_std[binary_mask_train]
    y_train_binary = y_train[binary_mask_train]
    X_val_binary = X_val_std[binary_mask_val]
    y_val_binary = y_val[binary_mask_val]
    X_test_binary = X_test_std[binary_mask_test]
    y_test_binary = y_test[binary_mask_test]
    
    # Create and train a binary classifier
    clf_binary = MLPClassifierOwn(
        hidden_layer_sizes=(16,),
        alpha=0.01,
        max_iter=5,
        random_state=42
    )
    
    # Train the classifier
    clf_binary.fit(X_train_binary, y_train_binary)
    
    # Evaluate the model
    train_acc = clf_binary.score(X_train_binary, y_train_binary)
    val_acc = clf_binary.score(X_val_binary, y_val_binary)
    test_acc = clf_binary.score(X_test_binary, y_test_binary)
    
    print(f"Binary Classification - Train accuracy: {train_acc:.4f}")
    print(f"Binary Classification - Validation accuracy: {val_acc:.4f}")
    print(f"Binary Classification - Test accuracy: {test_acc:.4f}")
    
    return clf_own