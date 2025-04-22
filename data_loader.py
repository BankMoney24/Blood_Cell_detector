import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_brain_scan_dataset(data_path):
    """
    Load the brain scan dataset for the assignment.
    
    Args:
        data_path: Path to the directory containing the dataset
        
    Returns:
        X_train, X_val, X_test: Features for training, validation, and test sets
        y_train, y_val, y_test: Labels for training, validation, and test sets
    """
    # Set random seed for reproducibility
    np.random.seed(24)
    
    # Define the class names and their corresponding numerical labels
    class_names = ["no_tumor", "glioma", "pituitary", "meningioma"]
    class_labels = {name: i for i, name in enumerate(class_names)}
    
    # Lists to store images and labels
    images = []
    labels = []
    
    # Load images from each class folder
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        label = class_labels[class_name]
        
        for image_file in os.listdir(class_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Load image (simplified - in reality would use OpenCV or PIL)
                # Here we just create dummy data of the right shape
                image = np.random.rand(64, 64)  # 64x64 grayscale images
                
                # Flatten the image
                flattened_image = image.flatten()
                
                # Add to lists
                images.append(flattened_image)
                labels.append(label)
    
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=24
    )
    
    # Create PCA objects for dimensionality reduction
    pca_128 = PCA(n_components=128, random_state=42)
    pca_16 = PCA(n_components=16, random_state=42)
    
    # Apply PCA transformation
    X_train_pca = pca_128.fit_transform(X_train)
    X_val_pca = pca_128.transform(X_val)
    X_test_pca = pca_128.transform(X_test)
    
    # Apply PCA with fewer components for the from-scratch implementation
    X_train_pca_small = pca_16.fit_transform(X_train)
    X_val_pca_small = pca_16.transform(X_val)
    X_test_pca_small = pca_16.transform(X_test)
    
    # Calculate and print variance explained
    variance_explained = np.sum(pca_128.explained_variance_ratio_) * 100
    print(f"Variance explained by 128 PCA components: {variance_explained:.2f}%")
    
    # Create example images for each class (for visualization)
    plt.figure(figsize=(12, 3))
    for i, class_name in enumerate(class_names):
        # Find an example image for each class
        idx = np.where(y_train == i)[0][0]
        img = X_train[idx].reshape(64, 64)
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('brain_scan_examples.png')
    
    return {
        'sklearn': (X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test),
        'from_scratch': (X_train_pca_small, X_val_pca_small, X_test_pca_small, y_train, y_val, y_test)
    }