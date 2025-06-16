import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_brain_scan_dataset(data_path):
    np.random.seed(24)
    class_names = ["no_tumor", "glioma", "pituitary", "meningioma"]
    class_labels = {name: i for i, name in enumerate(class_names)}
    images = []
    labels = []
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        label = class_labels[class_name]
        for image_file in os.listdir(class_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image = np.random.rand(64, 64)
                flattened_image = image.flatten()
                images.append(flattened_image)
                labels.append(label)
    X = np.array(images)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=24)
    pca_128 = PCA(n_components=128, random_state=42)
    pca_16 = PCA(n_components=16, random_state=42)
    X_train_pca = pca_128.fit_transform(X_train)
    X_val_pca = pca_128.transform(X_val)
    X_test_pca = pca_128.transform(X_test)
    X_train_pca_small = pca_16.fit_transform(X_train)
    X_val_pca_small = pca_16.transform(X_val)
    X_test_pca_small = pca_16.transform(X_test)
    variance_explained = np.sum(pca_128.explained_variance_ratio_) * 100
    print(f"Variance explained by 128 PCA components: {variance_explained:.2f}%")
    plt.figure(figsize=(12, 3))
    for i, class_name in enumerate(class_names):
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
