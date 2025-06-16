import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def train_nn_sklearn(X_train, X_val, X_test, y_train, y_val, y_test):
    np.random.seed(24)

    layer_configs = [(2,), (8,), (64,), (256,), (1024,), (128, 256, 128)]
    basic_results = []

    print("\nTraining models with various hidden layer sizes:")
    for layers in layer_configs:
        model = MLPClassifier(hidden_layer_sizes=layers, max_iter=100, solver='adam', random_state=1)
        model.fit(X_train, y_train)
        acc_train = model.score(X_train, y_train)
        acc_val = model.score(X_val, y_val)
        loss = model.loss_
        basic_results.append({'layers': layers, 'train_acc': acc_train, 'val_acc': acc_val, 'loss': loss})
        print(f"Layers {layers}: Train={acc_train:.4f}, Val={acc_val:.4f}, Loss={loss:.4f}")

    print("\nTraining with regularization:")
    reg_mode = 'a'
    reg_results = []

    for layers in layer_configs:
        if reg_mode == 'a':
            model = MLPClassifier(hidden_layer_sizes=layers, max_iter=100, solver='adam', random_state=1, alpha=0.1)
        elif reg_mode == 'b':
            model = MLPClassifier(hidden_layer_sizes=layers, max_iter=100, solver='adam', random_state=1, early_stopping=True)
        else:
            model = MLPClassifier(hidden_layer_sizes=layers, max_iter=100, solver='adam', random_state=1, alpha=0.1, early_stopping=True)

        model.fit(X_train, y_train)
        acc_train = model.score(X_train, y_train)
        acc_val = model.score(X_val, y_val)
        loss = model.loss_
        reg_results.append({'layers': layers, 'train_acc': acc_train, 'val_acc': acc_val, 'loss': loss})
        print(f"Layers {layers}: Train={acc_train:.4f}, Val={acc_val:.4f}, Loss={loss:.4f}")

    best_model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, solver='adam', random_state=1, alpha=0.1)
    best_model.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plt.plot(best_model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png')

    param_grid = {
        'alpha': [0.0, 0.1, 1.0],
        'batch_size': [32, 512],
        'hidden_layer_sizes': [(128,), (256,)]
    }

    total_configs = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal model configurations: {total_configs}")

    print("\nRunning GridSearchCV...")
    grid = GridSearchCV(
        MLPClassifier(max_iter=100, solver='adam', random_state=42),
        param_grid=param_grid,
        cv=5,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    final_model = grid.best_estimator_
    test_acc = final_model.score(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    display_eval_report(final_model, X_test, y_test)
    return final_model

def display_eval_report(model, X, y_true):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
