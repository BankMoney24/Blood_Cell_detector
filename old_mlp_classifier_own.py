import numpy as np
import random
from autodiff.scalar import Scalar
from autodiff.neural_net import MultiLayerPerceptron
import math

def relu(x):
    return Scalar(max(0.0, x.data)) if x.data < 0 else x

class MLPClassifierOwn:
    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.num_classes = None
        self.classes_ = None

    def fit(self, X, y, batch_size=32):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        layer_dims = [n_features] + list(self.hidden_layer_sizes) + [self.num_classes]
        self.model = MultiLayerPerceptron(layer_dims, activation=relu, random_state=self.random_state)
        y_idx = np.zeros(n_samples, dtype=int)
        for i, cls in enumerate(self.classes_):
            y_idx[y == cls] = i

        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y_idx[batch_indices]
                loss = Scalar(0.0)
                for i, x in enumerate(X_batch):
                    logits = self.model(x)
                    if self.num_classes == 2:
                        y_pred = self.sigmoid(logits[0])
                        sample_loss = self.binary_cross_entropy_loss(y_pred, y_batch[i])
                    else:
                        y_pred = self.softmax(logits)
                        sample_loss = self.multiclass_cross_entropy_loss(y_pred, y_batch[i])
                    loss = loss + sample_loss
                reg_loss = self.l2_regularization_term(batch_size)
                if reg_loss.data != 0:
                    loss = loss + reg_loss
                loss.backward()
                lr = 0.01
                for param in self.model.parameters():
                    param.data -= lr * param.grad
                    param.grad = 0.0
            print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss.data}")
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        for i, x in enumerate(X):
            logits = self.model(x)
            if self.num_classes == 2:
                y_pred[i] = 1 if self.sigmoid(logits[0]).data > 0.5 else 0
            else:
                y_pred[i] = np.argmax([logit.data for logit in logits])
        return self.classes_[y_pred]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def softmax(self, logits):
        max_logit = max([logit.data for logit in logits])
        exp_logits = [Scalar(math.exp(logit.data - max_logit)) for logit in logits]
        sum_exp = sum([exp_logit.data for exp_logit in exp_logits])
        return [exp_logit / Scalar(sum_exp) for exp_logit in exp_logits]

    def multiclass_cross_entropy_loss(self, y_pred, y_true):
        true_prob = y_pred[y_true]
        return -Scalar(math.log(true_prob.data))

    def sigmoid(self, x):
        if x.data > 0:
            z = Scalar(math.exp(-x.data))
            return Scalar(1.0) / (Scalar(1.0) + z)
        else:
            z = Scalar(math.exp(x.data))
            return z / (Scalar(1.0) + z)

    def binary_cross_entropy_loss(self, y_pred, y_true):
        y_true_scalar = Scalar(float(y_true))
        eps = 1e-15
        y_pred_safe = Scalar(max(min(y_pred.data, 1.0 - eps), eps))
        term1 = y_true_scalar * Scalar(math.log(y_pred_safe.data))
        term2 = (Scalar(1.0) - y_true_scalar) * Scalar(math.log(1.0 - y_pred_safe.data))
        return -(term1 + term2)

    def l2_regularization_term(self, batch_size):
        if self.alpha == 0:
            return Scalar(0.0)
        param_sum_squared = Scalar(0.0)
        for param in self.model.parameters():
            param_sum_squared = param_sum_squared + param * param
        return Scalar(self.alpha / (2 * batch_size)) * param_sum_squared
