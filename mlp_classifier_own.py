import numpy as np
import random
from autodiff.scalar import Scalar
from autodiff.neural_net import MultiLayerPerceptron
import math


def relu(x):
    """ReLU activation function."""
    return Scalar(max(0.0, x.data)) if x.data < 0 else x


class MLPClassifierOwn:
    """Multi-layer Perceptron classifier implemented from scratch.

    Attributes:
        hidden_layer_sizes: The number of neurons in each hidden layer.
        alpha: L2 regularization parameter.
        max_iter: Maximum number of iterations.
        random_state: Random state for reproducibility.
        model: The neural network model.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=200,
        random_state=None,
    ):
        """Initialize the classifier.

        Args:
            hidden_layer_sizes: The number of neurons in each hidden layer.
            alpha: L2 regularization parameter.
            max_iter: Maximum number of iterations.
            random_state: Random state for reproducibility.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.num_classes = None
        self.classes_ = None

    def fit(self, X, y, batch_size=32):
        """Fit the model to the data.

        Args:
            X: The training data.
            y: The target values.
            batch_size: The batch size for stochastic gradient descent.
        """
        # set seed for reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Get the number of features
        n_samples, n_features = X.shape

        # Get the unique classes
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)

        # Initialize the model
        layer_dims = [n_features] + list(self.hidden_layer_sizes) + [self.num_classes]
        self.model = MultiLayerPerceptron(
            layer_dims, activation=relu, random_state=self.random_state
        )

        # Convert labels to indices
        y_idx = np.zeros(n_samples, dtype=int)
        for i, cls in enumerate(self.classes_):
            y_idx[y == cls] = i

        # Training loop
        for epoch in range(self.max_iter):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y_idx[batch_indices]

                # For each sample in the batch
                loss = Scalar(0.0)
                for i, x in enumerate(X_batch):
                    # Forward pass
                    logits = self.model(x)

                    # Calculate the loss
                    if self.num_classes == 2:
                        # Binary classification
                        # Apply sigmoid and compute binary cross-entropy loss
                        y_pred = self.sigmoid(logits[0])
                        sample_loss = self.binary_cross_entropy_loss(y_pred, y_batch[i])
                    else:
                        # Multi-class classification
                        # Apply softmax and compute cross-entropy loss
                        y_pred = self.softmax(logits)
                        sample_loss = self.multiclass_cross_entropy_loss(
                            y_pred, y_batch[i]
                        )

                    loss = loss + sample_loss

                # Add regularization term
                reg_loss = self.l2_regularization_term(batch_size)
                if reg_loss.data != 0:
                    loss = loss + reg_loss

                # Compute gradients
                loss.backward()

                # Update weights
                lr = 0.01  # Learning rate
                for param in self.model.parameters():
                    # Gradient descent update
                    param.data -= lr * param.grad
                    # Reset gradients
                    param.grad = 0.0

            # Print training progress
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss.data}")

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        # Convert to numpy array
        X = np.asarray(X)

        # Get the number of samples
        n_samples = X.shape[0]

        # Initialize the predictions
        y_pred = np.zeros(n_samples, dtype=int)

        # For each sample
        for i, x in enumerate(X):
            # Forward pass
            logits = self.model(x)

            # Get the predicted class
            if self.num_classes == 2:
                # Binary classification
                y_pred[i] = 1 if self.sigmoid(logits[0]).data > 0.5 else 0
            else:
                # Multi-class classification
                y_pred[i] = np.argmax([logit.data for logit in logits])

        # Convert indices to class labels
        return self.classes_[y_pred]

    def score(self, X, y):
        """Calculate the accuracy on the given test data and labels.

        Args:
            X: The test samples.
            y: The true labels for X.

        Returns:
            The accuracy of the classifier.
        """
        # Get the predicted labels
        y_pred = self.predict(X)

        # Calculate the accuracy
        accuracy = np.mean(y_pred == y)

        return accuracy

    def softmax(self, logits):
        """Compute softmax values for each set of scores in logits.

        Args:
            logits: List of logits.

        Returns:
            List of softmax values.
        """
        # Subtract max for numerical stability
        max_logit = max([logit.data for logit in logits])
        exp_logits = [Scalar(math.exp(logit.data - max_logit)) for logit in logits]
        
        # Sum of exponentials
        sum_exp = sum([exp_logit.data for exp_logit in exp_logits])
        
        # Softmax values
        softmax_values = [exp_logit / Scalar(sum_exp) for exp_logit in exp_logits]
        
        return softmax_values

    def multiclass_cross_entropy_loss(self, y_pred, y_true):
        """Compute the cross-entropy loss for multi-class classification.

        Args:
            y_pred: List of predicted probabilities.
            y_true: True class index.

        Returns:
            The cross-entropy loss.
        """
        # Select the probability corresponding to the true class
        true_prob = y_pred[y_true]
        
        # Compute the negative log likelihood (cross-entropy)
        loss = -Scalar(math.log(true_prob.data))
        
        return loss

    def sigmoid(self, x):
        """Compute the sigmoid of x.

        Args:
            x: Input value.

        Returns:
            The sigmoid of x.
        """
        # Ensure numerical stability
        if x.data > 0:
            z = Scalar(math.exp(-x.data))
            return Scalar(1.0) / (Scalar(1.0) + z)
        else:
            z = Scalar(math.exp(x.data))
            return z / (Scalar(1.0) + z)

    def binary_cross_entropy_loss(self, y_pred, y_true):
        """Compute the binary cross-entropy loss.

        Args:
            y_pred: Predicted probability.
            y_true: True class (0 or 1).

        Returns:
            The binary cross-entropy loss.
        """
        # Convert y_true to a Scalar
        y_true_scalar = Scalar(float(y_true))
        
        # Avoid numerical issues with log(0) and log(1)
        eps = 1e-15
        y_pred_safe = Scalar(max(min(y_pred.data, 1.0 - eps), eps))
        
        # Binary cross-entropy formula
        term1 = y_true_scalar * Scalar(math.log(y_pred_safe.data))
        term2 = (Scalar(1.0) - y_true_scalar) * Scalar(math.log(1.0 - y_pred_safe.data))
        
        return -(term1 + term2)

    def l2_regularization_term(self, batch_size):
        """Compute the L2 regularization term.

        Args:
            batch_size: The batch size.

        Returns:
            The L2 regularization term.
        """
        if self.alpha == 0:
            return Scalar(0.0)
        
        # Compute the sum of squares of all parameters
        param_sum_squared = Scalar(0.0)
        for param in self.model.parameters():
            param_sum_squared = param_sum_squared + param * param
        
        # Scale by alpha / (2 * batch_size)
        reg_term = Scalar(self.alpha / (2 * batch_size)) * param_sum_squared
        
        return reg_term