import numpy as np
from error import Error
from learning_rater import Learning_rate

class Regressor():
    """
    A basic multivariate linear regression model using gradient descent.
    """
    def __init__(self):
        self.beta = None
        self.rater = Learning_rate()

    def gradient_descent(self, X, Y, learning_rate=0.001, iterations=1000, normalize = False):
        """
        Trains the regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Feature matrix.
        Y : np.ndarray, shape (m, t)
            Target matrix.
        learning_rate : float
            Step size for gradient updates.
        iterations : int
            Number of gradient descent steps.
        normalize : bool
            If True, scales features to [0, 1].

        Returns
        -------
        beta : np.ndarray
            Learned coefficient matrix.
        """
        if normalize:
            X = self.normalizer(X)

        m, n = X.shape
        X_b = np.hstack((np.ones((m, 1)), X))
        self.beta = np.zeros((n + 1, Y.shape[1]))

        for i in range(iterations):
            Y_pred = X_b @ self.beta
            gradients = (1 / m) * X_b.T @ (Y_pred - Y)
            self.beta -= learning_rate * gradients

            if i % 100 == 0:
                loss = Error.mean_squared_error(Y, Y_pred)
                print(f"Iteration {i}: Loss = {loss:.5f}")

        return self.beta

    def normalizer(self, X):
        """
        Normalizes each feature column to the range [0, 1].

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Normalized features.
        """
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X = (X - X_min) / (X_max - X_min)

        return X

    def predict(self, X, beta):
        """
        Predicts target values using the trained model.

        Parameters
        ----------
        X : np.ndarray
            New input features.
        beta : np.ndarray
            Coefficients from the trained model.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if beta is None:
            raise ValueError("train the model first.")
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))  # Bias ekleme
        return X_b @ beta
