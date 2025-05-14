import numpy as np

class Error():
    """
    Provides basic error metrics for regression tasks.
    """
    @staticmethod
    def mean_squared_error(Y, Y_pred):
        """
        Returns the Mean Squared Error between actual and predicted values.

        Parameters:
        Y : array-like — true values
        Y_pred : array-like — predicted values

        Returns:
        float — MSE value
        """
        m = len(Y)
        return (1 / (2 * m)) * np.sum((Y_pred - Y) ** 2)

    @staticmethod
    def r2_score(Y, Y_pred):
        """
        Returns the R-squared score for model performance.

        Parameters:
        Y : array-like — true values
        Y_pred : array-like — predicted values

        Returns:
        float — R² score
        """
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2