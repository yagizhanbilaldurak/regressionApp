import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """
    A class for visualizing simple 2D slices of multivariate regression results.
    """
    def plot(self, X, y, beta, feature_index=0, target_index=0, feature_name=None, target_name=None):
        """
        Plots data points and the regression line for a selected feature-target pair.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Input features.
        y : np.ndarray, shape (m, t)
            Target variables.
        beta : np.ndarray, shape (n+1, t)
            Regression coefficients, including intercept.
        feature_index : int
            Index of the feature to plot.
        target_index : int
            Index of the target variable.
        feature_name : str or None
            Label for the x-axis (feature).
        target_name : str or None
            Label for the y-axis (target).
        """

        if feature_name is None:
            feature_name = 'X'
        if target_name is None:
            target_name = 'y'

        means = np.mean(X, axis=0)
        fixed_offset = beta[0, target_index]
        for i in range(X.shape[1]):
            if i != feature_index:
                fixed_offset += beta[i + 1, target_index] * means[i]

        slope = beta[feature_index + 1, target_index]

        # regression line
        x_feature = X[:, feature_index]
        x_min, x_max = np.min(x_feature), np.max(x_feature)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = fixed_offset + slope * x_line

        # graph
        plt.figure(figsize=(8, 6))
        plt.scatter(x_feature, y[:, target_index], color='blue', label='data points')
        plt.plot(x_line, y_line, color='red', label='regression line')
        plt.xlabel(feature_name)
        plt.ylabel(target_name)
        plt.title(f"Regression: {target_name} vs {feature_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_streamlit(self, X, y, beta, feature_index=0, target_index=0, feature_name=None, target_name=None):
        """
        Returns a matplotlib figure for use in Streamlit or similar environments.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Input features.
        y : np.ndarray, shape (m, t)
            Target variables.
        beta : np.ndarray, shape (n+1, t)
            Regression coefficients, including intercept.
        feature_index : int
            Index of the feature to plot.
        target_index : int
            Index of the target variable.
        feature_name : str or None
            Label for the x-axis (feature).
        target_name : str or None
            Label for the y-axis (target).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated plot figure.
        """
        if feature_name is None:
            feature_name = f"Feature {feature_index}"
        if target_name is None:
            target_name = f"Target {target_index}"

        means = np.mean(X, axis=0)
        fixed_offset = beta[0, target_index]
        for i in range(X.shape[1]):
            if i != feature_index:
                fixed_offset += beta[i + 1, target_index] * means[i]

        slope = beta[feature_index + 1, target_index]

        x_feature = X[:, feature_index]
        x_min, x_max = np.min(x_feature), np.max(x_feature)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = fixed_offset + slope * x_line

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_feature, y[:, target_index], color='blue', label='data points')
        ax.plot(x_line, y_line, color='red', label='regression line')
        ax.set_xlabel(feature_name)
        ax.set_ylabel(target_name)
        ax.set_title(f"Regression: {target_name} vs {feature_name}")
        ax.legend()
        ax.grid(True)

        return fig
