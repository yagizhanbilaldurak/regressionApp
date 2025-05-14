import numpy as np

class Learning_rate():
    """
    Generates a range of learning rates based on a given step size.
    """
    @staticmethod
    def rater(step_size: int):
        """
        Returns a list of learning rates on a logarithmic scale.

        Parameters:
        step_size : int
            Controls the number of learning rates to generate.

        Returns:
        numpy.ndarray
            Array of rounded learning rates.
        """
        # Normalize density to control the number of points in the grid
        num_points = int(np.clip(step_size, 0, 100) * 0.2) + 6  # Ensures at least 6 points

        # Generate the learning rate grid using logarithmic scale
        learning_rates = np.logspace(-4, 1, num=num_points)

        for i in range(len(learning_rates)):
            learning_rates[i] = round(learning_rates[i], 2)

        return learning_rates

lr = Learning_rate()

print(len(lr.rater(15)))