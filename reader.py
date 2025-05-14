import pandas as pd
from pathlib import Path

class Reader():
    """
    A utility class to load specific columns from a CSV file
    for use in regression models.
    """
    @staticmethod
    def partial_reading(file: Path, independent_columns, dependent_columns):
        """
        Reads selected independent and dependent columns from a CSV file.

        Parameters
        ----------
        file : Path
            Path to the CSV file.
        independent_columns : list of str
            Names of the feature (input) columns.
        dependent_columns : list of str
            Names of the target (output) columns.

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values.
        """

        df = pd.read_csv(file, sep=',')

        df_selected = df[independent_columns + dependent_columns].dropna()

        X = df_selected[independent_columns].values
        y = df_selected[dependent_columns].values

        return X, y

