# https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        I will impute this columns as well with the most frequent value as 
        most of the columns are integer encoded, it doesnt make sense to get
        mean of those.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].value_counts().index[0] for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


