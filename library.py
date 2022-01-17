import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, mapping_column, mapping_dict: dict):
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column  # column to focus on

    def fit(self, X, y=None):
        print(f"Warning: MappingTransformer.fit does nothing.")
        return X

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
        X_ = X.copy()
        X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
        return X_

    def fit_transform(self, X, y=None):
        result = self.transform(X)
        return result


class RenamingTransformer(BaseEstimator, TransformerMixin):
    # your __init__ method below

    def __init__(self, mapping_dict: dict):
        self.mapping_dict = mapping_dict

    # write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.
    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
        # your assert code below

        column_list = X.columns.to_list()
        not_found = list(set(self.mapping_dict.keys()) - set(column_list))
        assert len(
            not_found) < 0, f"Columns {str(not_found)[1:-1]}, are not in the data table"

        X_ = X.copy()
        return X_.rename(columns=self.mapping_dict)


class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=True):
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f"Expected Pandas DF object, got {type(X)} instead"

        temp_df = X.copy()
        temp_df = pd.get_dummies(temp_df,
                                 prefix="Joined",
                                 prefix_sep="_",
                                 columns=[self.target_column],
                                 dummy_na=self.dummy_na,
                                 drop_first=self.drop_first)

        return temp_df

    def fit(self, X):
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_list, action='drop'):
        assert action in ['keep',
                          'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list,
                          list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list = column_list
        self.action = action

    def fit(self, X):
        print("Fit does nothing")
        return X

    def transform(self, X):

        temp_df = X.copy()
        if self.action == "drop":
            return temp_df.drop(columns=self.column_list)
        else:
            return temp_df[self.column_list]
            # return temp_df

    def fit_transform(self, X, X2=None):
        return self.transform(X)
