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


class PearsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, X2=None):
        print("From PearsonTransformer: Warning: Fit does nothing!")
        return X

    def transform(self, X):
        temp = X.copy()
        df_corr = temp.corr(method='pearson')
        masked_df = df_corr.abs() > self.threshold
        upper_mask = np.triu(masked_df, 1)
        t = np.any(upper_mask, 0)
        correlated_columns = [masked_df.columns[i] for i, j in
                              enumerate(upper_mask) if t[i]]
        new_df = transformed_df.drop(correlated_columns, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class Sigma3Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column

    def transform(self, X):
        assert isinstance(X,
                          pd.core.frame.DataFrame), f"From Sigma3Transformer: transform expected Pandas DF, got {type(X)}"

        temp = X.copy()
        mean = temp[self.target_column].mean()
        std = temp[self.target_column].std()  # sigma
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        temp[self.target_column] = temp[self.target_column].clip(
            lower=lower_bound, upper=upper_bound)

        return temp

    def fit(self, X, X2=None):
        print("Warning: Sigma3Transformer fit does nothing!")
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)


class TukeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, fence="outer"):
        assert fence in ["inner", "outer"]
        self.target_column = target_column
        self.fence = fence

    def transform(self, X, X2=None):
        temp = X.copy()
        q1 = temp[self.target_column].quantile(0.25)
        q3 = temp[self.target_column].quantile(0.75)
        iqr = q3 - q1
        outer_low = q1 - 3 * iqr
        outer_high = q3 + 3 * iqr
        inner_low = q3 - 1.5 * iqr
        inner_high = q3 + 1.5 * iqr

        if self.fence == "inner":
            temp[self.target_column] = temp[self.target_column].\
                clip(lower=inner_low, upper=inner_high)
        else:
            temp[self.target_column] = temp[self.target_column].\
                clip(lower=outer_low, upper=outer_high)
        return temp

    def fit(self, X, X2=None):
        print("Warning: TukeyTransformer fit does nothing!")
        return X

    def fit_transform(self, X, X2=None):
        return self.transform(X)