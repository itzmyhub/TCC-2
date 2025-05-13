import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class PreProcessor:
    def __init__(self, num_features, cat_features, verbose=False):
        self.num_features = num_features
        self.cat_features = cat_features
        self.verbose = verbose

        self.num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', self.num_transformer, self.num_features),
            ('cat', self.cat_transformer, self.cat_features)
        ])

    def fit_transform(self, X):
        if self.verbose:
            print("Fitting and transforming training data...")
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        if self.verbose:
            print("Transforming new data...")
        return self.preprocessor.transform(X)

    def transform_to_df(self, X):
        transformed = self.transform(X)
        cat_ohe_cols = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.cat_features)
        all_columns = list(self.num_features) + list(cat_ohe_cols)
        return pd.DataFrame(transformed, columns=all_columns, index=X.index)