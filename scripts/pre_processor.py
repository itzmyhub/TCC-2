import json
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)


class PreProcessor:
    """Wrapper para facilitar o uso e a serialização do ColumnTransformer."""

    def __init__(self, num_features: Iterable[str], cat_features: Iterable[str], verbose: bool = False):
        self.num_features = list(num_features)
        self.cat_features = list(cat_features)
        self.verbose = verbose

        self.num_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )

        self.cat_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, self.num_features),
                ('cat', self.cat_transformer, self.cat_features),
            ]
        )

        self._feature_names: Optional[List[str]] = None
        self._expected_columns: Optional[List[str]] = None

    def _validate_input(self, X: pd.DataFrame, strict: bool = True) -> None:
        """
        Valida se as colunas necessárias estão presentes.
        
        Args:
            X: DataFrame de entrada
            strict: Se True, levanta erro se colunas estiverem faltando.
                   Se False, apenas loga aviso (útil para colunas opcionais como Umidade)
        """
        missing = set(self.num_features + self.cat_features) - set(X.columns)
        if missing:
            if strict:
                raise ValueError(f"Colunas ausentes para transformação: {sorted(missing)}")
            else:
                logger.warning("Colunas ausentes (serão ignoradas): %s", sorted(missing))

    def fit(self, X: pd.DataFrame, strict: bool = True):
        """
        Ajusta o pré-processador aos dados.
        
        Args:
            X: DataFrame de entrada
            strict: Se True, validação estrita. Se False, permite colunas opcionais faltando.
        """
        self._validate_input(X, strict=strict)
        if self.verbose:
            logger.info("Ajustando pré-processador em %d linhas.", len(X))
        
        # Filtrar apenas colunas que existem em X
        num_features_existentes = [f for f in self.num_features if f in X.columns]
        cat_features_existentes = [f for f in self.cat_features if f in X.columns]
        
        # Criar preprocessor apenas com colunas existentes
        if num_features_existentes != self.num_features or cat_features_existentes != self.cat_features:
            logger.info("Ajustando preprocessor: %d num features, %d cat features (algumas podem estar faltando)",
                       len(num_features_existentes), len(cat_features_existentes))
            # Recriar preprocessor com apenas colunas existentes
            temp_num_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )
            temp_cat_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]
            )
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', temp_num_transformer, num_features_existentes),
                    ('cat', temp_cat_transformer, cat_features_existentes),
                ]
            )
        
        self.preprocessor.fit(X)
        self._feature_names = list(self.preprocessor.get_feature_names_out())
        self._expected_columns = list(X.columns)
        return self

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def transform(self, X: pd.DataFrame, strict: bool = False):
        """
        Transforma os dados usando o pré-processador ajustado.
        
        Args:
            X: DataFrame de entrada
            strict: Se True, validação estrita. Se False, permite colunas opcionais faltando.
        """
        self._validate_input(X, strict=strict)
        if self.verbose:
            logger.info("Transformando %d linhas.", len(X))
        
        # Garantir que X tenha as mesmas colunas que foram usadas no fit
        # Adicionar colunas faltantes com NaN (serão tratadas pelo imputer)
        colunas_esperadas = set(self.num_features + self.cat_features)
        colunas_faltando = colunas_esperadas - set(X.columns)
        
        if colunas_faltando:
            logger.warning("Colunas faltando em X (serão preenchidas com NaN): %s", sorted(colunas_faltando))
            for col in colunas_faltando:
                X = X.copy()
                X[col] = np.nan
        
        return self.preprocessor.transform(X)

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transform(X)
        columns = self.get_feature_names()
        return pd.DataFrame(transformed, columns=columns, index=X.index)

    def get_feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names
        try:
            self._feature_names = list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            cat_ohe_cols = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.cat_features)
            self._feature_names = list(self.num_features) + list(cat_ohe_cols)
        return self._feature_names

    def to_metadata(self) -> Dict[str, Optional[List[str]]]:
        return {
            'num_features': self.num_features,
            'cat_features': self.cat_features,
            'expected_columns': self._expected_columns,
            'feature_names': self._feature_names,
        }

    def save_metadata(self, path: str) -> None:
        metadata = self.to_metadata()
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(metadata, fp, indent=2)
        logger.info("Metadados do pré-processador salvos em %s", path)

    @classmethod
    def load_metadata(cls, path: str, verbose: bool = False) -> "PreProcessor":
        with open(path, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)
        instance = cls(
            num_features=metadata['num_features'],
            cat_features=metadata['cat_features'],
            verbose=verbose,
        )
        instance._expected_columns = metadata.get('expected_columns')
        instance._feature_names = metadata.get('feature_names')
        return instance