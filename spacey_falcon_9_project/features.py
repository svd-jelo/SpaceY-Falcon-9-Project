from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from spacey_falcon_9_project.config import (
    dataset_processed,
    processed_dir,
    random_state,
    test_set,
    training_set,
)


def load_training_set():
    training_set_path = Path(processed_dir / training_set)
    df = pd.read_csv(training_set_path)
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return X, y

def load_test_set():
    test_set_path = Path(processed_dir / test_set)
    df = pd.read_csv(test_set_path)
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return X, y

def load_whole_dataset():
    df = pd.read_csv(processed_dir / dataset_processed)
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return X, y

class GMMRBFSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.gmm_ = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.gmm_.fit(X)
        self.centers_ = self.gmm_.means_
        self.gamma_ = 1 / (self.n_components * self.gmm_.covariances_)
        return self

    def transform(self, X):
        rbf_features = []
        for center, gamma in zip(self.centers_.flatten(), self.gamma_.flatten()):
            kernel = rbf_kernel(X, [[center]], gamma=gamma)
            rbf_features.append(kernel)
        return np.hstack(rbf_features)

    def get_feature_names_out(self, names=None):
        return ["Component {} similarity".format(i) for i in range(self.n_components)]

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=20, other_label='Others'):
        self.threshold = threshold
        self.other_label = other_label

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        self.below_threshold_ = np.empty(len(self.feature_names_in_), dtype=np.ndarray)
        for i,column in enumerate(self.feature_names_in_):
            self.below_threshold_[i] = X[column].value_counts()[lambda s: s < self.threshold].index.to_numpy()
        return self

    def transform(self, X):
        X = X.copy()
        self.group_cond_ = np.empty(len(self.feature_names_in_), dtype=np.ndarray)
        features = []
        for i,column in enumerate(self.feature_names_in_):
            self.group_cond_[i] = X[column].isin(self.below_threshold_[i]).to_numpy()
            features.append(
                X[column].where(~self.group_cond_[i],
                                self.other_label)
            )

        if len(features)==1:
            return np.column_stack(features).reshape(-1,1)
        else:
            return np.column_stack(features)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

def build_features() -> ColumnTransformer:
    encode = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')  # For all cats
    regroup_encode = make_pipeline(RareCategoryGrouper(), encode)  # For orbit and mission_type
    gmmrbf = GMMRBFSimilarity(random_state=random_state)  # For payload_mass
    scale = StandardScaler()  # For all numericals except payload_mass
    log1p_scale = make_pipeline(
        FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one"), scale
    )  # For reused_count

    transformer = ColumnTransformer(
        [
            ("regroup_encode", regroup_encode, ["orbit", "mission_type"]),
            ("encode", encode, ["block"]),
            ("gmmrbf", gmmrbf, ["payload_mass"]),
            ("scale", scale, ["nearest_highway", "nearest_railway", "nearest_coastline"]),
            ("log1p_scale", log1p_scale, ["reused_count"]),
        ],
        remainder="drop"
    )

    return transformer
