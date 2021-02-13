from .linear import LinearRegression
from .ridge_lasso import (
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
)
from .knn import KNNRegression
from .nn import SKMLPRegression
from .rf import RandomForestRegression
from .svm import KernelSVRRegression
from .lgbm import LGBMRegression
from .cat_boost import CatBoostRegression
