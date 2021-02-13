from enum import Enum
from .regression import (
    LinearRegression,
    KNNRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
    SKMLPRegression,
    RandomForestRegression,
    KernelSVRRegression,
    LGBMRegression,
    CatBoostRegression,
)


class RegressionModel(Enum):
    LINEAR = LinearRegression
    KNN = KNNRegression
    RIDGE = RidgeRegression
    LASSO = LassoRegression
    ELASTIC_NET = ElasticNetRegression
    SKMLP = SKMLPRegression
    RANDOM_FOREST = RandomForestRegression
    KERNEL_SVR = KernelSVRRegression
    LGBM = LGBMRegression
    CAT_BOOST = CatBoostRegression
