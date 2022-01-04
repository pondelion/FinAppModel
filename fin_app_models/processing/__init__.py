from .base_processing import IStructuredDataProcessing
from .default_processing import DefaultStructuredDataProcessing
from .linear import (
    TrendLinearRegressionDataProcessing,
    LinearRegressionDataProcessing
)
from .ridge_lasso import (
    RidgeRegressionDataProcessing,
    LassoRegressionDataProcessing,
    ElasticNetRegressionDataProcessing,
)
from .random_forest import RandomForestDataProcessing
from .svm import KernelSVRDataProcessing
from .nn import SKMLPDataProcessing
from .knn import KNNDataProcessing
from .lgbm import LGBMDataProcessing
from .cat_boost import CatBoostDataProcessing
from .lstm import LSTMDataProcessing
