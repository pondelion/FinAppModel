from .base_processing import IStructuredDataProcessing
from .default_processing import DefaultStructuredDataProcessing
from .linear_regression import (
    TrendLinearRegressionDataProcessing,
    LinearRegressionDataProcessing
)
from .ridge_lasso_regression import (
    RidgeRegressionDataProcessing,
    LassoRegressionDataProcessing,
    ElasticNetRegressionDataProcessing,
)
from .random_forest_regression import RandomForestDataProcessing
from .svm_regression import KernelSVRDataProcessing
from .nn_regression import SKMLPDataProcessing
from .knn_regression import KNNDataProcessing
