from enum import Enum
from .cnn import CNN
from .lstm import BILSTM


class QModelType(Enum):
    CNN = CNN
    BILSTM = BILSTM
