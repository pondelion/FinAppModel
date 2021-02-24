from enum import Enum
from .cnn import CNN


class QModelType(Enum):
    CNN = CNN
