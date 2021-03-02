from enum import Enum

from .lstm import BILSTM

class TraderModel(Enum):
    BILSTM = BILSTM
