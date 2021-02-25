from enum import Enum


class Position(Enum):
    NO_POSI = 0
    HOLD = 1


class Action(Enum):
    NO_POSI = 0
    BUY = 1
    SELL = 2
    HOLD = 3

    @staticmethod
    def idx2action(idx: int):
        if idx == 0:
            return Action.NO_POSI
        elif idx == 1:
            return Action.BUY
        elif idx == 2:
            return Action.SELL
        elif idx == 3:
            return Action.HOLD
        else:
            raise Exception(f'Invalid index : {idx}')
