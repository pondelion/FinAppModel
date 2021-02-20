from abc import ABCMeta, abstractmethod


class BaseStorage(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def save_file():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def download_file():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_filelist():
        raise NotImplementedError
