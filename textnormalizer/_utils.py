from typing import List
import logging
from ._exceptions import InvalidDataError


class TNLogger:
    def __init__(self, level):
        self.logger = logging.getLogger('TextNormalizer')
        self.set_level(level)
        self.set_handler()
        self.logger.propagate = False

    def set_level(self, level: str):
        all_ = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if level not in all_:
            raise ValueError(f'{level} is not a valid logging level')
        self.logger.setLevel(level)

    def info(self, message):
        self.logger.info(f'{message}')

    def set_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


def is_empty(input_list: List[str]) -> bool:
    """
    Checks if the input is empty
    """
    return input_list == []


def contains_empty(input_list: List[str]) -> bool:
    """
    Checks if the input contains empty string
    """
    return any(map(lambda x: x == '', input_list))


def all_string(input_list: List[str]) -> bool:
    """
    Elements of the input list must be string type
    """
    return all(map(lambda x: isinstance(x, str), input_list))


def is_valid_input(input_list: List[str]):
    """
    Checks if the input is valid
    """
    if is_empty(input_list):
        raise InvalidDataError('Input List cannot be empty')

    if not all_string(input_list):
        raise InvalidDataError('Elements of the input list must be string type')

    if contains_empty(input_list):
        raise InvalidDataError('Input List cannot contain empty string')

    if not isinstance(input_list, list):
        raise TypeError('Input must be a list')
