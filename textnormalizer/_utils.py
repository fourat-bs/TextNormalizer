from typing import List
import logging
from ._exceptions import InvalidDataError
import numpy as np
import wandb
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


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

    def warning(self, message):
        self.logger.warning(f'{message}')


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


class DataGenerator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idx = 0
        while idx < self.steps:
            yield self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
            idx += 1


class Trainer:
    def __init__(self, model, data_loader, metrics, loss, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def train(self, logger,  callbacks, epochs=1):
        try:
            import wandb
        except:
            logger.warning('W&B is not installed. Please install it in order to track the experiment.')
            pass
