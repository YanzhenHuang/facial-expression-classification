from typing import List, Callable
from dataclasses import dataclass
from logging import getLogger

import torch
from torch import nn
from torch.cuda import device as TorchDevice
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from jett.modules import JETT

logger = getLogger()


@dataclass
class JETTConfigs:
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int


@dataclass
class JETTTrainParams:
    train_loader: DataLoader
    valid_loader: DataLoader
    criterion: nn.Module
    optimizer: Optimizer
    device: TorchDevice
    scheduler: LRScheduler


class Metrics:
    def __init__(self, primary_key: int, primary_name: str, values: List[float]):
        self.primary_key = primary_key
        self.primary_name: str = primary_name
        self.values: List[float] = values

    def get_primary(self) -> float:
        return self.values[self.primary_key]

    def get_primary_name(self) -> str:
        return self.primary_name

    def print(self):
        raise NotImplementedError()


class JETTTrainer:
    def __init__(self, model: JETT,
                 save_target: str,
                 configs: JETTConfigs,
                 train_params: JETTTrainParams,
                 train_one_epoch: Callable[[JETT, JETTTrainParams], List[float]],
                 valid_one_epoch: Callable[[JETT, JETTTrainParams], List[float]],
                 is_better: Callable[[float, List[float]], bool]):
        self.model = model

        # Saves
        self.save_target = save_target

        # Configurations and Training Parameters
        self.configs = configs
        self.train_params = train_params

        # Train adn Validate functions.
        self.train_one_epoch = train_one_epoch
        self.valid_one_epoch = valid_one_epoch
        self.is_better = is_better

    def train(self):
        self.model.to(self.train_params.device)
        logger.info("Train start!")
        best: float = 0.0
        # noinspection PyTypeChecker
        for epoch in range(1, self.configs.epochs + 1):
            train_metrics: Metrics = self.train_one_epoch(self.model, self.train_params)
            valid_metrics: Metrics = self.valid_one_epoch(self.model, self.train_params)
            self.train_params.scheduler.step()

            if self.is_better(best, valid_metrics.get_primary()):
                best = valid_metrics.get_primary()
                torch.save(self.model.state_dict(), self.save_target)
                logger.info(f"Best model saved! Best{valid_metrics.get_primary_name()}: "
                            f"{valid_metrics.get_primary()}")

            logger.info(valid_metrics.print())

        logger.info(f"Training completed!")
