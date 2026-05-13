import os
from typing import List, Callable, DefaultDict
from pathlib import Path
from dataclasses import dataclass
from logging import getLogger, StreamHandler, INFO

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.cuda import device as TorchDevice
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from jett.modules import JETT

logger = getLogger()
if not logger.handlers:
    handler = StreamHandler()
    handler.setLevel(INFO)
    logger.addHandler(handler)
    logger.setLevel(INFO)


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
    def __init__(self, identity: str, keys: List[str], values: List[float], primary_key: int):
        """
        The output metrics of `train_one_epoch` and `valid_one_epoch` function.
        :param identity: Specifies the identity of this metric, i.e., its origin.
        :param primary_key: The index of the primary parameter in the `values` list.
        :param values: The actual output value.
        """
        self.identity = identity
        self.keys = keys
        self.primary_key = primary_key
        self.values: List[float] = values

    def get_primary(self) -> float:
        return self.values[self.primary_key]

    def get_primary_name(self) -> str:
        return self.keys[self.primary_key]

    def print(self):
        raise NotImplementedError()


class JETTTrainer:
    def __init__(self, model: JETT,
                 save_target: str,
                 configs: JETTConfigs,
                 train_params: JETTTrainParams,
                 train_one_epoch: Callable[[JETT, JETTTrainParams], Metrics],
                 valid_one_epoch: Callable[[JETT, JETTTrainParams], Metrics],
                 is_better: Callable[[float, float], bool]):
        self.model = model

        # Saves
        self.save_target = save_target

        if not os.path.exists(self.save_target):
            os.makedirs(self.save_target)

        # Configurations and Training Parameters
        self.configs = configs
        self.train_params = train_params

        # Train adn Validate functions.
        self.train_one_epoch = train_one_epoch
        self.valid_one_epoch = valid_one_epoch
        self.is_better = is_better

        # Running metrics
        self.train_metrics_list: List[Metrics] = []
        self.valid_metrics_list: List[Metrics] = []

    def train(self):
        self.model.to(self.train_params.device)
        logger.info("Train start!")
        best: float = 0.0
        # noinspection PyTypeChecker
        for epoch in range(1, self.configs.epochs + 1):
            logger.info(f"Epoch {epoch}")
            train_metrics: Metrics = self.train_one_epoch(self.model, self.train_params)
            valid_metrics: Metrics = self.valid_one_epoch(self.model, self.train_params)
            self.train_params.scheduler.step()

            self.train_metrics_list.append(train_metrics)
            self.valid_metrics_list.append(valid_metrics)

            if self.is_better(best, valid_metrics.get_primary()):
                best = valid_metrics.get_primary()
                torch.save(self.model.state_dict(), os.path.join(self.save_target, "best.pth"))
                logger.info(f"Best model saved! Best {valid_metrics.get_primary_name()}: "
                            f"{valid_metrics.get_primary():.2f}%.")

            logger.info(valid_metrics.print())

        logger.info(f"Training completed!")
        return self

    def print_result(self):
        train_graph_data = DefaultDict(list)
        valid_graph_data = DefaultDict(list)

        for metrics in self.train_metrics_list:
            for key, value in zip(metrics.keys, metrics.values):
                train_graph_data[key].append(value)

        for metrics in self.valid_metrics_list:
            for key, value in zip(metrics.keys, metrics.values):
                valid_graph_data[key].append(value)

        # Get all metric keys
        all_keys = sorted(set(train_graph_data.keys()) | set(valid_graph_data.keys()))

        # Create individual comparison graph for each metric
        for key in all_keys:
            plt.figure(figsize=(10, 6))

            # Plot train data if available
            if key in train_graph_data:
                plt.plot(train_graph_data[key], 'b-o', label='Train', linewidth=2, markersize=5)

            # Plot valid data if available
            if key in valid_graph_data:
                plt.plot(valid_graph_data[key], 'r-s', label='Valid', linewidth=2, markersize=5)

            plt.title(f'{key} - Train vs Valid', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

            save_path = os.path.join(str(Path(self.save_target)), f"{key}-comparison.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()

            logger.info(f"Saved {key} comparison graph to: {save_path}")


