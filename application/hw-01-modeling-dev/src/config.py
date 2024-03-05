from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float
    width: int
    height: int


class TrainConfig(BaseModel):
    n_epochs: int
    accelerator: str
    device: int
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict


class Config(BaseModel):
    project_name: str
    experiment_name: str
    data_config: DataConfig
    num_classes: int
    monitor_metric: str
    monitor_mode: str
    losses: List[LossConfig]
    train_config: TrainConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
