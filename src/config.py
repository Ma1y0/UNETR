from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Config:
    # General parameters
    cuda: bool
    mode: Literal["training", "inference"]
    # Training parameters
    epochs: int
    learning_rate: float
    data_path: Path
    label_path: Path
    output_dir: Path
    tensorboard_dir: Path
    dropout_rate: float
    # Inference parameters
    pretrained_model_path: Path


def get_config(path: str) -> Config:
    """
    Load the configuration from a YAML file.

    Args:
        path (Path): The path to the YAML configuration file.

    Returns:
        Config: The loaded configuration.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
