from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class Config:
    model: Literal["SwinUNETR"]
    cuda: bool
    epochs: int
    learning_rate: float
    data_path: Path
    label_path: Path
    output_dir: Path
    tensorboard_dir: Path = Path("runs/swinunetr_finetune")
    pretrained: Optional[Path] = None


def get_config(path: Path) -> Config:
    """
    Load the configuration from a YAML file.

    Args:
        path (Path): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
