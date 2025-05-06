import logging
import os
from datetime import datetime

import numpy as np
import tifffile
from monai.config import print_config

from config import get_config
from data import get_data_loader
from trainer import train


def setup_logging():
    """Set up logging configuration"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/swinunetr_training-[{datetime.now()}].log"),
        ],
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("Loading configuration...")
    print_config()

    dataloder = get_data_loader()
    print(f"Data loader created with {len(dataloder)} samples.")

    print("Training started...")
    train(get_config("config.yaml"), dataloder)


if __name__ == "__main__":
    main()
