import logging
import os
from datetime import datetime
from io import StringIO

from monai.config import print_config  # type: ignore

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
    # Set up logging
    logger = setup_logging()
    logger.info("Loading configuration from config.yaml")
    config = get_config("config.yaml")
    logger.info(f"Configuration loaded: {config}")

    # Print MONAI info
    monai_info = StringIO()
    print_config(monai_info)
    logger.info(f"MONAI info: {monai_info.getvalue()}")

    # Get data loader
    dataloder = get_data_loader()
    print(f"Data loader created with {len(dataloder)} samples.")

    # Training
    logger.info("Training started")
    train(config, dataloder)


if __name__ == "__main__":
    main()
