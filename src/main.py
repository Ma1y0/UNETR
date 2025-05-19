import logging
import os
from datetime import datetime
from io import StringIO

from monai.config import print_config  # type: ignore

from config import get_config
from data import get_data_loader


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
    try:
        logger.info("Loading configuration from config.yaml")
        config = get_config("config.yaml")
        logger.info(f"Configuration loaded: {config}")

        # Print MONAI info
        monai_info = StringIO()
        print_config(monai_info)
        logger.info(f"MONAI info: {monai_info.getvalue()}")


        if config.mode == "training":
            from trainer import train

            train_loader, test_loader = get_data_loader(config)
            train(config, train_loader, test_loader)
        elif config.mode == "inference":
            from infer import infer

            data_loader = get_data_loader(config)
            # Help the type checker
            assert not isinstance(data_loader, tuple), "Expected a single DataLoader in inference mode"
            infer(config, data_loader)
        else:
            logger.error(f"Invalid mode: {config.mode}")
            raise ValueError(f"Unsupported mode: {config.mode}")

    except Exception as e:
        logger.exception("An error occurred during execution: %s", e)
        raise e


if __name__ == "__main__":
    main()