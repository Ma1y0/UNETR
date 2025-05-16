import logging
import os

import torch
from monai.networks.nets import UNETR  # type: ignore

from config import Config

logger = logging.getLogger(__name__)


def get_model(config: Config):
    model = UNETR(
        in_channels=1,
        out_channels=4,
        img_size=(32, 704, 576),
        dropout_rate=config.dropout_rate,
    )

    if config.mode == "inference":
        # Check if the pretrained model path exists
        if not os.path.exists(config.pretrained_model_path):
            logger.error(
                f"Pretrained model path {config.pretrained_model_path} does not exist."
            )
            raise FileNotFoundError(
                f"Pretrained model path {config.pretrained_model_path} does not exist."
            )

        # Load the pretrained model weights
        logger.info(f"Loading pretrained weights from {config.pretrained_model_path}")
        model.load_state_dict(torch.load(config.pretrained_model_path))

    return model.to("cuda" if config.cuda else "cpu")
