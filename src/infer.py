import logging

import numpy as np
import tifffile
import torch

from config import Config
from data import get_data_loader
from model import get_model

logger = logging.getLogger(__name__)

def infer(config: Config):
    # Get the model
    model = get_model(config)
    model.eval()

    # Get the data loader
    data_loader = get_data_loader(config)

    logger.info("Stating inference")
    output = []
    for batch in data_loader:
        image = batch["image"].to("cuda")
        with torch.no_grad():
            pred = model(image)
            pred = torch.argmax(pred, dim=1)
            output.append(pred)

    # Concatenate the predictions
    pred = torch.cat(output, dim=1)
    pred = pred.cpu().numpy().astype(np.uint8)

    logger.info(f"Predictions shape: {pred.shape}")
    logger.info(f"Predictions dtype: {pred.dtype}")
    logger.info(f"Predictions min: {pred.min()}, max: {pred.max()}")
    logger.info(f"Predictions unique values: {np.unique(pred, return_counts=True)}")

    # Save the predictions
    tifffile.imwrite("pred.tiff", pred)