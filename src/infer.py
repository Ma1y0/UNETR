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
    pred = pred.cpu().numpy().astype(np.uint8).squeeze()

    logger.info(f"Predictions shape: {pred.shape}")

    # Crop the prediction
    output_dim = (601, 674, 560)
    logger.info(f"Cropping predictions from {pred.shape} to {output_dim}")
    start = [(pred.shape[i] - output_dim[i]) // 2 for i in range(len(pred.shape))]
    end = [start[i] + output_dim[i] for i in range(len(pred.shape))]
    cropped_pred = pred[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    logger.info(f"Cropped predictions shape: {cropped_pred.shape}")
    logger.info(f"Cropped predictions dtype: {cropped_pred.dtype}")
    logger.info(
        f"Cropped predictions min: {cropped_pred.min()}, max: {cropped_pred.max()}"
    )
    logger.info(
        f"Cropped predictions unique values: {np.unique(cropped_pred, return_counts=True)}"
    )

    # Save the predictions
    tifffile.imwrite("pred.tiff", cropped_pred)
