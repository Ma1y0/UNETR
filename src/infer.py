import logging

import numpy as np
import tifffile
import torch

from config import Config
from data import get_data_loader
from model import get_model

logger = logging.getLogger(__name__)



def infer(config: Config):
    model = get_model(config)
    weight_path = (
        "/home/bar/UNETR/output/SwinUNETR/UNETR/2025-05-14_12:10:52/fina-model.pt"
    )
    print(f"Loading model weights from {weight_path}")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    data_loader = get_data_loader()
    output = []
    for batch in data_loader:
        image = batch["image"].to("cuda")
        with torch.no_grad():
            pred = model(image)
            pred = torch.argmax(pred, dim=1)
            output.append(pred)
            print(f"Batch shape: {pred.shape}")

    pred = torch.cat(output, dim=1)
    pred = pred.cpu().numpy().astype(np.uint8)

    print(f"Predictions shape: {pred.shape}")
    print(f"Predictions dtype: {pred.dtype}")
    print(f"Predictions min: {pred.min()}, max: {pred.max()}")
    print(f"Predictions unique values: {np.unique(pred, return_counts=True)}")
    tifffile.imwrite("pred.tiff", pred)
