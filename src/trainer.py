import logging
import os
from datetime import datetime

import torch
from monai.losses import DiceCELoss  # type: ignore
from monai.metrics import DiceMetric  # type: ignore
from monai.networks.nets import UNETR  # type: ignore
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config import Config

logger = logging.getLogger(__name__)


def get_model(config: Config):
    model = UNETR(
        in_channels=1,
        out_channels=4,
        img_size=(32, 704, 576),
        dropout_rate=config.dropout_rate,
    )

    return model.to("cuda" if config.cuda else "cpu")


def train(config: Config, dataloader):
    tensorboard_dir = f"{config.tensorboard_dir}/UNETER/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    output_dir = (
        f"{config.output_dir}/UNETER/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(config)
    logger.info(
        f"Model UNETER created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    writer = SummaryWriter(tensorboard_dir)

    loss_fn = DiceCELoss(softmax=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    dice_metric = DiceMetric()

    for epoch in range(config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            image = batch["image"].to("cuda" if config.cuda else "cpu")
            label = batch["label"].to("cuda" if config.cuda else "cpu")
            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"].to("cuda" if config.cuda else "cpu")
                label = batch["label"].to("cuda" if config.cuda else "cpu")
                outputs = model(image)
                val_outputs = torch.softmax(outputs, dim=1)
                val_preds = val_outputs.argmax(dim=1, keepdim=True)
                dice_metric(y_pred=val_preds, y=label.argmax(dim=1, keepdim=True))

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar("Dice/val", metric, epoch)

        logger.info(
            f"Epoch {epoch}/{config.epochs} - Train Loss: {avg_loss:.4f} - Val Dice: {metric:.4f}"
        )

        # Start saving the progress every 50 epochs after 60% of the epochs are completed
        if epoch > (config.epochs / 0.60) and epoch % 50 == 0:
            torch.save(
                model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pt")
            )

    writer.close()
    torch.save(model.state_dict(), os.path.join(output_dir, "final-model.pt"))
