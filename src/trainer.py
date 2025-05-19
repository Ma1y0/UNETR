import logging
import os
from datetime import datetime

import torch
from monai.losses import DiceCELoss  # type: ignore
from monai.metrics import DiceMetric  # type: ignore
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model import get_model

logger = logging.getLogger(__name__)

def train(config: Config, training_loader: DataLoader, test_loader: DataLoader):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(tensorboard_dir)

    loss_fn = DiceCELoss(softmax=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    dice_metric = DiceMetric()

    for epoch in range(config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in training_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(training_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for batch in test_loader:
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                outputs = model(image)
                loss = loss_fn(outputs, label)
                val_loss += loss.item()
                val_outputs = torch.softmax(outputs, dim=1)
                val_preds = val_outputs.argmax(dim=1, keepdim=True)
                dice_metric(y_pred=val_preds, y=label.argmax(dim=1, keepdim=True))
        avg_val_loss = val_loss / len(test_loader)
        metric = dice_metric.aggregate().item()
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Dice/val", metric, epoch)

        logger.info(
            f"Epoch {epoch}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Dice: {metric:.4f}"
        )

        # Start saving the progress every 50 epochs after 60% of epochs are completed
        if epoch > (config.epochs * 0.60) and epoch % 50 == 0:
            torch.save(
                model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pt")
            )

    writer.close()
    torch.save(model.state_dict(), os.path.join(output_dir, "final-model.pt"))
