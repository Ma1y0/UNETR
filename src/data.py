import logging

import tifffile
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, Dataset, ImageReader
from monai.transforms import (
    AsDiscreted,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    Lambdad,
    LoadImageD,
    MapTransform,
    ScaleIntensityRangePercentilesd,
    ToTensorD,
)

from config import get_config

logger = logging.getLogger(__name__)


class SplitData(MapTransform):
    def __init__(self, keys, num_splits=4, split_dim=1):
        super().__init__(keys)
        self.num_splits = num_splits
        self.split_dim = split_dim

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]
            logger.info(f"Splitting data for key: {key}, original shape: {x.shape}")
            # Split the data along the specified dimension
            split_size = x.shape[self.split_dim] // self.num_splits
            logger.info(f"Split size: {split_size}")
            splits = torch.split(x, split_size, dim=self.split_dim)
            clean_splits = [x for x in splits if x.shape[self.split_dim] == split_size]
            split_data = torch.stack(clean_splits)
            d[key] = split_data

        return d


class FlattenSplitDataset(Dataset):
    """
    Dataset wrapper that takes each record from the base dataset, which is
    assumed to have a split dimension (e.g. shape [num_splits, ...] in 'image' and 'label'),
    and creates separate data samples (records) for each split.
    """

    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: The dataset that outputs records with an added split dimension.
            split_key: The key in the record from which to infer the number of splits.
                       It is assumed that all keys have the same first dimension size.
        """
        self.base_dataset = base_dataset
        self.flat_records = []

        for data in self.base_dataset:
            num_splits = data["image"].shape[0]
            for i in range(num_splits):
                split_record = {}
                for key, value in data.items():
                    split_record[key] = value[i]
                self.flat_records.append(split_record)

    def __len__(self):
        return len(self.flat_records)

    def __getitem__(self, index):
        return self.flat_records[index]


class TiffReader(ImageReader):
    """
    Custom reader for TIFF files.
    """

    def read(self, data, **kwargs):
        logger.info(f"Loading TIFF file: {data[0]}")
        try:
            file = tifffile.memmap(data[0], mode="r")
            logger.info(f"TIFF file loaded. Shape: {file.shape}, Dtype: {file.dtype}")
            return file
        except Exception as e:
            logger.error(f"Error loading TIFF file: {e}")
            raise

    def get_data(self, img):
        return [img, {}]

    def verify_suffix(self, filename):
        suffixes = [".tiff", ".tif"]
        for suffix in suffixes:
            if filename.endswith(suffix):
                return True
        return False


class Pad(MapTransform):
    def __init__(self, keys, target_shape, pad_mode="constant", constant_values=0):
        super().__init__(keys)
        self.target_shape = target_shape
        self.pad_mode = pad_mode
        self.constant_values = constant_values

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]

            # Ensure input has a channel dimension.
            # If input shape is (D, H, W), add a channel dimension at position 0.
            if x.ndim == 3:
                x = x.unsqueeze(0)  # shape becomes (1, D, H, W)

            _, d_size, h_size, w_size = x.shape
            target_d, target_h, target_w = self.target_shape

            # Check if padding is required.3
            if d_size > target_d or h_size > target_h or w_size > target_w:
                raise ValueError(
                    f"Input {key} has shape {(d_size, h_size, w_size)} which exceeds the target shape {self.target_shape}."
                )

            # Calculate the padding on each side
            pad_d = target_d - d_size
            pad_h = target_h - h_size
            pad_w = target_w - w_size

            pad_left_d = pad_d // 2
            pad_right_d = pad_d - pad_left_d

            pad_left_h = pad_h // 2
            pad_right_h = pad_h - pad_left_h

            pad_left_w = pad_w // 2
            pad_right_w = pad_w - pad_left_w

            padding = (
                pad_left_w,
                pad_right_w,
                pad_left_h,
                pad_right_h,
                pad_left_d,
                pad_right_d,
            )

            x = F.pad(x, padding, mode=self.pad_mode, value=self.constant_values)

            d[key] = x
        return d


def get_data_loader():
    config = get_config("config.yaml")
    reader = TiffReader()
    transforms = Compose(
        [
            LoadImageD(keys=["image", "label"], reader=reader),
            ToTensorD(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.0, upper=97.0, b_min=0.0, b_max=1.0, clip=True
            ),
            Lambdad(
                keys=["label"], func=lambda x: x - 1
            ),  # AsDiscreted expects labels to start from 0
            AsDiscreted(keys=["label"], to_onehot=4),
            Pad(keys=["image", "label"], target_shape=(640, 704, 576), constant_values=3),
            SplitData(keys=["image", "label"], num_splits=20, split_dim=1),
        ]
    )

    data_dicts = [{"image": config.data_path, "label": config.label_path}]
    base_dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)

    logger.info(f"Dataset length: {len(base_dataset)}")

    # Wrap the base dataset to flatten the record into multiple samples.
    flat_dataset = FlattenSplitDataset(base_dataset)

    logger.info(f"Flattened dataset length: {len(flat_dataset)}")

    data_loader = DataLoader(
        flat_dataset, batch_size=1, shuffle=False, pin_memory=config.cuda
    )
    logger.info(f"Data loader created with {len(data_loader)} batches.")
    for batch in data_loader:
        logger.info(f"Batch shape: {batch['image'].shape}, {batch['label'].shape}")
        break
    return data_loader
