"""
This script computes the motion statistics for the BOBSL3DT dataset which is too large to fit in memory.
"""
import hydra
import logging
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

import psutil


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="motion_stats", version_base="1.3")
def motion_stats(cfg: DictConfig):
    logger.info("Computing motion stats")

    train_dataset = instantiate(cfg.data, split="train")

    mean = 0
    mean_square = 0
    num_frames = 0
    logging.info('Computing mean and std')

    for x in tqdm(train_dataset):
        feats = x["x"]
        n = feats.shape[0]
        batch_mean = feats.mean(0)
        batch_mean_square = (feats ** 2).mean(0)
        mean = (batch_mean * n) / (num_frames + n) + (mean * num_frames) / (num_frames + n)
        mean_square = (batch_mean_square * n) / (num_frames + n) + (mean_square * num_frames) / (num_frames + n)
        num_frames = num_frames + n

    std = torch.sqrt(torch.abs(mean_square - mean ** 2))

    normalizer = instantiate(cfg.motion_normalizer)
    logger.info(f"Saving them in {normalizer.base_dir}")
    normalizer.save(mean, std)


if __name__ == "__main__":
    motion_stats()
