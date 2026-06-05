import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="text_stats", version_base="1.3")
def text_stats(cfg: DictConfig):
    logger.info("Computing text stats")

    train_dataset = instantiate(cfg.data, split="train")

    normalizer = instantiate(cfg.text_normalizer)
    import torch
    
    feats = torch.cat([x["tx"]["x"] for x in tqdm(train_dataset)])
    mean = feats.mean(0)
    std = feats.std(0)

    logger.info(f"Saving them in {normalizer.base_dir}")
    normalizer.save(mean, std)


if __name__ == "__main__":
    text_stats()
