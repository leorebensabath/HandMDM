import glob
import hydra
import logging
import os
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.config import read_config, save_config

logger = logging.getLogger(__name__)


"""
def save_stats(dataset, run_dir):
    is_training = dataset.is_training
    # don't drop anything
    dataset.is_training = False

    motion_stats_dir = os.path.join(run_dir, "motion_stats")
    os.makedirs(motion_stats_dir, exist_ok=True)

    text_stats_dir = os.path.join(run_dir, "text_stats")
    os.makedirs(text_stats_dir, exist_ok=True)

    from tqdm import tqdm
    import torch
    from src.normalizer import Normalizer

    logger.info("Compute motion embedding stats")
    motionfeats = torch.cat([x["x"] for x in tqdm(dataset)])
    mean_motionfeats = motionfeats.mean(0)
    std_motionfeats = motionfeats.std(0)

    motion_normalizer = Normalizer(base_dir=motion_stats_dir, disable=True)
    motion_normalizer.save(mean_motionfeats, std_motionfeats)

    logger.info("Compute text embedding stats")
    textfeats = torch.cat([x["tx"]["x"] for x in tqdm(dataset)])
    mean_textfeats = textfeats.mean(0)
    std_textfeats = textfeats.std(0)

    text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
    text_normalizer.save(mean_textfeats, std_textfeats)

    # re enable droping
    dataset.is_training = is_training
"""

@hydra.main(config_path="configs", config_name="train_bobsl", version_base="1.3")
def train(cfg: DictConfig):

    # Resuming if needed
    ckpt = cfg.ckpt
    if cfg.resume_dir is not None and os.path.exists(os.path.join(cfg.resume_dir, "config.json")):
        resume_dir = cfg.resume_dir
        max_epochs = cfg.trainer.max_epochs
        num_nodes = cfg.trainer.num_nodes
        devices = cfg.trainer.devices
        num_workers = cfg.dataloader.num_workers
        batch_size = cfg.dataloader.batch_size
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        cfg.trainer.max_epochs = max_epochs
        cfg.trainer.num_nodes = num_nodes
        cfg.trainer.devices = devices
        cfg.dataloader.num_workers = num_workers
        cfg.dataloader.batch_size = batch_size
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{resume_dir}")
        project_path = os.path.join(cfg.resume_dir, cfg.trainer.logger.project)
        breakpoint()
        if len(glob.glob(os.path.join(project_path, "*"))) > 0:
            project_id = os.listdir(project_path)[0]
            cfg.trainer.logger.id = project_id
        resume=True
    else:
        resume_dir = None
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")
        resume=False
        
    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")
    
    if cfg.val_split is None:
        cfg.val_split = cfg.train_split

    train_dataset = instantiate(cfg.data, split=cfg.train_split)
    val_dataset = instantiate(cfg.data, split=cfg.val_split)
    
    limit_val_batches = 1.0
    # If train_split and val_split are the same, we skip the validation step. 
    if cfg.train_split == cfg.val_split: 
        limit_val_batches = 0.0

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )
    
    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)
    
    logger.info("Training")
    
    trainer = instantiate(cfg.trainer, limit_val_batches=limit_val_batches)
    
    # TODO: comment on mean and std that are poped
    if ckpt is not None and not resume and os.path.isfile(ckpt):
        state_dict = torch.load(ckpt)
        _ = state_dict["state_dict"].pop("motion_normalizer.mean")
        _ = state_dict["state_dict"].pop("motion_normalizer.std")
        diffusion.load_state_dict(state_dict["state_dict"], strict=False)
        trainer.fit(diffusion, train_dataloader, val_dataloader)
    else:
        trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
