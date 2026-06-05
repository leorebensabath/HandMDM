import hydra
import logging
import os
import re
import torch

import numpy as np
import pytorch_lightning as pl

from hydra.utils import instantiate
from omegaconf import DictConfig

from prepare.motion_feats_to_smpl import np_feats_to_smplx
from src.config import read_config
from src.data.collate import collate_tensor_with_padding, length_to_mask
from src.tools.vis import visualize_seqs

logger = logging.getLogger(__name__)


def find_run_dir(checkpoint_path: str) -> str:
    path = os.path.abspath(os.path.dirname(checkpoint_path))
    while True:
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    raise FileNotFoundError(
        f"Could not find config.json for checkpoint: {checkpoint_path}"
    )


def resolve_checkpoint_path(checkpoint: str) -> str:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return os.path.abspath(checkpoint)


def sanitize_filename(text: str) -> str:
    sanitized = re.sub(r"[^\w\s-]", "", text.lower())
    sanitized = re.sub(r"[\s_-]+", "_", sanitized).strip("_")
    return sanitized[:80] or "motion"


def encode_text(text_encoder, text: str):
    x_dict = text_encoder(text)
    if isinstance(x_dict, torch.Tensor):
        x = x_dict[0] if x_dict.dim() > 1 else x_dict
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return {"x": x, "length": len(x)}
    return x_dict


def prepare_text_batch(tx_emb, device):
    tx_x = collate_tensor_with_padding([tx_emb["x"]]).to(device)
    tx_length = torch.tensor([tx_emb["length"]], device=device)
    return {
        "x": tx_x,
        "length": tx_length,
        "mask": length_to_mask(tx_length, device=device),
    }


@torch.inference_mode()
def generate_motion(diffusion, text_encoder, input_text, length, guidance, device):
    tx_emb = prepare_text_batch(encode_text(text_encoder, input_text), device)
    tx_emb_uncond_raw = encode_text(text_encoder, "")
    tx_emb_uncond = {
        "x": torch.stack(
            [tx_emb_uncond_raw["x"] for _ in range(len(tx_emb["x"]))]
        ).to(device),
        "length": torch.tensor(
            [tx_emb_uncond_raw["length"] for _ in range(len(tx_emb["x"]))]
        ).to(device),
    }

    lengths = torch.tensor([length], device=device)
    infos = {
        "all_lengths": lengths,
        "all_texts": [input_text],
        "guidance_weight": guidance,
    }

    motion = diffusion(tx_emb, tx_emb_uncond, infos, progress_bar=None)
    return motion[0, :length].detach().cpu().numpy()


@hydra.main(config_path="configs", config_name="inference", version_base="1.3")
def inference(cfg: DictConfig) -> None:
    device = cfg.device
    input_text = cfg.input_text
    guidance = cfg.guidance
    length = cfg.length

    ckpt_path = resolve_checkpoint_path(cfg.checkpoint)
    run_dir = find_run_dir(ckpt_path)
    logger.info(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = read_config(run_dir)
    model_cfg.data.text_encoder.device = device
    model_cfg.data.text_encoder.preload = True
    model_cfg.data.text_encoder.no_model = False

    seed = cfg.seed if cfg.seed is not None else model_cfg.seed
    pl.seed_everything(seed)

    diffusion = instantiate(model_cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])
    diffusion.eval()
    diffusion.to(device)

    text_encoder = instantiate(model_cfg.data.text_encoder)
    logger.info(f'Generating motion for: "{input_text}"')
    motion = generate_motion(
        diffusion, text_encoder, input_text, length, guidance, device
    )

    if cfg.output_path is not None:
        output_path = cfg.output_path
    else:
        output_dir = os.path.join(run_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{sanitize_filename(input_text)}_{length}.npy")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, motion)
    logger.info(f"Saved motion to {output_path}")

    if cfg.render:
        render_path = os.path.splitext(output_path)[0] + ".mp4"
        smplx_params = np_feats_to_smplx(motion)
        visualize_seqs(
            render_path,
            seqs=smplx_params,
            delete_img=True,
            frame_range=None,
            with_mano=cfg.with_mano,
        )
        logger.info(f"Saved rendering to {render_path}")


if __name__ == "__main__":
    inference()
