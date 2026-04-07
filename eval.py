import numpy as np
import glob
import hydra
import json
import logging
import os
import pickle
import pytorch_lightning as pl
import random
import smplx
import torch
import yaml

from collections import defaultdict
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from prepare.motion_feats_to_smpl import np_feats_to_smplx
from src.config import read_config
from src.data.collate import collate_tensor_with_padding, collate_text_motion
from src.tools.constants import UPPER_BODY_HANDS_JOINT_IDX
from src.tools.geometry import to_matrix, matrix_to
from src.tools.vis import visualize_seqs
from src.model.smoothness_score import rotation_smoothness_loss_quaternion

from TMR.src.load import load_model_from_cfg as load_tmr_model_from_cfg
from TMR.src.model.metrics import (
    calculate_activation_statistics, calculate_fid, retrieval, print_latex_metrics, save_metric,
    all_contrastive_metrics_motion_to_gt_motions, all_contrastive_metrics_motion_to_text, 
)
from TMR.src.model.tmr import get_sim_matrix
from TMR.src.model.temos import length_to_mask
from TMR.src.data.collate import collate_x_dict


logger = logging.getLogger(__name__)

_TMR_FOLDER = "./TMR"


random.seed(0)

def compute_features(diffusion, retrieval_model, dataset, tmr_text_to_token_emb, keyids, guidance,
        batch_size=256, inference_saving_path=None, fixed_length=None):

    device = diffusion.device

    nsplit = int(np.ceil(len(keyids) / batch_size))
    inference_list = []

    needs_inference = True

    if os.path.isfile(inference_saving_path):
        logger.info("Loading generated motion from file")
        with open(inference_saving_path, "rb") as f:
            inference_list = pickle.load(f)
            previous_keyids = pickle.load(f)
        try:
            assert previous_keyids == keyids, "keyids are different from previous run"
            inference_list = np.array(inference_list, dtype=object)
            needs_inference = False
            inference_list_cursor = 0
        except Exception as e:
            logger.error(f"When loading the inference results, the following error occured: {e}")
            inference_list = []
            pass

    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid, all_texts=False) for keyid in keyids]
        description_texts = [elt["text"] for elt in all_data]

        all_data_splitted = np.array_split(all_data, nsplit)

        gt_latent_motions = []
        generated_latent_motions = []
        generated_motions = []
        for batch_idx, data in tqdm(enumerate(all_data_splitted), leave=True):
            #batch = collate_text_motion_text_ensemble(data, device=device)
            batch = collate_text_motion(data, device=device)
            if fixed_length is not None:
                lengths = torch.tensor([fixed_length for _ in range(len(batch["length"]))])
            else:
                lengths = batch["length"]
            infos = {
                "all_lengths": lengths,
                "all_texts": batch["text"],
                "guidance_weight": guidance
            }
            if needs_inference:
                tx_emb = batch["tx"]
                tx_emb_uncond = torch.stack([data[0]["tx_uncond"]["x"] for _ in range(len(tx_emb["x"]))]).to(device)

                tx_emb_uncond = {
                        "x": tx_emb_uncond,
                        "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(device),
                        }
                xstarts = diffusion(tx_emb, tx_emb_uncond, infos)
                for i in range(len(xstarts)):
                    inference_list.append(xstarts[i, :infos["all_lengths"][i]].detach().cpu().numpy())
            else:
                batch_xstarts = inference_list[inference_list_cursor: (inference_list_cursor + len(infos["all_texts"]))]
                inference_list_cursor += len(infos["all_texts"])
                xstarts = collate_tensor_with_padding([torch.Tensor(elt).to(device) for elt in batch_xstarts])
                xstarts = xstarts[..., : diffusion.motion_normalizer.mean.shape[0]] # for the no face thing

            # xstarts: not normalized
            if diffusion.motion_normalizer.mean.shape[0] == 284:
                insert = torch.zeros((batch["x"].shape[0], batch["x"].shape[1], 10)).to(device)
                if batch["x"].shape[-1] == 274:
                    batch["x"] = torch.cat([batch["x"][..., 0: 264], insert, batch["x"][..., 264:274]], axis=2)

            if diffusion.motion_normalizer.mean.shape[0] == 258:
                batch["x"] = batch["x"][..., : 258]

            if batch["x"].shape[-1] == 284:
                batch["x"] = torch.cat([batch["x"][..., 0:264], batch["x"][..., 274:284]], dim=2)

            gt_motion_x_dict = {"x": diffusion.motion_normalizer(batch["x"]),
                    "length": batch["length"], "mask": batch["mask"]}
            generated_motion_x_dict = {"x": diffusion.motion_normalizer(xstarts),
                    "length": infos["all_lengths"],
                    "mask": length_to_mask(infos["all_lengths"], device=device)}

            for i in range(len(xstarts)):
                generated_motions.append(xstarts[i][:lengths[i]].detach().cpu())

            # Encode both gt motion and generated
            if gt_motion_x_dict["x"].shape[-1] == 284:
                gt_motion_x_dict["x"] = torch.cat([gt_motion_x_dict["x"][..., 0:264], gt_motion_x_dict["x"][..., 274:284]], dim=2)
            if generated_motion_x_dict["x"].shape[-1] == 284:
                generated_motion_x_dict["x"] = torch.cat([generated_motion_x_dict["x"][..., 0:264], generated_motion_x_dict["x"][..., 274:284]], dim=2)

            if retrieval_model.motion_encoder.nfeats == 180:
                gt_motion_x_dict["x"] = gt_motion_x_dict["x"][..., 78: 258]
                generated_motion_x_dict["x"] = generated_motion_x_dict["x"][..., 78: 258]
            elif retrieval_model.motion_encoder.nfeats == 216:
                gt_motion_x_dict["x"] = gt_motion_x_dict["x"][..., 42: 258]
                generated_motion_x_dict["x"] = generated_motion_x_dict["x"][..., 42: 258]
            if retrieval_model.motion_encoder.nfeats == 258:
                gt_motion_x_dict["x"] = gt_motion_x_dict["x"][..., : 258]
                generated_motion_x_dict["x"] = generated_motion_x_dict["x"][..., : 258]

            gt_latent_motion = retrieval_model.encode(gt_motion_x_dict, sample_mean=True)
            generated_latent_motion = retrieval_model.encode(generated_motion_x_dict, sample_mean=True)

            gt_latent_motions.append(gt_latent_motion)
            generated_latent_motions.append(generated_latent_motion)

        latent_texts = []

        unique_texts = np.unique(description_texts)
        unique_texts_encoded = [tmr_text_to_token_emb(text) for text in unique_texts]
        unique_texts_encoded_splitted = np.array_split(unique_texts_encoded, nsplit)

        for text_batch in unique_texts_encoded_splitted:
            text_dict = collate_x_dict(text_batch)
            generated_latent_text = retrieval_model.encode(text_dict, sample_mean=True)
            latent_texts.append(generated_latent_text)

        gt_latent_motions = torch.cat(gt_latent_motions)
        generated_latent_motions = torch.cat(generated_latent_motions)
        latent_texts = torch.cat(latent_texts)

        if needs_inference and inference_saving_path is not None:
            with open(inference_saving_path, "wb") as f:
                pickle.dump(generated_motions, f)
                pickle.dump(keyids, f)

    returned = {
        "keyids": keyids,
        "texts": description_texts,
        "unique_texts": unique_texts,
        "latent_texts": latent_texts,
        "gt_latent_motions": gt_latent_motions,
        "generated_latent_motions": generated_latent_motions,
        "generated_motions": generated_motions,
    }
    return returned


@hydra.main(config_path="configs", config_name="metrics", version_base='1.3') # TODO
def metrics(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt
    batch_size = newcfg.batch_size
    split = newcfg.split
    guidance = newcfg.guidance

    tmr_run_dir = os.path.join(_TMR_FOLDER, newcfg.tmr_run_dir)
    tmr_ckpt_name = newcfg.tmr_ckpt

    ckpt_path = glob.glob(os.path.join(newcfg.run_dir, f"**/checkpoints/{ckpt_name}.ckpt"), recursive=True)[0]
    logger.info("Loading the checkpoint")
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = ckpt["epoch"]
    #print(f"EPOCH {epoch}") # do not delete, used in bash script

    cfg = read_config(run_dir)

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])
    diffusion.eval()
    diffusion.to(device)

    # Load TMR model #######################################################################
    tmr_cfg = read_config(tmr_run_dir)
    for key, val in tmr_cfg.model.items():
        if type(val) == DictConfig:
            for k, v in val.items():
                if type(v) == str:
                    val[k] = v.replace("src", "TMR.src")
        if type(val) == str:
            tmr_cfg.model[key] = val.replace("src", "TMR.src")

    tmr_model, tmr_epoch = load_tmr_model_from_cfg(tmr_cfg, tmr_ckpt_name, eval_mode=True, device=device, return_epoch=True, reload_model=False)

    tmr_cfg.data.text_to_token_emb._target_ = tmr_cfg.data.text_to_token_emb._target_.replace("src", "TMR.src")
    tmr_cfg.data.text_to_token_emb.path = os.path.join(_TMR_FOLDER, "datasets", "annotations", newcfg.tmr_dataset)
    tmr_cfg.data.text_to_token_emb.device = device
    tmr_cfg.data.text_to_token_emb.preload = False
    tmr_text_to_token_emb = instantiate(tmr_cfg.data.text_to_token_emb)
    ########################################################################################

    seed = newcfg.seed
    if seed is None:
        seed = cfg.seed
    pl.seed_everything(seed)

    logger.info("Loading the model")

    train_dataset = cfg.data.name
    train_split = cfg.train_split
    train_annotations_path = os.path.join("datasets/annotations", train_dataset, "annotations.json")
    train_split_path = os.path.join("datasets/annotations", train_dataset, "splits", f"{train_split}.txt")

    with open(train_annotations_path, "r") as f:
        train_annotations = json.load(f)
    with open(train_split_path, "r") as f:
        train_split = f.read().split("\n")

    description2keyids = defaultdict(list)
    for keyid in train_split:
        if keyid in train_annotations:
            for annot in train_annotations[keyid]["annotations"]:
                description2keyids[annot["text"]].append(keyid)

    data = newcfg.data
    dataset = instantiate(data, split=split)
    
    dataset_camel = newcfg.dataset.split("_")
    dataset_camel = dataset_camel[0] + "".join([elt.capitalize() for elt in dataset_camel[1:]])
    tmr_run_dir_camel = "_".join(tmr_run_dir.split("outputs/")[1].split("/")).split('_')
    tmr_run_dir_camel = tmr_run_dir_camel[0] + "".join([elt.capitalize() for elt in tmr_run_dir_camel[1:]])
    tmr_run_dir_camel = f"{tmr_run_dir_camel}_epoch{tmr_epoch}"

    inference_folder_path = os.path.join(run_dir, f"metrics/{dataset_camel}/{split}/guidance{guidance}_epoch{epoch}")
    metrics_folder_path = os.path.join(inference_folder_path, tmr_run_dir_camel)
    inference_path = os.path.join(inference_folder_path, f"inference_{seed}.pickle")

    os.makedirs(metrics_folder_path, exist_ok=True)

    # Compute sim_matrix for each protocol
    result = compute_features(
        diffusion, tmr_model, dataset, tmr_text_to_token_emb, dataset.keyids, guidance, batch_size=batch_size,
        inference_saving_path=inference_path, fixed_length=newcfg.fixed_length
    )

    # Compute the metrics
    # Retrieval
    keyids = result["keyids"]
    gt_latent_motions = result["gt_latent_motions"]
    generated_latent_motions = result["generated_latent_motions"]
    latent_texts = result["latent_texts"]
    m2m_sim_matrix = get_sim_matrix(generated_latent_motions, gt_latent_motions).cpu().numpy()
    m2t_sim_matrix = get_sim_matrix(generated_latent_motions, latent_texts).cpu().numpy()
    m2t_gt_sim_matrix = get_sim_matrix(gt_latent_motions, latent_texts).cpu().numpy()

    unique_texts = result["unique_texts"]
    texts = result["texts"]
    generated_motions = result["generated_motions"]
    acc_score = []
    smooth_score = []
    jerk_score = []

    kwargs = dict(gender='neutral',
        num_betas=10,
        create_betas=True,
        betas=None,
        use_face_contour=True,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=1
    )
    smplx_model = smplx.create(
        '/lustre/fswork/projects/rech/qgw/ujv31bi/bobsl_smplerx/common/utils/human_model_files', 'smplx',
        **kwargs
    )
    smplx_model = smplx_model.to("cuda")
    smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3),
                'body_pose': (-1, 21, 3),
                'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3),
                'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3),
                'expression': (-1, 10)}

    for motion in generated_motions:
        quaternion = matrix_to("quaternion", to_matrix("rot6d", motion[:, : 258].view(motion.shape[0], -1, 6)))
        loss = rotation_smoothness_loss_quaternion(quaternion)
        acc_score.append(loss["loss_acc"].item())
        smooth_score.append(loss["loss_smooth"].item())
        smplx_motion = np_feats_to_smplx(motion)

        # Extract 3D keypoints using SMPLX library
        smplx_3dkp = []
        intersect_key = list(set(smplx_motion[0].keys()) & set(smpl_shape.keys()))
        for frame_smplx in smplx_motion:
            body_model_param_tensor = {
                key: torch.tensor(np.array(frame_smplx[key]).reshape(smpl_shape[key]), device='cuda', dtype=torch.float32)
                    for key in intersect_key if len(frame_smplx[key]) > 0
            }        
            smplx_output = smplx_model(**body_model_param_tensor, return_verts=True, return_dict=True)
            joints = smplx_output["joints"][0].to("cpu").detach().numpy()
            smplx_3dkp.append(joints[UPPER_BODY_HANDS_JOINT_IDX])
        smplx_3dkp = np.stack(smplx_3dkp)
        fps = 25
        dt = 1.0 / fps
        jerk_pred = (smplx_3dkp[3:] - 3*smplx_3dkp[2:-1] + 3*smplx_3dkp[1:-2] - smplx_3dkp[:-3]) / (dt**3)
        jerk_score.append(np.mean(abs(jerk_pred[:, 2:])))

    scores = {
        "acc_score": round(torch.mean(torch.tensor(acc_score)).item(), 6), 
        "smooth_score": round(torch.mean(torch.tensor(smooth_score)).item(), 6),
        "jerk_score": round(float(np.mean(jerk_score)), 6)
    }
    
    metrics, cols_motion, gt_cols_motion = all_contrastive_metrics_motion_to_gt_motions(m2m_sim_matrix, texts, return_cols=True)
    metrics_text, cols_text, gt_cols_text = all_contrastive_metrics_motion_to_text(m2t_sim_matrix, texts, unique_texts, return_cols=True)

    metrics_gt_text, cols_gt_text, gt_cols_text_gt = all_contrastive_metrics_motion_to_text(m2t_gt_sim_matrix, texts, unique_texts, return_cols=True)

    metrics.update(metrics_text)
    metrics.update(scores)

    # FID
    gt_stats = calculate_activation_statistics(gt_latent_motions, normalize=True)
    generated_stats = calculate_activation_statistics(generated_latent_motions, normalize=True)

    fid = calculate_fid(gt_stats, generated_stats)
    metrics["FID"] = float(round(fid, 3))

    # GT FID
    indices = list(range(gt_latent_motions.shape[0]))
    random.shuffle(indices)

    indices1 = indices[: len(indices) // 2]
    indices2 = indices[len(indices) // 2:]

    gt_stats1 = calculate_activation_statistics(gt_latent_motions[indices1], normalize=True)
    gt_stats2 = calculate_activation_statistics(gt_latent_motions[indices2], normalize=True)

    gt_fid = calculate_fid(gt_stats1, gt_stats2)

    if newcfg.save_metrics:
        metrics["unique_texts/len"] = len(unique_texts)

        keys = ["m2m/R01", "m2m/R02", "m2m/R03", "m2m/R05", "m2m/R10", "m2t/R01", "m2t/R02", "m2t/R03", "m2t/R05", "m2t/R10", "FID", "acc_score", "smooth_score"]
        metric_str = print_latex_metrics(metrics, keys=keys, ranks=[1, 2, 3, 5, 10], m2m=True, m2t=True, MedR=False, fid=True)

        protocol_name = "metrics"
        metric_name = f"{protocol_name}_{seed}.yaml"
        path = os.path.join(metrics_folder_path, metric_name)

        save_metric(path, metrics, metric_str=metric_str)
        
        gt_keys = ['m2t/R01', 'm2t/R02', 'm2t/R03', 'm2t/R05', 'm2t/R10']
        print("GT")
        _ = print_latex_metrics(metrics_gt_text, keys=gt_keys, ranks=[1, 2, 3, 5, 10], m2m=False, m2t=True, MedR=False, fid=False)

        logger.info(f"Evaluation done, metrics saved in:\n{path}")

    if newcfg.render:
        m2m_keyids = [keyids[idx] for idx in gt_cols_motion]
        m2m_descriptions = [texts[idx] for idx in gt_cols_motion]

        motion_retrieved_keyids = retrieval(m2m_sim_matrix, 3, keyids)

        to_save = {
                keyids[i]: {
                    "text": texts[i],
                    "generated_motion": generated_motions[i].detach().cpu(),
                    "motion_rank": cols_motion[i],
                    "m2m_keyid": m2m_keyids[i],
                    "m2m_description": m2m_descriptions[i],
                    "motion_retrieved_keyid": motion_retrieved_keyids[i]
                }
            for i in range(len(keyids))
        }
        
        generation_folder_path = os.path.join(run_dir.replace("outputs/", "rendering_inference/"), dataset_camel, split, f"guidance{guidance}_epoch{epoch}_seed{seed}")
        rendering_folder_path = os.path.join(generation_folder_path, "renderings")
        mesh_folder_path = os.path.join(generation_folder_path, "meshes")
        os.makedirs(mesh_folder_path, exist_ok=True)
        generation_eval_folder_path = os.path.join(generation_folder_path, f"TMR_{tmr_run_dir_camel}")
        os.makedirs(generation_eval_folder_path, exist_ok=True)

        test_gt_folder_path = os.path.join("rendering_ground_truth", newcfg.motion_test_gt_folder)
        test_gt_mesh_folder_path = os.path.join("rendering_ground_truth/meshes", newcfg.motion_test_gt_folder)
        os.makedirs(test_gt_folder_path, exist_ok=True)
        os.makedirs(test_gt_mesh_folder_path, exist_ok=True)

        render_max = newcfg.render_max
        if render_max is None:
            render_max = len(to_save)

        saved_descriptions = set()
        counter = 0
        for keyid, val in to_save.items():
            #if to_save[keyid]["motion_rank"] > 2:
            #    continue
            description = val["text"]
            if description in saved_descriptions:
                continue
            mp4_path = os.path.join(rendering_folder_path, f"{keyid}.mp4")
            mesh_path = os.path.join(mesh_folder_path, f"{keyid}.npy")
            if (not os.path.exists(mp4_path) and not os.path.isdir(mp4_path.replace(".mp4", ""))) or (not os.path.exists(mesh_path)) or newcfg.overwrite_rendering:
                smplx_params = np_feats_to_smplx(val["generated_motion"])
                os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
                mesh = visualize_seqs(mp4_path, seqs=smplx_params, delete_img=True, frame_range=None, with_mano=newcfg.with_mano)
                np.save(mesh_path, mesh)

            gt_mp4_path = os.path.join(test_gt_folder_path, f"{keyid}.mp4")
            gt_mesh_path = os.path.join(test_gt_mesh_folder_path, f"{keyid}.npy")
            if not os.path.exists(gt_mp4_path) and not os.path.isdir(gt_mp4_path.replace(".mp4", "")):
                seq = dataset.load_keyid(keyid)["x"]
                gt_smplx_params = np_feats_to_smplx(seq)
                os.makedirs(os.path.dirname(gt_mp4_path), exist_ok=True)
                mesh = visualize_seqs(gt_mp4_path, seqs=gt_smplx_params, delete_img=True, frame_range=None, with_mano=False)
                np.save(gt_mesh_path, mesh)

            with open(os.path.join(generation_eval_folder_path, f"{keyid}.pickle"), "wb") as f:
                pickle.dump(to_save[keyid], f)

            saved_descriptions.add(val["text"])
            counter += 1
            if counter >= render_max:
                break


if __name__=="__main__":
    metrics()

