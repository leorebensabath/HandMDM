import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        # only during training
        drop_cond: float = 0.10,
    ):
        if tiny:
            split = split + "_tiny"
        
        path = f"datasets/annotations/{name}"
        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)
        
        self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        self.is_training = split.startswith("train")
        self.drop_cond = drop_cond

        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.keyids_one_text = [keyid for keyid in self.keyids if len(self.annotations[keyid]["annotations"]) == 1]  

        if os.environ.get("EXP_DATASET_EW", "0") == "1": # TODELETE
            keyids = []
            all_videos = os.listdir("/lustre/fswork/projects/rech/qgw/ujv31bi/stmc_bobsl/datasets/motions/bobsl_w_mano_6d_v6_ew")
            all_videos = [elt.replace(".npy", "") for elt in all_videos]
            for keyid in self.keyids:
                if self.annotations[keyid]["path"] in all_videos:
                    keyids.append(keyid)
            self.keyids = keyids
            
        self.nfeats = self.motion_loader.nfeats
        
        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)
    
    def get_all_texts(self):
        texts = []
        for val in self.annotations.values():
            for elt in val["annotations"]:
                texts.append(elt["text"])
        return texts

    def get_unique_texts(self):
        return set(self.get_all_texts())


    def load_keyid(self, keyid, all_texts=False, random_text=False, index=None):
        annotations = self.annotations[keyid]
        
        # Take the first one for testing/validation
        # Otherwise take a random one
        if not all_texts:
            if index is not None:
                index = index
            elif self.is_training or random_text:
                index = np.random.randint(len(annotations["annotations"]))
            else:
                index = 0
            annotation = annotations["annotations"][index]
            text = annotation["text"]
        else:
            text = [annotations["annotations"][index]["text"] for index in range(len(annotations["annotations"]))]
            annotation = annotations["annotations"][0]
        
        if self.is_training:
            drop_cond = self.drop_cond
            if drop_cond is not None:
                if np.random.binomial(1, drop_cond) == 1:
                    # uncondionnal
                    text = ""

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"])
        if not all_texts:
            text_encoded = self.text_encoder(text)
        else:
            text_encoded = [self.text_encoder(t) for t in text] 
        text_uncond_encoded = self.text_encoder("")

        x = motion_x_dict["x"]
        length = motion_x_dict["length"]

        output = {
            "x": x,
            "text": text,
            "tx": text_encoded,
            "tx_uncond": text_uncond_encoded,
            "keyid": keyid,
            "length": length,
        }
        return output
    
    """
    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]

            # remove humanact12
            # buggy left/right + no SMPL
            if "humanact12" in path:
                continue
            
            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations
    """

def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
