import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    length = length.to(device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()
    other_keys = [key for key in keys if key not in ["x", "tx"]]
    
    try:
        batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    except:
        for key in other_keys:
            print("KEY:", key)
            print("VALUE:", value)

    x = collate_tensor_with_padding([x["x"] for x in lst_elements])
    
    if device is not None:
        x = x.to(device)

    batch["x"] = x
    
    if "length" in batch:
        batch["mask"] = length_to_mask(batch["length"], device=x.device)

    # text embeddings
    if "tx" in keys:
        assert "x" in one_el["tx"]
        assert "length" in one_el["tx"]
        tx_x = collate_tensor_with_padding([x["tx"]["x"] for x in lst_elements]).to(device)
        tx_length = default_collate([x["tx"]["length"] for x in lst_elements]).to(device)
        tx_mask = length_to_mask(tx_length, device=tx_x.device)
        batch["tx"] = {"x": tx_x.to(device), "length": tx_length.to(device), "mask": tx_mask.to(device)}

    if "tx_uncond" in keys:
        # only one is enough
        batch["tx_uncond"] = one_el["tx_uncond"]
    
    return batch


def collate_text_motion_text_ensemble(lst_elements: List, *, device = None):
    # To be used with all_texts=True in load_keyid of dataset class
    one_el = lst_elements[0]
    keys = one_el.keys()

    other_keys = [key for key in keys if key not in ["x", "tx", "text"]]
    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    batch["text"] = [elt["text"] for elt in lst_elements]
    
    x = collate_tensor_with_padding([x["x"] for x in lst_elements])
    if device is not None:
        x = x.to(device)

    batch["x"] = x
    
    if "length" in batch:
        batch["mask"] = length_to_mask(batch["length"], device=x.device)

    batch["text_ensemble_indices"] = []
    current_index = 0
    for elt in lst_elements:
        batch["text_ensemble_indices"].append((current_index, current_index + len(elt["text"])))
        current_index += len(elt["text"])

    # text embeddings
    if "tx" in keys:
        assert "x" in one_el["tx"][0]
        assert "length" in one_el["tx"][0]
        lst_flat = [tx for elt in lst_elements for tx in elt["tx"]]
        tx_x = collate_tensor_with_padding([tx["x"] for tx in lst_flat]).to(device)
        tx_length = default_collate([tx["length"] for tx in lst_flat]).to(device)
        tx_mask = length_to_mask(tx_length, device=tx_x.device)
        batch["tx"] = {"x": tx_x, "length": tx_length, "mask": tx_mask}

    if "tx_uncond" in keys:
        # only one is enough
        batch["tx_uncond"] = one_el["tx_uncond"]

    return batch

