#From cslr2-main
from pathlib import Path
from typing import List
import lmdb
import numpy as np


def lmdb_key_list(episode_name: str, begin_frame: int, end_frame: int) -> List:
    """
    Returns list of keys for episode

    Args:
        episode_name (str): Episode name.
        begin_frame (int): Begin frame.
        end_frame (int): End frame.

    Returns:
        List: List of keys mapping to RGB frames in lmdb environment.
    """
    return [f"{Path(episode_name.split('.')[0])}/{frame_idx + 1:07d}.np".encode('ascii') \
            for frame_idx in range(begin_frame, end_frame + 1)]


def get_frame_feats(lmdb_keys: List[str], lmdb_env: lmdb.Environment) -> List:
    """
    Returns list of episode frames

    Args:
        lmdb_keys (List[str]): List of keys mapping to episode frames in lmdb environment.
        lmdb_env (lmdb.Environment): lmdb environment.

    Returns:
        frames (List): List of episode frames.
    """
    frames = []
    for key in lmdb_keys:
        try:
            with lmdb_env.begin() as txn:
                frame_feats = txn.get(key)
            frame_feats = np.frombuffer(frame_feats, dtype=np.float16)
            frames.append(frame_feats)
        except:
             break
    return np.array(frames)

