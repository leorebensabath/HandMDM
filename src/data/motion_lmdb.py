import lmdb 
import os
import torch
from src.data.lmdb_utils import lmdb_key_list, get_frame_feats


class LmdbMotionLoader:
    def __init__(
        self, base_dir, fps, normalizer=None, disable: bool = False, nfeats=None, umin_s=0.4, umax_s=2.0
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.normalizer = normalizer
        self.disable = disable
        self.nfeats = nfeats

        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)
        self.lmdb_env = None
        self.lmdb_env = lmdb.open(
            self.base_dir, readonly=True, lock=False, max_readers=512
        )

    def __call__(self, path, start, end):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}
       
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.base_dir, readonly=True, lock=False, max_readers=512
            )

        begin = int(start * self.fps)
        end = int(end * self.fps)
        video_name = path

        lmdb_keys = lmdb_key_list(video_name,
                                  begin_frame=begin,
                                  end_frame=end - 1)

        motion = get_frame_feats(lmdb_keys, self.lmdb_env)
        motion = torch.from_numpy(motion).to(torch.float)
            
        if self.normalizer is not None:
            motion = self.normalizer(motion)

        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
