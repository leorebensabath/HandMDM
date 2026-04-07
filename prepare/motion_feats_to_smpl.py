import numpy as np
import torch
from src.tools.geometry import to_matrix, matrix_to


body_pose_lower_joints = [0, 1, 3, 4, 6, 7, 9, 10]
smplx_keys_284 = {'body_pose': (0, 78), 'left_hand_pose': (78, 168), 'right_hand_pose': (168, 258), 'jaw_pose': (258, 264), 'betas': (264, 274), 'expression': (274, 284)}
smplx_keys_274 = {'body_pose': (0, 78), 'left_hand_pose': (78, 168), 'right_hand_pose': (168, 258), 'jaw_pose': (258, 264), 'expression': (264, 274)}

def np_feats_to_smplx(smplx_np):
    smplx = []
    for frame in smplx_np:
        smplx_params = {}
        if frame.shape[-1] == 284:
            smplx_keys = smplx_keys_284
        elif frame.shape[-1] == 274:
            smplx_keys = smplx_keys_274
        for key in smplx_keys: 
            start, end = smplx_keys[key][0], smplx_keys[key][1]

            params = frame[start:end]

            if key in ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']:
                params = params.reshape((-1,6))
                params_torch = torch.tensor(params)
                params = - matrix_to("axisangle", to_matrix("rot6d", params_torch)).detach().numpy()
                if key == 'body_pose':
                    for idx in body_pose_lower_joints:
                        params = np.insert(params, idx, [0.0, 0.0, 0.0], axis=0)
            else:
                params = params.reshape((1, -1))
            smplx_params[key] = params

        smplx_params["global_orient"] = np.array([[2.9, -0.005, -0.1]])
        smplx_params["transl"] = np.array([[-1.8e-02,  7.0e-01,  3.0e+01]])
        smplx_params["reye_pose"] = np.array([[0.0, 0.0, 0.0]])
        smplx_params["leye_pose"] = np.array([[0.0, 0.0, 0.0]])

        smplx.append(smplx_params)

    return smplx

"""
#npy_file = "/linkhome/rech/genlgm01/ujv31bi/TMR_bobsl/datasets/motions/bobsl_w_mano_6d/5085344787448740525.npy"
npy_file = "experiments/5587444932710561032_75.92_76.8.npy"
pickle_file = "/linkhome/rech/genlgm01/ujv31bi/prepare_bobsl/experiments/5587444932710561032_75.92_76.8.pickle"

smplx_np = np.load(npy_file)
smplx = np_feats_to_smplx(smplx_np)

with open(pickle_file, "wb") as f:
    pickle.dump(smplx, f)
"""

