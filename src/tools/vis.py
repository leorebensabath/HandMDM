import cv2
import glob
import numpy as np
import os
import pickle
import pyrender
import smplx
import shutil
import torch
import trimesh
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from smplx.body_models import *


_BODY_MODEL_FOLDER = "/lustre/fswork/projects/rech/qgw/ujv31bi/bobsl_smplerx/common/utils/human_model_files"
focal_length = [
    14420.572,
    14420.572
]
principal_point = [
    221.5,
    221.5
]

img_shape = (444, 444, 3)

focal_length_mano = [
    5500,
    5500
]
principal_point_mano = [
    111,
    -100
]
img_shape_mano = (444, 222, 3)


def render_pose(img, body_model_param, body_model, body_model_name, camera, return_mask=False, render=True):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    if body_model_name == "mano":
        if "right_hand_pose" in body_model_param:
            body_model_param["hand_pose"] = body_model_param.pop("right_hand_pose").reshape((1, 45))
        else:
            body_model_param["hand_pose"] = body_model_param.pop("left_hand_pose").reshape((1, 45))
        #body_model_param["global_orient"] = body_model_param["global_orient"]
        body_model_param["global_orient"][0] = body_model_param["global_orient"][0] / 2
        body_model_param["transl"] = torch.tensor([[0.0000, 0.6000, 7.9500]])

    output = body_model(**body_model_param, return_verts=True, return_dict=True)
    
    #if body_model_name == "smplx" or body_model_name == "mano":
    if True:
        vertices = output['vertices'].detach().cpu().numpy().squeeze()
        faces = body_model.faces
    #elif body_model_name == "smplh":
    #    vertices = output["v"].detach().cpu().numpy().squeeze()
    #    faces = body_model.f
    
    if not render:
        return None, vertices

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)

    material_new = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.4,
        alphaMode='OPAQUE',
        emissiveFactor=(0.2, 0.2, 0.2),
        baseColorFactor=(0.7, 0.7, 0.7, 1)
    )
    material = material_new
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)

    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)


    img = (output_img * 255).astype(np.uint8)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img, vertices


def visualize_seqs(save_path, pickle_path=None, seqs=None, frame_range=None, img_folder=None, img_start=0, delete_img=False, fps=25, body_model="smplx", with_mano=False, mano_only=False, render=True):
    if seqs is None:
        with open(pickle_path, 'rb') as f:
            seqs = pickle.load(f)
    
    if body_model == "smplx":
        kwargs = dict(gender='neutral',
            num_betas=10,
            create_betas=True,
            betas=None,
            use_face_contour=True,
            flat_hand_mean=True,
            use_pca=False,
            batch_size=1
        )
    
        smpl_model = smplx.create(
            _BODY_MODEL_FOLDER, 'smplx',
            **kwargs
        )

        smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3),
                       'body_pose': (-1, 21, 3), 
                       'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3),
                       'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 
                       'expression': (-1, 10)}
    
    kwargs = dict(gender='neutral',
        num_betas=10,
        create_betas=True,
        betas=None,
        use_face_contour=True,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=1
    )

    if with_mano:
        kwargs = dict(gender='neutral',
            num_betas=10,
            create_betas=True,
            betas=None,
            use_face_contour=True,
            flat_hand_mean=True,
            use_pca=False,
            batch_size=1
        )

        mano_right_model = smplx.create(
            f"{_BODY_MODEL_FOLDER}/smplh/MANO_RIGHT.pkl", 'mano',
            **kwargs
        )
        mano_left_model = smplx.create(
            f"{_BODY_MODEL_FOLDER}/smplh/MANO_LEFT.pkl", 'mano',
            **kwargs
        )

        mano_smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3),
                       'right_hand_pose': (-1, 15, 3), 'left_hand_pose': (-1, 15, 3)}

    if with_mano:
        mano_camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length_mano[0], fy=focal_length_mano[1],
                        cx=principal_point_mano[0], cy=principal_point_mano[1])

    camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length[0], fy=focal_length[1],
                        cx=principal_point[0], cy=principal_point[1])

    if img_folder is not None:
        image_files = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
        if len(image_files) == 0:
            image_files = sorted(glob.glob(os.path.join(img_folder, "*.png")))

    frame_idx = 1
    if frame_range is None:
        frame_range = [0, len(seqs)]
    
    meshes = []
    for i in range(frame_range[0], frame_range[1]):
        seq = seqs[i]
        
        intersect_key = list(set(seq.keys()) & set(smpl_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(seq[key]).reshape(smpl_shape[key]), device='cpu', dtype=torch.float32)
                        for key in intersect_key if len(seq[key]) > 0}

        if img_folder is not None:
            try:
                image = cv2.imread(image_files[i + img_start])
            except: 
                breakpoint()
        else:
            image = np.zeros(img_shape)
            image.astype("uint8")
        
        rendered_image, vertices = render_pose(
            image, body_model_param_tensor,
            smpl_model, body_model,
            camera, render=render
        )
        meshes.append(vertices)
        
        if with_mano:
            mano_image = np.zeros(img_shape_mano)
            mano_image.astype("uint8")

            intersect_key = list(set(seq.keys()) & set(mano_smpl_shape.keys()))
            body_model_param_tensor = {
                key: torch.tensor(np.array(seq[key]).reshape(mano_smpl_shape[key]), 
                                  device='cpu', dtype=torch.float32)
                for key in intersect_key if len(seq[key]) > 0
            }
            rendered_image_right, _ = render_pose(img=mano_image,
                                               body_model_param=body_model_param_tensor,
                                               body_model=mano_right_model,
                                               body_model_name="mano",
                                               camera=mano_camera, render=render)
            rendered_image_right = np.flip(rendered_image_right, axis=1)
            rendered_image_left, _ = render_pose(img=mano_image,
                                               body_model_param=body_model_param_tensor,
                                               body_model=mano_left_model,
                                               body_model_name="mano",
                                               camera=mano_camera, render=render)
            rendered_image_left = np.flip(rendered_image_left, axis=1)
            
            if not mano_only:
                rendered_image = np.concatenate([rendered_image, rendered_image_right, rendered_image_left], axis=1)
            else:
                rendered_image = np.concatenate([rendered_image_right, rendered_image_left], axis=1)

        if render:
            save_folder = save_path.replace('.mp4', '')
            save_folder = save_folder.replace("'", "")
            save_path = save_path.replace("'", "")
            os.makedirs(save_folder, exist_ok=True)
            img_path = os.path.join(save_folder, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(img_path, rendered_image)
            frame_idx += 1
    
    if not render:
        return meshes

    source = f'{save_folder}/frame_%06d.jpg'
    cmd = f'ffmpeg -y -f image2 -r {fps} -i {source} -vcodec libx264 -qscale 0 -pix_fmt yuv420p {save_path}'
    print(cmd)
    os.system(cmd)
    """
    elif body_model == "mano":
        source = f'{save_folder}/frame_%06d.jpg'
        cmd = f'ffmpeg -y -f image2 -r {fps} -i {source} -vcodec libx264 -qscale 0 -pix_fmt yuv420p {save_path}'
    """
    if delete_img:
        shutil.rmtree(save_folder)
    return meshes
