import numpy as np
from smplx.joint_names import JOINT_NAMES


UPPER_BODY_JOINT_NAMES = [
    "pelvis", #0
    "spine1", #1
    "spine2", #2
    "spine3", #3
    "neck", #4
    "left_collar", #5
    "right_collar", #6
    "head", #7
    "left_shoulder", #8 
    "right_shoulder", #9
    "left_elbow", #10
    "right_elbow", #11
    "left_wrist", #12
    "right_wrist", #13
]

RIGHT_HAND_JOINT_NAMES = [
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

LEFT_HAND_JOINT_NAMES = [
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky"
]

UPPER_BODY_JOINT_IDX = np.array([JOINT_NAMES.index(elt) for elt in UPPER_BODY_JOINT_NAMES])
LEFT_HAND_JOINT_IDX = np.array([JOINT_NAMES.index(elt) for elt in LEFT_HAND_JOINT_NAMES])
RIGHT_HAND_JOINT_IDX = np.array([JOINT_NAMES.index(elt) for elt in RIGHT_HAND_JOINT_NAMES])
UPPER_BODY_HANDS_JOINT_IDX = np.concatenate([UPPER_BODY_JOINT_IDX, LEFT_HAND_JOINT_IDX, RIGHT_HAND_JOINT_IDX])

