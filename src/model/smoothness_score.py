import torch
import torch.nn.functional as F
# --------------------------
# Helpers: axis-angle <-> quaternion
# --------------------------

def quat_mul(q1, q2):
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def quat_inv(q):
    """Quaternion inverse (for unit quaternion)."""
    w, xyz = q[..., 0:1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)

# --------------------------
# Core: log/exp on SO(3)
# --------------------------

def quat_log(q):
    """Quaternion logarithm → rotation vector (axis * angle)."""
    q = F.normalize(q, dim=-1)
    w, xyz = q[..., 0:1], q[..., 1:]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True).repeat(1, 1, 3)
    angle = 2.0 * torch.atan2(sin_half, w)
    small = (sin_half < 1e-8)
    axis = torch.zeros_like(xyz)
    axis[~small] = xyz[~small] / sin_half[~small]
    axis[small] = xyz[small] * 0.0  # arbitrary
    return axis * angle

def quat_exp(v):
    """Rotation vector → quaternion."""
    angle = torch.norm(v, dim=-1, keepdim=True)  # (T, J, 1)
    half = 0.5 * angle  # (T, J, 1)

    w = torch.cos(half)  # (T, J, 1)

    # Avoid division by zero
    safe_angle = torch.where(angle < 1e-8, torch.ones_like(angle), angle)
    xyz = torch.sin(half) / safe_angle * v  # (T, J, 3)

    # For small angles, use first-order approximation: sin(θ/2) ≈ θ/2
    small_mask = (angle < 1e-8).expand_as(v)
    xyz = torch.where(small_mask, 0.5 * v, xyz)

    return torch.cat([w, xyz], dim=-1)  # (T, J, 4)

# --------------------------
# Smoothness loss (3-frame avg)
# --------------------------

def rotation_smoothness_loss_quaternion(quaternions):
    """
    quaternions: (T, J, 4) tensor of quaternions
    returns: scalar loss
    """
    T, J, _ = quaternions.shape
    if T < 3:
        raise ValueError("Need at least 3 frames to compute smoothness loss")

    q = quaternions  # (T, J, 4)
    # fix quaternion sign continuity across time (avoid flips)
    dot = (q[:-1] * q[1:]).sum(-1, keepdim=True)
    flip_mask = dot < 0
    q[1:][flip_mask.expand_as(q[1:])] *= -1.0

    # Compute log vectors (rotation vectors in tangent)
    r = quat_log(q)  # (T, J, 3)

    # Moving average in tangent space (central frame)
    r_smooth = (r[:-2] + r[1:-1] + r[2:]) / 3.0  # (T-2, J, 3)
    q_smooth = quat_exp(r_smooth)                 # (T-2, J, 4)

    # Compute geodesic difference between q_smooth and original q[1:-1]
    delta = quat_mul(quat_inv(q_smooth), q[1:-1])  # relative rotation
    diff_vec = quat_log(delta)  # tangent difference (T-2, J, 3)

    # Smoothness loss: L2 norm of tangent difference
    loss_smooth = torch.mean(torch.sum(diff_vec ** 2, dim=-1))

    # ---- Acceleration loss (based on angular velocity) ----
    # angular velocity between frames
    dt = 1.0
    v = quat_log(quat_mul(quat_inv(q[:-1]), q[1:])) / dt  # (T-1, J, 3)
    # second derivative (angular acceleration)
    acc = (v[1:] - v[:-1]) / dt
    loss_acc = torch.mean(torch.sum(acc ** 2, dim=-1))

    # ---- Combine ----
    w_smooth = 1.0
    w_acc = 10.0
    loss_total = w_smooth * loss_smooth + w_acc * loss_acc

    return {
        'loss_total': loss_total,
        'loss_smooth': loss_smooth,
        'loss_acc': loss_acc
    }