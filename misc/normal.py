import torch

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternions: q = q1 ⊗ q2.
    Supports q1, q2 of shape (..., 4).
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion(s) q of shape (...,4)."""
    sign = q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return q * sign


def rotation_quaternion_from_negx_to_b(vs: torch.Tensor, a: torch.Tensor=None) -> torch.Tensor:
    """
    Compute quaternions that rotate constant vector (-1,0,0) to vs, per-pixel batch [B,3,H,W].
    Optionally apply spin angle a of shape [B,H,W].

    Returns qs of shape [B,H,W,4].
    """
    # vs: [B,3,H,W] normalized
    B, C, H, W = vs.shape
    # reshape to [...,3]
    b = vs.permute(0,2,3,1)  # [B,H,W,3]
    b_x, b_y, b_z = b.unbind(dim=-1)
    # quaternion from a=(-1,0,0) to b: q0 = [w, x=0, y, z]
    # dot = a·b = (-1)*b_x => w = 1 + dot = 1 - b_x
    w = 1.0 - b_x
    y = b_z
    z = -b_y
    x = torch.zeros_like(w)
    q0 = torch.stack((w, x, y, z), dim=-1)  # [B,H,W,4]
    # normalize, handle near-zero
    norm = q0.norm(dim=-1, keepdim=True)
    eps = 1e-6
    mask = (norm < eps).expand_as(q0)
    q0 = q0 / (norm + eps)
    # for opposite case b ≈ (-1,0,0), set a 180° rotation around any orthogonal axis, e.g. [0,0,1]
    q0 = torch.where(mask, torch.tensor([0.0,0.0,0.0,1.0], device=vs.device), q0)
    qs = q0
    if a is not None:
        # spin around b
        # a: [B,H,W]
        wa = torch.cos(a / 2)
        sa = torch.sin(a / 2)
        r = torch.stack((wa, b_x*sa, b_y*sa, b_z*sa), dim=-1)
        # combine: q = r ⊗ q0
        qs = quaternion_multiply(r, q0)
    return qs.permute(0, 3, 1, 2)  # 从 [B, H, W, 4] -> [B, 4, H, W]



def rotate_vectors_by_quaternions(vs: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors vs by quaternions qs. vs [...,3], qs [...,4]."""
    # embed vs into quaternions
    zeros = torch.zeros_like(vs[..., :1])
    v_quat = torch.cat((zeros, vs), dim=-1)
    q_conj = quaternion_conjugate(qs)
    tmp = quaternion_multiply(qs, v_quat)
    rotated = quaternion_multiply(tmp, q_conj)
    return rotated[..., 1:]