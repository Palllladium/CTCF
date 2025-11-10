
import torch
import torch.nn.functional as F

def make_coordinate_grid_3d(depth, height, width, device, dtype=torch.float32):
    """
    Create a normalized 3D coordinate grid in [-1,1] for grid_sample.
    Returns tensor of shape [1, D, H, W, 3] with (x,y,z) ordering for grid_sample.
    Note: grid_sample expects grid[..., 0]=x in [-1,1] over W, grid[...,1]=y over H, grid[...,2]=z over D.
    """
    z = torch.linspace(-1.0, 1.0, steps=depth, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=width,  device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # [D,H,W]
    grid = torch.stack((xx, yy, zz), dim=-1)[None, ...]   # [1,D,H,W,3]
    return grid

def vox2grid(flow_vox, size_dhw):
    """
    Convert flow in voxel units [B,3,D,H,W] to normalized units for grid_sample ([-1,1]).
    size_dhw: (D,H,W) of the image.
    The scaling per axis: grid = base + 2*flow/(size-1)
    """
    B, C, D, H, W = flow_vox.shape
    assert C == 3, "flow must be [B,3,D,H,W]"
    sdz = 2.0 / max(D-1, 1)
    sdy = 2.0 / max(H-1, 1)
    sdx = 2.0 / max(W-1, 1)
    # our flow channels are [z, y, x]; grid_sample expects order [x, y, z]
    fx = flow_vox[:, 2] * sdx  # [B,D,H,W]
    fy = flow_vox[:, 1] * sdy
    fz = flow_vox[:, 0] * sdz
    return torch.stack([fx, fy, fz], dim=-1)  # [B, D, H, W, 3]

def warp(image, flow_vox, mode='bilinear', padding_mode='border', align_corners=True):
    """
    Warp a 5D tensor image [B,C,D,H,W] by voxel flow [B,3,D,H,W].
    Returns warped image [B,C,D,H,W].
    """
    B, C, D, H, W = image.shape
    base_grid = make_coordinate_grid_3d(D, H, W, device=image.device, dtype=image.dtype)  # [1,D,H,W,3]
    grid_flow = vox2grid(flow_vox, (D, H, W))  # [B,D,H,W,3]
    grid = base_grid + grid_flow               # broadcast по батчу
    return F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def compose_flows(flow_1, flow_2, mode='bilinear'):
    """
    Compose two flows phi1 (applied first) and phi2 (applied second) in voxel coordinates.
    Both: [B,3,D,H,W]. Returns phi = phi1 + warp(flow_2, phi1).
    """
    # warp flow_2 by flow_1
    B, C, D, H, W = flow_1.shape
    flow_2_warped = warp(flow_2, flow_1, mode=mode)  # [B,3,D,H,W]
    return flow_1 + flow_2_warped

def identity_flow_like(flow):
    """Return zero flow with same shape as input flow [B,3,D,H,W]."""
    return torch.zeros_like(flow)

def gradient_3d(u):
    """
    Central differences gradient for a scalar/each channel in u: [B,C,D,H,W] -> tuple (ux, uy, uz)
    Returns tensors of same shape.
    """
    B, C, D, H, W = u.shape
    def pad(x, dim):
        # pad one voxel on both sides of dim
        pads = [0,0, 0,0, 0,0]
        pads[2*(4-dim)] = 1
        pads[2*(4-dim)+1] = 1
        return F.pad(x, pads, mode='replicate')
    # finite differences
    u_x = (pad(u, 4)[:,:,:,:,2:] - pad(u, 4)[:,:,:,:,:-2]) * 0.5
    u_y = (pad(u, 3)[:,:,:,2:,:] - pad(u, 3)[:,:,:,:-2,:]) * 0.5
    u_z = (pad(u, 2)[:,:,2:,:,:] - pad(u, 2)[:,:,:-2,:,:]) * 0.5
    # crop to original size
    u_x = u_x[:,:,:,:,:W]
    u_y = u_y[:,:,:,:H,:]
    u_z = u_z[:,:,:D,:,:]
    return u_x, u_y, u_z

def jacobian_det(flow):
    """
    Compute Jacobian determinant of deformation x -> x + flow.
    flow: [B,3,D,H,W] in voxel units (components order z,y,x).
    Returns detJ: [B,1,D,H,W].
    """
    # Reorder to components u (x), v (y), w (z) for clarity
    u = flow[:,2:3]  # x-displacement
    v = flow[:,1:2]  # y-displacement
    w = flow[:,0:1]  # z-displacement

    ux, uy, uz = gradient_3d(u)
    vx, vy, vz = gradient_3d(v)
    wx, wy, wz = gradient_3d(w)

    # J = I + grad(u,v,w)
    J11 = 1.0 + ux; J12 = uy;      J13 = uz
    J21 = vx;      J22 = 1.0 + vy; J23 = vz
    J31 = wx;      J32 = wy;      J33 = 1.0 + wz

    detJ = (J11*(J22*J33 - J23*J32)
           -J12*(J21*J33 - J23*J31)
           +J13*(J21*J32 - J22*J31))
    return detJ.unsqueeze(1)

def neg_jacobian_penalty(flow, eps=0.0):
    """
    Penalize negative Jacobian determinants: sum(ReLU(-(detJ - eps))).
    Returns scalar loss.
    """
    detJ = jacobian_det(flow)
    return torch.relu(-(detJ - eps)).mean()

def sdlogj_metric(flow, eps=1e-6):
    """Standard deviation of log|detJ| as a regularity metric (not a loss)."""
    detJ = jacobian_det(flow).clamp_min(1e-6)
    return torch.log(detJ.abs()+eps).std()
