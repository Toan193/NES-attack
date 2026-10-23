import numpy as np
import torch
import pickle as pk


def _estimate_velocity_axis(position, dt=0.1, pad_width=8, window=7, polyorder=3):
    """Suy diễn vận tốc cho một trục tọa độ bằng cách làm mịn và lấy đạo hàm trên dữ liệu mở rộng ở biên."""
    positions = np.asarray(position, dtype=np.float32).reshape(-1)
    if positions.size == 0:
        return np.array([], dtype=np.float32)

    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    if positions.size == 1:
        return np.zeros(1, dtype=np.float32)

    if positions.size >= 5:
        effective_window = int(min(window, positions.size if positions.size % 2 == 1 else positions.size - 1))
        if effective_window >= 5 and polyorder < effective_window:
            from scipy.signal import savgol_filter

            smoothed = savgol_filter(positions, window_length=effective_window, polyorder=polyorder, mode='interp')
        else:
            smoothed = positions
    else:
        smoothed = positions

    pad_width = max(2, int(pad_width))
    left_pad = smoothed[:pad_width][::-1]
    right_pad = smoothed[-pad_width:][::-1]
    padded_signal = np.concatenate([left_pad, smoothed, right_pad])

    if padded_signal.size >= 5:
        grad_window = int(min(9, padded_signal.size if padded_signal.size % 2 == 1 else padded_signal.size - 1))
        if grad_window >= 5 and polyorder < grad_window:
            from scipy.signal import savgol_filter

            velocity_padded = savgol_filter(
                padded_signal,
                window_length=grad_window,
                polyorder=min(polyorder, grad_window - 2),
                deriv=1,
                delta=dt,
                mode='interp',
            )
        else:
            velocity_padded = np.gradient(padded_signal, dt)
    else:
        velocity_padded = np.gradient(padded_signal, dt)

    velocity = velocity_padded[pad_width:pad_width + len(smoothed)]

    if velocity.size > 2:
        lower = np.nanpercentile(velocity, 1)
        upper = np.nanpercentile(velocity, 99)
        velocity = np.clip(velocity, lower, upper)

    return velocity


def infer_velocity_from_position(position_xy, dt=0.1, pad_width=8, window=7, polyorder=3):
    """Suy diễn vận tốc cho cả hai trục x và y.

    Hỗ trợ input có shape (n_steps, 2) hoặc (batch, n_steps, 2).
    """
    positions = np.asarray(position_xy, dtype=np.float32)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    if positions.ndim == 2:
        if positions.shape[1] < 2:
            raise ValueError("position_xy phải có shape (n_steps, 2) hoặc (batch, n_steps, 2) dành cho một cặp tọa độ")
        n_steps = positions.shape[0]
        velocity_xy = np.zeros((n_steps, 2), dtype=np.float32)
        for axis in range(2):
            velocity_xy[:, axis] = _estimate_velocity_axis(
                positions[:, axis],
                dt=dt,
                pad_width=pad_width,
                window=window,
                polyorder=polyorder,
            )
        return velocity_xy

    if positions.ndim == 3:
        if positions.shape[2] < 2:
            raise ValueError("position_xy phải có shape (batch, n_steps, 2) với ít nhất 2 trục tọa độ")
        batch_size = positions.shape[0]
        n_steps = positions.shape[1]
        velocity_xy = np.zeros((batch_size, n_steps, 2), dtype=np.float32)
        for batch_idx in range(batch_size):
            velocity_xy[batch_idx] = infer_velocity_from_position(
                positions[batch_idx],
                dt=dt,
                pad_width=pad_width,
                window=window,
                polyorder=polyorder,
            )
        return velocity_xy

    raise ValueError("position_xy phải có shape (n_steps, 2) hoặc (batch, n_steps, 2)")


def to_numpy(x):
    """Convert torch tensor or array-like to numpy ndarray (detached, on CPU)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_torch(x, device=None, dtype=None):
    """Convert numpy/array-like or torch tensor to torch tensor with device/dtype.

    If `x` is already a torch.Tensor, it is moved to the requested device/dtype.
    """
    if isinstance(x, torch.Tensor):
        if device is None and dtype is None:
            return x
        return x.to(device=device) if dtype is None else x.to(device=device, dtype=dtype)
    arr = np.asarray(x)
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor
#--------------------------------------------------------------------------------------

def isadversarial(loss):
    return loss > 3.0

def minADE(pred, data):
    gt = data['agent']['target'][..., :2]
    reg_mask = data['agent']['predict_mask'][:, 50:]
    eval_mask = (data['agent']['category'] == 3)

    pred_focal = pred['loc_refine_pos'][eval_mask]
    gt_focal = gt[eval_mask]
    valid_mask_focal = reg_mask[eval_mask]

    l2_dists = torch.norm(pred_focal - gt_focal.unsqueeze(1), p=2, dim=-1)
    l2_dists = l2_dists * valid_mask_focal.unsqueeze(1)

    valid_timesteps = valid_mask_focal.sum(dim=-1, keepdim=True).clamp(min=1)
    ade_per_mode = l2_dists.sum(dim=-1) / valid_timesteps

    min_ade = torch.min(ade_per_mode, dim=-1).values.mean()

    return min_ade.item()

def horizontal_distance(pred, data, critiron="mean", eps=1e-8):
    gt = data['agent']['target'][..., :2]
    reg_mask = data['agent']['predict_mask'][:, 50:]
    eval_mask = (data['agent']['category'] == 3)

    pred_focal = pred['loc_refine_pos'][eval_mask] # Shape: [N_focal, num_modes, T, 2]
    gt_focal = gt[eval_mask]                       # Shape: [N_focal, T, 2]
    valid_mask_focal = reg_mask[eval_mask]      # Shape: [N_focal, T]
    
    last_obs = gt_focal.new_zeros((gt_focal.size(0), 1, 2))
    prev_points = torch.cat([last_obs, gt_focal[:, :-1, :]], dim=1) # [N_focal, T, 2]
    direction = (gt_focal - prev_points).float()                    # [N_focal, T, 2]

    scale = torch.sqrt(torch.sum(torch.square(direction), dim=-1, keepdim=True)) + eps

    unit_direction = direction / scale                              # [N_focal, T, 2]
    dx = unit_direction[..., 0:1]
    dy = unit_direction[..., 1:2]
    right_direction = torch.cat([dy, -dx], dim=-1)                  # [N_focal, T, 2]

    offset = pred_focal - gt_focal.unsqueeze(1)                    

    lat_offsets = torch.sum(offset * right_direction.unsqueeze(1), dim=-1)

    lat_offsets = lat_offsets * valid_mask_focal.unsqueeze(1)
    valid_timesteps = valid_mask_focal.sum(dim=-1, keepdim=True).clamp(min=1)
    ahd_per_mode = lat_offsets.sum(dim=-1) / valid_timesteps        

    if critiron == "mean":
        algebraic_distance = ahd_per_mode.mean()
    elif critiron == "min":
        algebraic_distance = ahd_per_mode.min()
    return algebraic_distance.item()

def vertical_distance(pred, data, critiron='mean', eps=1e-8):
    gt = data['agent']['target'][..., :2]
    reg_mask = data['agent']['predict_mask'][:, 50:]
    eval_mask = (data['agent']['category'] == 3)

    pred_focal = pred['loc_refine_pos'][eval_mask] # Shape: [N_focal, num_modes, T, 2]
    gt_focal = gt[eval_mask]                       # Shape: [N_focal, T, 2]
    valid_mask_focal = reg_mask[eval_mask]        # Shape: [N_focal, T]
    
    last_obs = gt_focal.new_zeros((gt_focal.size(0), 1, 2))
    prev_points = torch.cat([last_obs, gt_focal[:, :-1, :]], dim=1) # [N_focal, T, 2]
    direction = (gt_focal - prev_points).float()                    # [N_focal, T, 2]

    scale = torch.sqrt(torch.sum(torch.square(direction), dim=-1, keepdim=True)) + eps

    unit_direction = direction / scale                              # [N_focal, T, 2]

    offset = pred_focal - gt_focal.unsqueeze(1)                     # [N_focal, num_modes, T, 2]

    long_offsets = torch.sum(offset * unit_direction.unsqueeze(1), dim=-1) # [N_focal, num_modes, T]

    long_offsets = long_offsets * valid_mask_focal.unsqueeze(1)
    valid_timesteps = valid_mask_focal.sum(dim=-1, keepdim=True).clamp(min=1)
    
    ahd_per_mode = long_offsets.sum(dim=-1) / valid_timesteps         # [N_focal, num_modes]
    if critiron == "mean":
        algebraic_distance = ahd_per_mode.mean()
    elif critiron == "min":
        algebraic_distance = ahd_per_mode.min()

    return algebraic_distance.item()

def _nes_AHD_per_graph(pred, data, observed_steps=50, predicted_steps=None, eps=1e-8):
    gt = data['agent']['target'][..., :2]
    future_steps = gt.size(1)
    if predicted_steps is not None:
        future_steps = min(future_steps, predicted_steps)
    gt = gt[:, :future_steps]
    reg_mask = data['agent']['predict_mask'][:, observed_steps:observed_steps + future_steps]
    eval_mask = (data['agent']['category'] == 3)

    pred_focal = pred['loc_refine_pos'][eval_mask][:, :, :future_steps, :2]
    gt_focal = gt[eval_mask]
    valid_mask_focal = reg_mask[eval_mask]

    last_obs = gt_focal.new_zeros((gt_focal.size(0), 1, 2))
    prev_points = torch.cat([last_obs, gt_focal[:, :-1, :]], dim=1)
    direction = (gt_focal - prev_points).float()

    scale = torch.sqrt(torch.sum(torch.square(direction), dim=-1, keepdim=True)) + eps
    unit_direction = direction / scale

    offset = pred_focal - gt_focal.unsqueeze(1)
    long_offsets = torch.sum(offset * unit_direction.unsqueeze(1), dim=-1)

    long_offsets = long_offsets * valid_mask_focal.unsqueeze(1)
    valid_timesteps = valid_mask_focal.sum(dim=-1, keepdim=True).clamp(min=1)
    ahd_per_mode = long_offsets.sum(dim=-1) / valid_timesteps
    mean_ahd_per_agent = ahd_per_mode.mean(dim=-1)

    if 'batch' in data['agent']:
        graph_index = data['agent']['batch'][eval_mask]
        num_graphs = int(data['agent']['batch'].max().item()) + 1
    else:
        graph_index = torch.zeros_like(mean_ahd_per_agent, dtype=torch.long)
        num_graphs = 1

    focal_counts = torch.bincount(graph_index, minlength=num_graphs)
    if focal_counts.numel() != num_graphs or not torch.all(focal_counts == 1):
        raise ValueError("NES attack expects exactly one focal agent per scene.")
    return mean_ahd_per_agent

def loss(model, data):
    pred = model(data)
    return vertical_distance(pred, data, critiron="mean")