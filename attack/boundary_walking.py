import numpy as np
import torch
from .utils import infer_velocity_from_position

def get_diff(sample_1, sample_2):
    """Per-point norm of difference between samples."""
    return np.linalg.norm(sample_1 - sample_2, axis=-1)


def forward_perturbation(epsilon, prev_sample, target_sample):
    """Generate forward perturbation."""
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb *= epsilon
    return perturb


def orthogonal_perturbation(delta, prev_sample, target_sample, obs_length, pre_length, bound):
    """Generate an orthogonal perturbation for trajectory tensors."""
    prev_sample = np.asarray(prev_sample, dtype=np.float32)
    target_sample = np.asarray(target_sample, dtype=np.float32)

    perturb = np.random.randn(*prev_sample.shape).astype(np.float32)
    perturb_norms = np.linalg.norm(perturb, axis=-1, keepdims=True)
    perturb = perturb / np.where(perturb_norms > 0, perturb_norms, 1.0)

    diff = (target_sample - prev_sample).astype(np.float32)
    diff_norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    scale = delta * np.mean(diff_norms[diff_norms > 0]) if np.any(diff_norms > 0) else delta
    perturb *= scale

    diff_unit = diff / np.where(diff_norms > 0, diff_norms, 1.0)
    proj = np.sum(perturb * diff_unit, axis=-1, keepdims=True)
    perturb = perturb - proj * diff_unit

    perturb = np.clip(perturb, -bound, bound)
    return perturb
def _dtp_orthogonal_perturbation(delta, prev_sample, target_sample, obs_length):
    prev_sample = np.asarray(prev_sample, dtype=np.float32)
    target_sample = np.asarray(target_sample, dtype=np.float32)
    perturb = np.random.randn(*prev_sample.shape).astype(np.float32)

    perturb_norms = np.linalg.norm(perturb, axis=-1, keepdims=True)
    perturb /= np.where(perturb_norms > 0, perturb_norms, 1.0)
    target_observations = target_sample[..., :obs_length, :]
    prev_observations = prev_sample[..., :obs_length, :]
    mean_distance = np.mean(get_diff(target_observations, prev_observations))
    perturb *= delta * mean_distance

    diff = target_observations - prev_observations
    diff_norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    diff_unit = diff / np.where(diff_norms > 0, diff_norms, 1.0)
    projection = np.sum(perturb * diff) / max(float(np.sum(np.square(diff))), 1e-12)
    perturb -= projection * diff
    return np.clip(perturb, -2.0, 2.0)

def infer_velocity(noise_positions, clean_positions):
    noise = noise_positions - clean_positions
    velocity = infer_velocity_from_position(noise)
    return velocity

def _get_focal_mask(data):
    focal_mask = data['agent']['category'] == 3
    if not focal_mask.any():
        raise ValueError("Attack requires at least one focal agent (category == 3).")
    return focal_mask


def _get_observation_positions(data, obs_length):
    focal_mask = _get_focal_mask(data)
    return data['agent']['position'][focal_mask, :obs_length, :2]


def _set_observation_positions(data, positions, clean_positions, obs_length):
    device = data['agent']['position'].device
    target_dtype = data['agent']['position'].dtype

    # ensure positions is a torch tensor on correct device/dtype
    if isinstance(positions, torch.Tensor):
        pos_t = positions.to(device=device, dtype=target_dtype)
    else:
        pos_t = torch.from_numpy(np.asarray(positions)).to(device=device, dtype=target_dtype)

    if pos_t.dim() == 2:
        pos_t = pos_t.unsqueeze(0)

    focal_mask = _get_focal_mask(data)
    if pos_t.size(0) != int(focal_mask.sum().item()):
        raise ValueError("positions must contain one trajectory per focal agent.")

    data['agent']['position'][focal_mask, :obs_length, :2] = pos_t[:, :obs_length, :2]


def _evaluate_loss(model, data):
    from attack.utils import loss

    with torch.no_grad():
        return loss(model, data)


def _is_adversarial(loss_value):
    from attack.utils import isadversarial
    return isadversarial(loss_value)


def _build_candidate_positions(current_positions, clean_positions, epsilon, delta, obs_length, pred_length, bound, count):
    current_np = current_positions.detach().cpu().numpy()
    clean_np = clean_positions.detach().cpu().numpy()
    candidates = []

    directions = []
    directions.append((clean_np - current_np) * epsilon)
    directions.append((current_np - clean_np) * epsilon)

    for _ in range(max(0, count // 2)):
        directions.append(orthogonal_perturbation(delta, current_np, clean_np, obs_length, pred_length, bound))

    for _ in range(max(0, count // 2)):
        directions.append(np.random.randn(*current_np.shape).astype(np.float32) * (epsilon * 0.5))

    for direction in directions[:count]:
        candidate = current_np + direction
        candidate = np.clip(candidate, clean_np - bound, clean_np + bound)
        candidates.append(torch.from_numpy(candidate.astype(np.float32)))

    if len(candidates) < count:
        for _ in range(count - len(candidates)):
            candidates.append(torch.clone(current_positions))

    return candidates


def _find_initial_adversarial_positions(model, data, obs_length, noise_scale=0.2):
    clean_positions = _get_observation_positions(data, obs_length).detach().clone()
    current_positions = clean_positions.clone()
    #candidate_positions = current_positions.clone()
    #loss_value = _evaluate_loss(model, data)
    iter = 0
    while True:
        iter += 1
        noise = torch.randn_like(current_positions) * noise_scale
        candidate_positions = clean_positions + noise
        # backup slices
        pos_backup = data['agent']['position'][:, :obs_length, :2].detach().clone()
        vel_backup = data['agent']['velocity'][:, :obs_length, :2].detach().clone()

        _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
        loss_value = _evaluate_loss(model, data)
        if _is_adversarial(loss_value) or iter > 100:
            break

        # restore original slices after trial
        data['agent']['position'][:, :obs_length, :2] = pos_backup
        data['agent']['velocity'][:, :obs_length, :2] = vel_backup
    if not _is_adversarial(loss_value):
        print(f"\n số lần lặp lấy mẫu: {iter}, loss: {loss_value}")
    return candidate_positions.detach().clone(), loss_value

def _find_initial_adv_positions_ver1(
    model,
    data,
    obs_length=50,
    noise_bound=10.0,
    objective="vertical",
    direction="positive",
    max_attempts=500,
    evaluate_loss=None,
):
    """Sample uniform focal-position noise until an adversarial candidate is found."""
    from .utils import horizontal_distance, vertical_distance

    if obs_length <= 0:
        raise ValueError("obs_length must be greater than 0.")
    if noise_bound <= 0:
        raise ValueError("noise_bound must be greater than 0.")
    if objective not in {"vertical", "horizontal"}:
        raise ValueError("objective must be 'vertical' or 'horizontal'.")
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'.")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be greater than 0 when provided.")
    max_attempts = min(max_attempts or 1000, 1000)

    clean_positions = _get_observation_positions(data, obs_length).detach().clone()
    attempts = 0
    candidate_positions = clean_positions
    loss_value = None
    while attempts < max_attempts:
        attempts += 1
        noise = torch.empty_like(clean_positions).uniform_(-noise_bound, noise_bound)
        candidate_positions = clean_positions + noise
        _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
        try:
            with torch.no_grad():
                prediction = model(data)
                if objective == "vertical":
                    loss_value = vertical_distance(prediction, data)
                else:
                    loss_value = horizontal_distance(prediction, data)
            if direction == "negative":
                loss_value = -loss_value
        except Exception:
            _set_observation_positions(data, clean_positions, clean_positions, obs_length)
            raise

        if _is_adversarial(loss_value):
            return candidate_positions.detach().clone(), loss_value, attempts
        _set_observation_positions(data, clean_positions, clean_positions, obs_length)

    # Return the last tried sample and its score when the attempt limit is reached.
    _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
    return candidate_positions.detach().clone(), loss_value, attempts

def _find_initial_adversarial_positions_ver2(
    model,
    data,
    obs_length=50,
    noise_scale=0.2,
    max_attempts=20,
    evaluate_loss=None,
):
    clean_positions = _get_observation_positions(data, obs_length).detach().clone()

    for attempt in range(max_attempts):
        current_noise_scale = noise_scale * (attempt + 1)
        noise = torch.randn_like(clean_positions) * current_noise_scale
        candidate_positions = clean_positions + noise
        # backup slices
        pos_backup = data['agent']['position'][:, :obs_length, :2].detach().clone()
        vel_backup = data['agent']['velocity'][:, :obs_length, :2].detach().clone()

        _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
        loss_value = (
            evaluate_loss()
            if evaluate_loss is not None
            else _evaluate_loss(model, data)
        )
        if _is_adversarial(loss_value):
            return candidate_positions.detach().clone(), loss_value

        # restore original slices after trial
        data['agent']['position'][:, :obs_length, :2] = pos_backup
        data['agent']['velocity'][:, :obs_length, :2] = vel_backup

    # If no adversarial example found, return the last candidate and its loss
    _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
    return candidate_positions.detach().clone(), loss_value

def boundary_attack(model, data, obs_length=50, pred_length=60, max_iters=20, steps=10, device=None, noise_scale=0.2, bound=1.0):
    """Apply a boundary-walking attack that alternates between orthogonal and forward steps.

    The current Argoverse 2 data structure stores trajectories under
    data['agent']['position']; this function perturbs that tensor in place and
    evaluates the attack using the loss implementation from attack.py.
    """
    if device is None:
        device = next(model.parameters()).device

    data = data.to(device)
    clean_positions = _get_observation_positions(data, obs_length).detach().clone()

    current_positions, current_loss = _find_initial_adversarial_positions(
        model,
        data,
        obs_length,
        device,
        noise_scale=noise_scale,
    )

    best_positions = current_positions.detach().clone()
    best_loss = current_loss
    best_distance = float(np.mean(get_diff(current_positions.detach().cpu().numpy(), clean_positions.detach().cpu().numpy())))

    delta = 1.0
    epsilon = 0.1
    tolerance = 1e-6
    prev_distance = best_distance
    ran_steps = 0

    for _ in range(max_iters):
        current_candidate = current_positions.detach().clone()
        current_candidate_loss = current_loss

        orthogonal_accepted = False
        orthogonal_scale = delta
        best_orthogonal_candidate = current_candidate.detach().clone()
        best_orthogonal_loss = current_candidate_loss

        for _ in range(max(20, steps)):
            orthogonal_perturb = orthogonal_perturbation(
                orthogonal_scale,
                current_candidate.detach().cpu().numpy(),
                clean_positions.detach().cpu().numpy(),
                obs_length,
                pred_length,
                bound,
            )
            orthogonal_candidate = current_candidate + torch.from_numpy(orthogonal_perturb).to(device, dtype=current_candidate.dtype)
            orthogonal_candidate = torch.clamp(orthogonal_candidate, clean_positions - bound, clean_positions + bound)

            # backup original slices
            pos_backup = data['agent']['position'][:, :obs_length, :2].detach().clone()
            vel_backup = data['agent']['velocity'][:, :obs_length, :2].detach().clone()

            _set_observation_positions(data, orthogonal_candidate, clean_positions, obs_length)
            trial_loss = _evaluate_loss(model, data)

            if trial_loss > best_orthogonal_loss:
                best_orthogonal_loss = trial_loss
                best_orthogonal_candidate = orthogonal_candidate.detach().clone()

            if _is_adversarial(trial_loss):
                current_candidate = orthogonal_candidate.detach().clone()
                current_candidate_loss = trial_loss
                orthogonal_accepted = True
                # keep the assignment (do not restore)
                break

            # restore original slices after unsuccessful trial
            data['agent']['position'][:, :obs_length, :2] = pos_backup
            data['agent']['velocity'][:, :obs_length, :2] = vel_backup

            orthogonal_scale *= 0.95
            if orthogonal_scale < tolerance:
                break

        if orthogonal_accepted:
            current_candidate = current_candidate.detach().clone()
            current_candidate_loss = current_candidate_loss
        else:
            current_candidate = best_orthogonal_candidate.detach().clone()
            current_candidate_loss = best_orthogonal_loss

        forward_accepted = False
        forward_scale = epsilon
        forward_candidate = current_candidate.detach().clone()
        best_forward_candidate = forward_candidate.detach().clone()
        best_forward_loss = current_candidate_loss

        for _ in range(max(20, steps)):
            diff = clean_positions.to(device) - forward_candidate.to(device)
            diff_norms = torch.linalg.norm(diff, dim=-1, keepdim=True)
            diff_unit = diff / torch.where(diff_norms > 0, diff_norms, torch.ones_like(diff_norms))
            candidate_positions = forward_candidate + diff_unit * forward_scale
            candidate_positions = torch.clamp(candidate_positions, clean_positions - bound, clean_positions + bound)

            # backup original slices
            pos_backup = data['agent']['position'][:, :obs_length, :2].detach().clone()
            vel_backup = data['agent']['velocity'][:, :obs_length, :2].detach().clone()

            _set_observation_positions(data, candidate_positions, clean_positions, obs_length)
            trial_loss = _evaluate_loss(model, data)

            if trial_loss > best_forward_loss:
                best_forward_loss = trial_loss
                best_forward_candidate = candidate_positions.detach().clone()

            if _is_adversarial(trial_loss):
                forward_candidate = candidate_positions.detach().clone()
                best_forward_candidate = forward_candidate.detach().clone()
                best_forward_loss = trial_loss
                forward_accepted = True
                # keep the assignment
                break

            # restore original slices after unsuccessful trial
            data['agent']['position'][:, :obs_length, :2] = pos_backup
            data['agent']['velocity'][:, :obs_length, :2] = vel_backup

            forward_scale *= 0.9
            if forward_scale < tolerance:
                break

        if forward_accepted:
            current_positions = forward_candidate.detach().clone()
            current_loss = best_forward_loss
        else:
            current_positions = best_forward_candidate.detach().clone()
            current_loss = best_forward_loss

        current_distance = float(np.mean(get_diff(current_positions.detach().cpu().numpy(), clean_positions.detach().cpu().numpy())))

        if current_loss > best_loss or (abs(current_loss - best_loss) < 1e-8 and current_distance < best_distance):
            best_positions = current_positions.detach().clone()
            best_loss = current_loss
            best_distance = current_distance

        if abs(prev_distance - current_distance) < 1e-8:
            break

        prev_distance = current_distance
        delta = max(delta * 0.95, tolerance)
        epsilon = max(epsilon * 0.95, tolerance)

    _set_observation_positions(data, best_positions, clean_positions, obs_length)
    return best_loss

def original_boundary_attack(
    model,
    data,
    obs_length=50,
    pred_length=60,
    device=None,
    objective="vertical",
    direction="positive",
    init_noise_bound=0.2,
    epsilon=1.0,
    delta=0.1,
    orthogonal_samples=10,
    max_steps=100,
    min_distance=1e-3,
    max_init_attempts=40,
    max_orthogonal_batches=500,
    max_forward_steps=500,
):
    """Run a DTP-style attack; horizontal positive is right, vertical positive is forward.

    Returns the best perturbation distance, directional score, focal perturbation,
    and number of model evaluations. The clean observations are restored on return.
    """
    if device is None:
        device = next(model.parameters()).device

    if objective not in {"vertical", "horizontal"}:
        raise ValueError("objective must be 'vertical' or 'horizontal'.")
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'.")
    if init_noise_bound <= 0:
        raise ValueError("init_noise_bound must be greater than 0.")
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be greater than 0.")
    if orthogonal_samples <= 0 or max_steps < 0:
        raise ValueError("orthogonal_samples must be positive and max_steps non-negative.")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if min(max_init_attempts, max_orthogonal_batches, max_forward_steps) <= 0:
        raise ValueError("All attack attempt limits must be greater than 0.")
    data = data.to(device)
    clean_positions = _get_observation_positions(data, obs_length).detach().clone()
    if clean_positions.size(0) == 0:
        raise ValueError("DTP-style boundary attack requires at least one focal agent.")

    position = data['agent']['position']
    original_observations = position[:, :obs_length, :2].detach().clone()
    num_evaluations = 0

    def evaluate_directional_loss():
        nonlocal num_evaluations
        from .utils import horizontal_distance, vertical_distance

        with torch.no_grad():
            pred = model(data)
            num_evaluations += 1
            if objective == "vertical":
                score = vertical_distance(pred, data)
            else:
                score = horizontal_distance(pred, data)
        return score if direction == "positive" else -score

    def assign_focal_positions(candidate_positions):
        _set_observation_positions(data, candidate_positions, clean_positions, obs_length)

    target_positions = clean_positions.detach().cpu().numpy()
    initial_positions, current_loss,_ = _find_initial_adv_positions_ver1(
        model,
        data,
        max_attempts=200
    )
    current_positions = initial_positions.detach().cpu().numpy()

    adversarial_sample = current_positions.copy()
    if not _is_adversarial(current_loss):
        assign_focal_positions(clean_positions)
        return (
            initial_positions.to(device=device, dtype=clean_positions.dtype),
            current_loss,
            0.0,
            0,
        )

    current_distance = float(np.mean(get_diff(adversarial_sample, target_positions)))
    best_distance = current_distance
    global_adversarial_sample = adversarial_sample.copy()
    global_loss = current_loss
    n_steps = 0

    while n_steps <= max_steps:
        for _ in range(max_orthogonal_batches):
            trial_samples = []
            adversarial_flags = []
            trial_losses = []
            for _ in range(orthogonal_samples):
                orthogonal_noise = _dtp_orthogonal_perturbation(
                    delta,
                    adversarial_sample,
                    target_positions,
                    obs_length,
                )
                trial_sample = adversarial_sample + orthogonal_noise
                assign_focal_positions(torch.from_numpy(trial_sample).to(device=device, dtype=clean_positions.dtype))
                trial_loss = evaluate_directional_loss()
                trial_samples.append(trial_sample)
                trial_losses.append(trial_loss)
                adversarial_flags.append(_is_adversarial(trial_loss))

            d_score = float(np.mean(adversarial_flags))
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.95
                elif d_score > 0.7:
                    delta /= 0.95
                successful_indices = np.flatnonzero(adversarial_flags)
                selected_index = int(successful_indices[0])
                adversarial_sample = trial_samples[selected_index]
                current_loss = trial_losses[selected_index]
                break
            delta *= 0.95
        else:
            assign_focal_positions(
                torch.from_numpy(adversarial_sample).to(
                    device=device,
                    dtype=clean_positions.dtype,
                )
            )
            raise RuntimeError(
                "Could not find an adversarial orthogonal step after "
                f"{max_orthogonal_batches} batches."
            )

        e_step = 0
        while True:
            e_step += 1
            trial_sample = adversarial_sample + forward_perturbation(
                epsilon,
                adversarial_sample,
                target_positions,
            )
            assign_focal_positions(torch.from_numpy(trial_sample).to(device=device, dtype=clean_positions.dtype))
            trial_loss = evaluate_directional_loss()
            if _is_adversarial(trial_loss):
                adversarial_sample = trial_sample
                current_loss = trial_loss
                epsilon /= 0.5
                break
            if e_step > max_forward_steps:
                break
            epsilon *= 0.5

        current_distance = float(np.mean(get_diff(adversarial_sample, target_positions)))
        if current_distance < best_distance:
            best_distance = current_distance
            global_adversarial_sample = adversarial_sample.copy()
            global_loss = current_loss

        n_steps += 1
        if (
            current_distance <= min_distance
            or e_step > max_forward_steps
            or current_distance > 30
            or n_steps > max_steps
        ):
            break

    perturbation = global_adversarial_sample - target_positions
    assign_focal_positions(clean_positions)
    position[:, :obs_length, :2] = original_observations
    return (
        torch.from_numpy(perturbation).to(device=device, dtype=clean_positions.dtype),
        global_loss,
        best_distance,
        num_evaluations
    )
