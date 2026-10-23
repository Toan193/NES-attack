import torch
from torch_geometric.data import Batch
from attack.utils import _nes_AHD_per_graph

def _prepare_sample_batch_context(data):
    base_data_list = data.to_data_list() if isinstance(data, Batch) else [data]
    if len(base_data_list) != 1:
        raise ValueError("NES attack expects exactly one scene in the input batch.")

    base_data = base_data_list[0]
    focal_indices = (base_data['agent']['category'] == 3).nonzero(as_tuple=True)[0]
    if focal_indices.numel() != 1:
        raise ValueError("NES attack expects exactly one focal agent (category == 3).")
    return base_data, int(focal_indices.item())


def _build_sample_batch(base_data, focal_index, samples, observed_steps):
    sample_data_list = []
    for sample in samples:
        sample_data = base_data.clone()
        target = sample_data['agent']['position'][focal_index, :observed_steps, :2]
        sample_data['agent']['position'][focal_index, :observed_steps, :2] = sample.to(
            device=target.device,
            dtype=target.dtype,
        )
        sample_data_list.append(sample_data)

    return Batch.from_data_list(sample_data_list).to(samples.device)


def _evaluate_ahd_batch(
    model,
    base_data,
    focal_index,
    samples,
    observed_steps,
    predicted_steps,
    eval_batch_size=1,
):
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be greater than 0.")

    loss_chunks = []
    with torch.no_grad():
        for start in range(0, samples.size(0), eval_batch_size):
            sample_batch = _build_sample_batch(
                base_data,
                focal_index,
                samples[start:start + eval_batch_size],
                observed_steps,
            )
            pred = model(sample_batch)
            graph_losses = _nes_AHD_per_graph(
                pred,
                sample_batch,
                observed_steps=observed_steps,
                predicted_steps=predicted_steps,
            )
            loss_chunks.append(graph_losses.detach().cpu())
            del graph_losses, pred, sample_batch

    return torch.cat(loss_chunks)


def nes_trajectory_attack(model, data,
                          max_iters=100, n_samples=30, sigma=0.01, eta=0.01,
                          max_epsilon=0.2, obsersed_steps=50, predicted_step=60,
                          seed=42, eval_batch_size=1):
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0 for NES gradient estimation.")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be greater than 0.")

    base_data, focal_index = _prepare_sample_batch_context(data)
    past_trajectory = data['agent']['position'][focal_index, :obsersed_steps, :2].detach().clone()
    device = past_trajectory.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    theta = past_trajectory.clone().detach()

    for step in range(max_iters):

        noise = torch.randn(
            n_samples,
            *past_trajectory.shape,
            device=device,
            generator=generator,
        )
        
        pos_samples = theta.unsqueeze(0) + sigma * noise
        neg_samples = theta.unsqueeze(0) - sigma * noise
        
        all_samples = torch.cat([pos_samples, neg_samples], dim=0)
        
        losses = _evaluate_ahd_batch(
            model,
            base_data,
            focal_index,
            all_samples,
            obsersed_steps,
            predicted_step,
            eval_batch_size=eval_batch_size,
        ).to(device=device, dtype=theta.dtype)
            
        loss_pos = losses[:n_samples]
        loss_neg = losses[n_samples:]
        
        loss_diff = (loss_pos - loss_neg).view(n_samples, *([1] * theta.dim()))
        
        grad_hat = torch.mean(loss_diff * noise, dim=0) / (2 * sigma)
        
        theta = theta + eta * grad_hat
        
        delta = torch.clamp(theta - past_trajectory, min=-max_epsilon, max=max_epsilon)
        theta = past_trajectory + delta
        
        """if (step + 1) % 20 == 0:
            sample_ahd = losses.mean().item()
            print(f"Bước {step+1:03d}/{max_iters} | AHD samples TB: {sample_ahd:.4f}")"""

    final_noise = theta - past_trajectory
    final_ahd_score = _evaluate_ahd_batch(
        model,
        base_data,
        focal_index,
        theta.unsqueeze(0),
        obsersed_steps,
        predicted_step,
        eval_batch_size=eval_batch_size,
    )[0].item()
    data['agent']['position'][focal_index, :obsersed_steps, :2] = theta.to(
        device=data['agent']['position'].device,
        dtype=data['agent']['position'].dtype,
    )
    avg_noise = torch.linalg.vector_norm(final_noise, dim=-1).mean().item()
    
    return final_noise.cpu(), float(final_ahd_score), avg_noise, 2*n_samples*max_iters
