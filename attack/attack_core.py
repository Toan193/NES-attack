import numpy as np
import pickle
from pathlib import Path
import os
import time

import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Batch, HeteroData
from attack.boundary_walking import original_boundary_attack
from attack.boundary_walking import boundary_attack
from attack.nes_attack import nes_trajectory_attack
from .utils import vertical_distance, horizontal_distance
from .utils import isadversarial

def attack_adv(model, data):
    return original_boundary_attack(model, data)


def compute_avg_noise_norm(batch, clean_positions, attacked_positions):
    noise = attacked_positions - clean_positions.to(attacked_positions.device)
    eval_mask = batch['agent']['category'] == 3

    if not eval_mask.any():
        return 0.0

    selected_noise = noise[eval_mask]
    return torch.linalg.norm(selected_noise, dim=-1).mean().item()

def run_prediction_and_loss(model, data, objective="vertical", direction="positive"):
    with torch.no_grad():
        pred = model(data)

    eval_idx = (data['agent']['category'] == 3).nonzero(as_tuple=True)[0]
    focal_predictions = pred['loc_refine_pos'][eval_idx]
    focal_confidences = torch.softmax(pred['pi'][eval_idx], dim=-1)

    if objective == "vertical":
        loss = vertical_distance(pred=pred, data=data, critiron="mean")
    elif objective == "horizon":
        loss = horizontal_distance(pred=pred, data=data, critiron="mean")
    else:
        raise ValueError("objective must be vertical or horizon")

    if direction == "negative":
        loss = -loss
    elif direction != "positive":
        raise ValueError("direction must be positive or negative")

    return (
        focal_predictions.detach().cpu(),
        focal_confidences.detach().cpu(),
        loss,
    )


def calculate_per_prediction_ade_fde(predictions, ground_truth, valid_mask):
    """Calculate ADE and FDE for each predicted mode of one focal agent."""
    if predictions.ndim != 4 or predictions.size(0) != 1:
        raise ValueError(
            "Expected predictions with shape [1, num_modes, timesteps, 2]."
        )
    if ground_truth.ndim != 3 or ground_truth.size(0) != 1:
        raise ValueError("Expected ground_truth with shape [1, timesteps, 2].")
    if valid_mask.ndim != 2 or valid_mask.size(0) != 1:
        raise ValueError("Expected valid_mask with shape [1, timesteps].")

    target = ground_truth[0].to(device=predictions.device, dtype=predictions.dtype)
    mask = valid_mask[0].to(device=predictions.device, dtype=torch.bool)
    if target.size(0) != predictions.size(2) or mask.size(0) != predictions.size(2):
        raise ValueError("Prediction, target, and valid-mask horizons must match.")
    if not mask.any():
        raise ValueError("Cannot calculate ADE/FDE without valid future timesteps.")

    distances = torch.linalg.vector_norm(predictions[0] - target.unsqueeze(0), dim=-1)
    ade = distances[:, mask].mean(dim=-1)
    final_valid_index = torch.where(mask)[0][-1]
    fde = distances[:, final_valid_index]
    return ade.detach().cpu().numpy(), fde.detach().cpu().numpy()


def run_attack(model, data, file_path):
    device = model.device
    model.eval()

    processed_scenes = set()
    if os.path.isfile(file_path):
        dfs = []
        with open(file_path, "rb") as f:
            while True:
                try:
                    dfs.append(pickle.load(f))
                except EOFError:
                    break
    
        df = pd.concat(dfs, ignore_index=True)
        processed_scenes = set(df['scene_id'])

    for batch_idx, batch in enumerate(tqdm(data, desc="Attacked Scenes")):
        scene_id = None
        failed = None
        try:
            batch = batch.to(device)

            if isinstance(batch, Batch):
                batch['agent']['av_index'] += batch['agent']['ptr'][:-1]

            scene_id = batch['scenario_id']

            if scene_id in processed_scenes:
                continue

            with torch.no_grad():
                clean_predictions, _, clean_loss = run_prediction_and_loss(model, batch)

            eval_idx = (batch['agent']['category'] == 3).nonzero(as_tuple=True)[0]
            clean_position = batch['agent']['position'][eval_idx, :50, :2].detach().clone()
            ground_truth = batch['agent']['target'][eval_idx, :, :2].detach().clone()
            valid_future_mask = batch['agent']['predict_mask'][eval_idx, 50:]
            ade_before, fde_before = calculate_per_prediction_ade_fde(
                        clean_predictions,
                        ground_truth,
                        valid_future_mask,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attack_started_at = time.perf_counter()
            noise, adv_loss, avg_noise, num_eval = attack_adv(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attack_time_ms = (time.perf_counter() - attack_started_at) * 1000.0

            disturbation = noise.to(device=device, dtype=clean_position.dtype)
            batch['agent']['position'][eval_idx, :50, :2] = clean_position + disturbation
            predictions, confidences, _ = run_prediction_and_loss(model, batch)
            ade_after, fde_after = calculate_per_prediction_ade_fde(
                predictions,
                ground_truth,
                valid_future_mask,
            )
        except torch.cuda.OutOfMemoryError as e:
            failed = str(e)[:200]
        if failed is not None:                   # xử lý NGOÀI khối except
            with open("output/failed", "a") as f:
                f.write(f"{scene_id}\t{failed}\n")
            batch = None
            torch.cuda.empty_cache()
            continue

        is_success = isadversarial(adv_loss)

        df_results = pd.DataFrame([
            {
                "scene_id": scene_id,
                "clean_positions": clean_position.cpu().numpy(),
                "noise": noise.cpu().numpy(),
                "ground_truth": ground_truth.cpu().numpy(),
                "prediction": predictions.numpy(),
                "confidence": confidences.numpy(),
                "ade_before_noise": ade_before,
                "fde_before_noise": fde_before,
                "ade_after_noise": ade_after,
                "fde_after_noise": fde_after,
                "clean_loss": clean_loss,
                "adv_loss": adv_loss,
                "degradation": adv_loss - clean_loss,
                "avg_noise_norm": avg_noise,
                "num_eval": num_eval,
                "attack_time_ms": attack_time_ms,
                "is_attack_success": is_success
            }
        ])

        try:
            with open(file_path, 'ab') as f:
                pickle.dump(df_results, f)
        except Exception as e:
            print(f"Không thể lưu kết quả: {e}")

        if device.type == "cuda":
            del batch
            del clean_position, ground_truth, valid_future_mask
            del disturbation
            torch.cuda.empty_cache()

    dfs = []
    with open(file_path, "rb") as f:
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break

    data_results = pd.concat(dfs, ignore_index=True)
    data_results.to_pickle(file_path)

    avg_clean = data_results["clean_loss"].mean()
    avg_adv = data_results["adv_loss"].mean()
    avg_noise_norm = data_results["avg_noise_norm"].mean()
    number_eval = data_results["num_eval"].mean()
    success_rate = data_results["is_attack_success"].mean() * 100

    print("\n" + "=" * 50)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ/TẤN CÔNG TOÀN BỘ VAL DATASET")
    print("=" * 50)

    print(f"score trung bình (Gốc) : {avg_clean:.4f} mét")
    print(f"score trung bình (DTP)  : {avg_adv:.4f} mét")
    print(f"Độ lớn nhiễu trung bình (cuối cùng) : {avg_noise_norm:.4f}")
    print(f"số lần chạy eval trung bình : {number_eval:.2f}")
    print(f"Tỷ lệ tấn công thành công : {success_rate:.2f}%")
    print("=" * 50)
