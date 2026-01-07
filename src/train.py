import os
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import AttentionModel
from tsp_env import TSPEnv  # (unused here but keeping as you had it)

import wandb

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 64
TRAIN_BATCHES = 100
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
NUM_NODES = 50
VALIDATION_PATH = "../tsp_instances"
SOLUTION_PATH = "../tsp_solutions"

# W&B settings
WANDB_PROJECT = "tsp-attention-rl"
WANDB_RUN_NAME = None  # e.g. "tsp50-baseline-v1"


def load_validation_data():
    """Loads the fixed validation set (the 500 instances you created)."""
    val_nodes = []
    val_solutions = []

    files = sorted([f for f in os.listdir(VALIDATION_PATH) if f.endswith(".txt")])[:50]
    print(f"Loading {len(files)} validation instances...")

    for f in files:
        nodes = []
        with open(os.path.join(VALIDATION_PATH, f), "r") as file:
            c_section = False
            for line in file:
                if "NODE_COORD_SECTION" in line:
                    c_section = True
                    continue
                if c_section and "EOF" not in line:
                    _, x, y = line.split()
                    nodes.append([float(x), float(y)])
        val_nodes.append(nodes)

        sol_name = f.replace(".txt", "_sol.txt")
        cost = 0
        sol_path = os.path.join(SOLUTION_PATH, sol_name)
        if os.path.exists(sol_path):
            with open(sol_path, "r") as file:
                for line in file:
                    if "COST" in line:
                        cost = float(line.split(":")[-1].strip())
        val_solutions.append(cost)

    return torch.tensor(val_nodes, dtype=torch.float32).to(DEVICE), val_solutions


def rollout(model, dataset, greedy=False):
    """
    Runs the model on a batch of graphs (dataset).
    Returns:
      - greedy=True: tour_lengths
      - greedy=False: (tour_lengths, summed_log_prob)
    """
    model.eval() if greedy else model.train()

    batch_size, num_nodes, _ = dataset.size()

    curr_node = torch.zeros(batch_size, 1, dtype=torch.long).to(DEVICE)
    visited = torch.zeros(batch_size, num_nodes).to(DEVICE)
    visited[:, 0] = 1

    tour_log_probs = []
    tour_lengths = torch.zeros(batch_size).to(DEVICE)

    start_nodes = dataset[:, 0, :]
    prev_nodes = dataset[:, 0, :]

    for _step in range(num_nodes - 1):
        probs, _log_probs = model(dataset, curr_node, visited)

        if greedy:
            action = probs.argmax(dim=1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            tour_log_probs.append(dist.log_prob(action))

        visited.scatter_(1, action.unsqueeze(1), 1)
        curr_node = action.unsqueeze(1)

        curr_coords = dataset[torch.arange(batch_size), action]
        dist_step = torch.norm(curr_coords - prev_nodes, dim=1)
        tour_lengths += dist_step
        prev_nodes = curr_coords

    tour_lengths += torch.norm(start_nodes - prev_nodes, dim=1)

    if greedy:
        return tour_lengths
    return tour_lengths, torch.stack(tour_log_probs, dim=1).sum(dim=1)


def grad_norm_l2(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return float(total ** 0.5)


def train():
    # -----------------------
    # 0) W&B init
    # -----------------------
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_batches_per_epoch": TRAIN_BATCHES,
            "learning_rate": LEARNING_RATE,
            "embedding_dim": EMBEDDING_DIM,
            "num_nodes": NUM_NODES,
            "device": str(DEVICE),
        },
    )

    # -----------------------
    # 1) Initialize Models
    # -----------------------
    policy = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    baseline = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    baseline.load_state_dict(policy.state_dict())
    baseline.eval()

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # Optional: track params/gradients (can be a bit heavy)
    wandb.watch(policy, log="gradients", log_freq=200)

    # -----------------------
    # 2) Load Validation Data
    # -----------------------
    val_data, val_optimal_costs = load_validation_data()

    print(f"--- Starting Training on {DEVICE} ---")

    global_step = 0
    best_val_cost = float("inf")
    best_val_gap = float("inf")

    for epoch in range(NUM_EPOCHS):
        policy.train()
        epoch_loss_sum = 0.0
        epoch_cost_sum = 0.0
        epoch_baseline_cost_sum = 0.0
        epoch_adv_sum = 0.0

        t0 = time.time()
        iterator = tqdm(range(TRAIN_BATCHES), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for _ in iterator:
            global_step += 1

            data = torch.rand(BATCH_SIZE, NUM_NODES, 2).to(DEVICE) * 1000

            cost, log_probs = rollout(policy, data, greedy=False)

            with torch.no_grad():
                baseline_cost = rollout(baseline, data, greedy=True)

            advantage = cost - baseline_cost
            loss = (advantage * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            loss_item = float(loss.item())
            cost_mean = float(cost.mean().item())
            baseline_cost_mean = float(baseline_cost.mean().item())
            adv_mean = float(advantage.mean().item())
            gnorm = grad_norm_l2(policy)
            lr = optimizer.param_groups[0]["lr"]

            epoch_loss_sum += loss_item
            epoch_cost_sum += cost_mean
            epoch_baseline_cost_sum += baseline_cost_mean
            epoch_adv_sum += adv_mean

            iterator.set_postfix(loss=loss_item, cost=cost_mean)

            # -----------------------
            # W&B: per-step logging
            # -----------------------
            wandb.log(
                {
                    "train/loss": loss_item,
                    "train/cost": cost_mean,
                    "train/baseline_cost": baseline_cost_mean,
                    "train/advantage": adv_mean,
                    "train/grad_norm": gnorm,
                    "train/lr": lr,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        epoch_time = time.time() - t0

        # -----------------------
        # Validation
        # -----------------------
        print("\nValidating...")
        policy.eval()
        with torch.no_grad():
            val_batch = val_data[:, :NUM_NODES, :]
            rl_costs = rollout(policy, val_batch, greedy=True)
            val_cost_mean = float(rl_costs.mean().item())

            gaps = []
            for i, c in enumerate(rl_costs):
                opt = val_optimal_costs[i]
                if opt > 0:
                    gaps.append(((float(c.item()) - opt) / opt) * 100.0)
            val_gap_mean = float(np.mean(gaps)) if len(gaps) > 0 else float("nan")

            baseline_val_costs = rollout(baseline, val_batch, greedy=True)
            baseline_val_mean = float(baseline_val_costs.mean().item())

        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Avg Loss: {epoch_loss_sum / TRAIN_BATCHES:.4f}")
        print(f"  Val Avg Cost: {val_cost_mean:.2f}")
        print(f"  Val Optimality Gap (%): {val_gap_mean:.2f}")
        print(f"  Baseline Val Avg Cost: {baseline_val_mean:.2f}")

        # -----------------------
        # W&B: per-epoch logging
        # -----------------------
        wandb.log(
            {
                "epoch_metrics/train_loss_avg": epoch_loss_sum / TRAIN_BATCHES,
                "epoch_metrics/train_cost_avg": epoch_cost_sum / TRAIN_BATCHES,
                "epoch_metrics/train_baseline_cost_avg": epoch_baseline_cost_sum / TRAIN_BATCHES,
                "epoch_metrics/train_advantage_avg": epoch_adv_sum / TRAIN_BATCHES,
                "val/cost": val_cost_mean,
                "val/gap_percent": val_gap_mean,
                "val/baseline_cost": baseline_val_mean,
                "time/epoch_sec": epoch_time,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        # -----------------------
        # Baseline update + saving best model
        # -----------------------
        upgraded = False
        if val_cost_mean < baseline_val_mean:
            print("  >> UPGRADE: Policy beat Baseline. Updating Baseline Network.")
            baseline.load_state_dict(policy.state_dict())
            upgraded = True

        # Save best by validation cost (common)
        if val_cost_mean < best_val_cost:
            best_val_cost = val_cost_mean
            torch.save(policy.state_dict(), "best_tsp_model.pth")
            # log checkpoint to W&B
            wandb.save("best_tsp_model.pth")
            print("  >> SAVED: New best_tsp_model.pth (by val cost).")

        # Track "best gap" too (optional)
        if np.isfinite(val_gap_mean) and val_gap_mean < best_val_gap:
            best_val_gap = val_gap_mean

        wandb.log(
            {
                "baseline/upgraded": int(upgraded),
                "best/val_cost": best_val_cost,
                "best/val_gap_percent": best_val_gap,
            },
            step=global_step,
        )

    wandb.finish()


if __name__ == "__main__":
    train()
