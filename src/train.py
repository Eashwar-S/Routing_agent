import os
import time
import argparse
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import AttentionModel
from tsp_env import TSPEnv 

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
BATCH_SIZE = 64
TRAIN_BATCHES = 100
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
NUM_NODES = 50
VALIDATION_PATH = "../tsp_instances"
SOLUTION_PATH = "../tsp_solutions"

# W&B settings (Default, can be overridden by args)
WANDB_PROJECT = "tsp-attention-rl"
WANDB_RUN_NAME = "tsp50-baseline-v7"

def parse_args():
    parser = argparse.ArgumentParser(description="Train TSP RL Agent")
    # Boolean flag for wandb
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='Disable W&B logging')
    parser.set_defaults(wandb=True) # Default is True
    return parser.parse_args()

def load_validation_data():
    """Loads the fixed validation set."""
    val_nodes = []
    val_solutions = []

    if not os.path.exists(VALIDATION_PATH):
        print(f"Error: Validation path '{VALIDATION_PATH}' not found.")
        return torch.tensor([], device=DEVICE), []

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
                    parts = line.split()
                    if len(parts) >= 3:
                        nodes.append([float(parts[1]), float(parts[2])])
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
        
    if len(val_nodes) == 0:
        return torch.tensor([], device=DEVICE), []
        
    return torch.tensor(val_nodes, dtype=torch.float32).to(DEVICE), val_solutions


def rollout(model, dataset, greedy=False):
    model.eval() if greedy else model.train()
    batch_size, num_nodes, _ = dataset.size()
    curr_node = torch.zeros(batch_size, 1, dtype=torch.long).to(DEVICE)
    visited = torch.zeros(batch_size, num_nodes).to(DEVICE)
    visited[:, 0] = 1
    tour_log_probs = []
    tour_entropies = []
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
            tour_entropies.append(dist.entropy())
        visited.scatter_(1, action.unsqueeze(1), 1)
        curr_node = action.unsqueeze(1)
        curr_coords = dataset[torch.arange(batch_size), action]
        dist_step = torch.norm(curr_coords - prev_nodes, dim=1)
        tour_lengths += dist_step
        prev_nodes = curr_coords

    tour_lengths += torch.norm(start_nodes - prev_nodes, dim=1)
    if greedy:
        return tour_lengths
    return tour_lengths, torch.stack(tour_log_probs, dim=1).sum(dim=1), torch.stack(tour_entropies, dim=1).sum(dim=1)

def grad_norm_l2(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return float(total ** 0.5)


def train():
    # 1. Parse Args
    args = parse_args()
    use_wandb = args.wandb

    # Safe Import
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config={
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_nodes": NUM_NODES,
                    "device": str(DEVICE),
                },
            )
            print(f"--> WandB logging enabled. Run: {WANDB_RUN_NAME}")
        except ImportError:
            print("Warning: 'wandb' library not found. Disabling W&B logging.")
            use_wandb = False
    else:
        print("--> WandB logging DISABLED.")

    # 2. Init Model
    policy = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    baseline = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    ckpt_path = "best_tsp_model.pth"
    if os.path.exists(ckpt_path):
        print(f"Resuming training from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE)
        policy.load_state_dict(state)
        baseline.load_state_dict(state)
    else:
        print("No checkpoint found. Initializing baseline from policy.")
        baseline.load_state_dict(policy.state_dict())
    baseline.eval()

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    total_steps = NUM_EPOCHS * TRAIN_BATCHES
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.05)

    if use_wandb:
        wandb.watch(policy, log="gradients", log_freq=200)

    # 3. Load Data
    val_data, val_optimal_costs = load_validation_data()

    print(f"--- Starting Training on {DEVICE} ---")
    global_step = 0
    best_val_cost = float("inf")

    for epoch in range(NUM_EPOCHS):
        policy.train()
        epoch_loss_sum = 0.0
        
        t0 = time.time()
        iterator = tqdm(range(TRAIN_BATCHES), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for _ in iterator:
            global_step += 1
            data = torch.rand(BATCH_SIZE, NUM_NODES, 2).to(DEVICE) * 1000
            
            cost, log_probs, ent = rollout(policy, data, greedy=False)
            with torch.no_grad():
                baseline_cost, _, _ = rollout(baseline, data, greedy=False)

            entropy_coef = max(0.01 * (1 - 3*global_step / total_steps), 0.01)
            advantage = cost - baseline_cost
            loss = (advantage * log_probs).mean() - entropy_coef * ent.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Baseline EMA
            with torch.no_grad():
                for bp, pp in zip(baseline.parameters(), policy.parameters()):
                    bp.data.mul_(0.99).add_(pp.data, alpha=0.01)

            loss_val = loss.item()
            cost_val = cost.mean().item()
            epoch_loss_sum += loss_val
            iterator.set_postfix(loss=loss_val, cost=cost_val)

            if use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/cost": cost_val,
                    "train/advantage": advantage.mean().item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/entropy": ent.mean().item()
                }, step=global_step)

        # Validation
        policy.eval()
        val_cost_mean = 0.0
        if len(val_data) > 0:
            with torch.no_grad():
                val_batch = val_data[:, :NUM_NODES, :]
                rl_costs = rollout(policy, val_batch, greedy=True)
                val_cost_mean = rl_costs.mean().item()

        print(f"Epoch {epoch+1} Val Cost: {val_cost_mean:.2f}")

        if use_wandb:
            wandb.log({"val/cost": val_cost_mean, "epoch": epoch+1}, step=global_step)

        if val_cost_mean < best_val_cost and len(val_data) > 0:
            best_val_cost = val_cost_mean
            torch.save(policy.state_dict(), "best_tsp_model.pth")
            print("  >> SAVED: New best_tsp_model.pth")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()