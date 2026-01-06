import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import AttentionModel # Assumes model code is in model.py
from tsp_env import TSPEnv       # Assumes env code is in tsp_env.py

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20           # Total training epochs
BATCH_SIZE = 64           # Batch size for training
TRAIN_BATCHES = 100       # How many batches per epoch (Total = 6400 samples/epoch)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
NUM_NODES = 20            # Training on 20 nodes first (easier to learn)
VALIDATION_PATH = "../tsp_instances"
SOLUTION_PATH = "../tsp_solutions"

def load_validation_data():
    """Loads the fixed validation set (the 500 instances you created)."""
    val_nodes = []
    val_solutions = []
    
    # We load just the first 50 files for speed in this demo
    files = sorted([f for f in os.listdir(VALIDATION_PATH) if f.endswith(".txt")])[:50]
    
    print(f"Loading {len(files)} validation instances...")
    
    for f in files:
        # Load Nodes
        nodes = []
        with open(os.path.join(VALIDATION_PATH, f), 'r') as file:
            c_section = False
            for line in file:
                if "NODE_COORD_SECTION" in line: c_section = True; continue
                if c_section and "EOF" not in line:
                    _, x, y = line.split()
                    nodes.append([float(x), float(y)])
        val_nodes.append(nodes)
        
        # Load Optimal Cost (for Gap calculation)
        sol_name = f.replace(".txt", "_sol.txt")
        cost = 0
        if os.path.exists(os.path.join(SOLUTION_PATH, sol_name)):
            with open(os.path.join(SOLUTION_PATH, sol_name), 'r') as file:
                for line in file:
                    if "COST" in line:
                        cost = float(line.split(":")[-1].strip())
        val_solutions.append(cost)
        
    return torch.tensor(val_nodes, dtype=torch.float32).to(DEVICE), val_solutions

def rollout(model, dataset, greedy=False):
    """
    Runs the model on a batch of graphs (dataset).
    Returns: tour_lengths, log_probabilities
    """
    model.eval() if greedy else model.train()
    
    batch_size, num_nodes, _ = dataset.size()
    
    # Initial State
    curr_node = torch.zeros(batch_size, 1, dtype=torch.long).to(DEVICE) # Start at 0
    visited = torch.zeros(batch_size, num_nodes).to(DEVICE)
    visited[:, 0] = 1 # Mark start as visited
    
    tour_log_probs = []
    tour_lengths = torch.zeros(batch_size).to(DEVICE)
    
    # Store initial nodes to calculate return distance later
    start_nodes = dataset[:, 0, :] 
    prev_nodes = dataset[:, 0, :]
    
    # Decode Steps
    for step in range(num_nodes - 1):
        # 1. Get Probabilities from Model
        probs, log_probs = model(dataset, curr_node, visited)
        
        # 2. Select Action
        if greedy:
            action = probs.argmax(dim=1) # Greedy decoding
        else:
            # Sampling decoding (needed for exploration during training)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Store log_prob of the chosen action for backprop
            tour_log_probs.append(dist.log_prob(action))
        
        # 3. Update State
        visited.scatter_(1, action.unsqueeze(1), 1)
        curr_node = action.unsqueeze(1)
        
        # 4. Calculate Step Distance
        # Gather coordinates of chosen nodes
        # action is [Batch], dataset is [Batch, N, 2] -> selected is [Batch, 2]
        # We use a gather trick or simple indexing
        curr_coords = dataset[torch.arange(batch_size), action]
        dist = torch.norm(curr_coords - prev_nodes, dim=1)
        tour_lengths += dist
        
        prev_nodes = curr_coords

    # 5. Return to Start Distance
    tour_lengths += torch.norm(start_nodes - prev_nodes, dim=1)
    
    if greedy:
        return tour_lengths
    else:
        return tour_lengths, torch.stack(tour_log_probs, dim=1).sum(dim=1)

def train():
    # 1. Initialize Models
    # 'policy' is the agent we are training
    # 'baseline' is a frozen copy of the best agent so far
    policy = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    baseline = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    baseline.load_state_dict(policy.state_dict())
    baseline.eval()
    
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # 2. Load Validation Data
    val_data, val_optimal_costs = load_validation_data()
    
    print(f"--- Starting Training on {DEVICE} ---")
    
    for epoch in range(NUM_EPOCHS):
        policy.train()
        avg_loss = 0
        
        # tqdm for progress bar
        iterator = tqdm(range(TRAIN_BATCHES), desc=f"Epoch {epoch+1}")
        
        for _ in iterator:
            # A. Generate Random Training Batch on the fly
            data = torch.rand(BATCH_SIZE, NUM_NODES, 2).to(DEVICE) * 1000 # Scale 0-1000
            
            # B. Run Policy (Sampling Mode) -> Get Costs and LogProbs
            cost, log_probs = rollout(policy, data, greedy=False)
            
            # C. Run Baseline (Greedy Mode) -> Get Baseline Costs
            with torch.no_grad():
                baseline_cost = rollout(baseline, data, greedy=True)
                
            # D. Calculate Advantage
            # If (Cost - Baseline) is negative, we did better! -> Positive Reinforcement
            advantage = cost - baseline_cost
            
            # E. Loss = Advantage * LogProb
            # We minimize loss, so negative advantage (good) * log_prob -> minimizes
            loss = (advantage * log_probs).mean()
            
            # F. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            iterator.set_postfix(loss=loss.item(), cost=cost.mean().item())
            
        # --- Validation & Baseline Update ---
        print("\nValidating...")
        with torch.no_grad():
            # 1. Check Optimality Gap
            # Note: val_data might need resizing if validation files have different N than training
            # For simplicity, we assume validation files match NUM_NODES or we slice them
            val_batch = val_data[:, :NUM_NODES, :] 
            
            rl_costs = rollout(policy, val_batch, greedy=True)
            rl_mean = rl_costs.mean().item()
            
            # Calculate Gap against OR-Tools
            gaps = []
            for i, c in enumerate(rl_costs):
                opt = val_optimal_costs[i]
                if opt > 0:
                    gap = ((c.item() - opt) / opt) * 100
                    gaps.append(gap)
            avg_gap = np.mean(gaps)
            
            print(f"Epoch {epoch+1} Results:")
            print(f"  Average Cost: {rl_mean:.2f}")
            print(f"  Optimality Gap (vs OR-Tools): {avg_gap:.2f}%")
            
            # 2. Baseline Update Check (One-Sample t-test logic simplified)
            # If current policy is consistently better than baseline on validation, update baseline
            baseline_val_costs = rollout(baseline, val_batch, greedy=True)
            
            if rl_mean < baseline_val_costs.mean().item():
                print("  >> UPGRADE: Policy beat Baseline. Updating Baseline Network.")
                baseline.load_state_dict(policy.state_dict())
                torch.save(policy.state_dict(), "best_tsp_model.pth")
            else:
                print("  >> KEEP: Baseline is still stronger.")

if __name__ == "__main__":
    train()