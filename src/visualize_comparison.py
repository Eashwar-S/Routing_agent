import torch
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Import your model definition
from model import AttentionModel 

# --- Config ---
MODEL_PATH = "best_tsp_model.pth"
NUM_NODES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 128

# ----------------------------------------
# 1. Helper: OR-Tools Solver (The "Baseline")
# ----------------------------------------
def get_ortools_tour(nodes):
    """Returns the list of node indices for the optimal tour."""
    # Convert list of coords to dict for OR-Tools logic
    node_dict = {i: nodes[i] for i in range(len(nodes))}
    
    # Distance Matrix
    size = len(nodes)
    dist_matrix = {}
    for i in range(size):
        dist_matrix[i] = {}
        for j in range(size):
            if i == j: dist_matrix[i][j] = 0
            else:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                dist = int(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
                dist_matrix[i][j] = dist

    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2 

    sol = routing.SolveWithParameters(params)
    
    tour = []
    if sol:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = sol.Value(routing.NextVar(index))
        tour.append(0) # Return to start
    return tour, sol.ObjectiveValue()

# ----------------------------------------
# 2. Helper: RL Agent Solver (The "Challenger")
# ----------------------------------------
def get_rl_tour(model, nodes):
    """Runs the model greedily to get the tour sequence."""
    model.eval()
    
    # Prepare Input Tensor [1, 50, 2]
    # Scale nodes to 0-1000 if they aren't already, or normalize if model expects 0-1
    # Assuming model trained on 0-1000 scale:
    data = torch.tensor(nodes, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    batch_size, num_nodes, _ = data.size()
    
    # Init State
    curr_node = torch.zeros(batch_size, 1, dtype=torch.long).to(DEVICE)
    visited = torch.zeros(batch_size, num_nodes).to(DEVICE)
    visited[:, 0] = 1
    
    tour = [0] # Start at 0
    
    with torch.no_grad():
        for _ in range(num_nodes - 1):
            probs, _ = model(data, curr_node, visited)
            
            # Greedy Selection (Max Probability)
            next_node = probs.argmax(dim=1)
            
            # Update State
            visited.scatter_(1, next_node.unsqueeze(1), 1)
            curr_node = next_node.unsqueeze(1)
            
            tour.append(next_node.item())
            
    tour.append(0) # Return to start
    
    # Calculate Cost (Euclidean)
    cost = 0
    for i in range(len(tour)-1):
        p1 = np.array(nodes[tour[i]])
        p2 = np.array(nodes[tour[i+1]])
        cost += np.linalg.norm(p1 - p2)
        
    return tour, cost

# ----------------------------------------
# 3. Main Video Generation
# ----------------------------------------
def create_comparison_video():
    print("--- 1. Generating Instance ---")
    # Generate 50 random nodes (0-1000 scale)
    nodes = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(NUM_NODES)]
    
    print("--- 2. Solving with OR-Tools ---")
    or_tour, or_cost = get_ortools_tour(nodes)
    
    print("--- 3. Solving with RL Agent ---")
    # Load Model
    model = AttentionModel(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    rl_tour, rl_cost = get_rl_tour(model, nodes)
    
    print(f"Results -> OR-Tools: {or_cost:.2f} | RL Agent: {rl_cost:.2f}")
    
    print("--- 4. Creating Animation ---")
    
    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Unpack coords for easy plotting
    x_coords = [n[0] for n in nodes]
    y_coords = [n[1] for n in nodes]
    
    # Setup Axes
    for ax, title, c in zip([ax1, ax2], 
                            [f"OR-Tools (Benchmark)\nCost: {or_cost:.0f}", f"RL Agent (Ours)\nCost: {rl_cost:.0f}"], 
                            ['green', 'blue']):
        ax.set_title(title, fontsize=14, fontweight='bold', color=c)
        ax.set_xlim(-50, 1050)
        ax.set_ylim(-50, 1050)
        ax.set_aspect('equal')
        # Plot all nodes initially
        ax.scatter(x_coords, y_coords, c='black', s=40, zorder=2)
        # Highlight start node
        ax.scatter(x_coords[0], y_coords[0], c='red', s=100, marker='*', zorder=3, label='Depot')
        ax.legend(loc='upper right')

    # Lines to be updated
    line1, = ax1.plot([], [], 'g-', lw=2, alpha=0.7)
    line2, = ax2.plot([], [], 'b-', lw=2, alpha=0.7)
    
    # Dot for current head
    head1, = ax1.plot([], [], 'go', markersize=8)
    head2, = ax2.plot([], [], 'bo', markersize=8)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        head1.set_data([], [])
        head2.set_data([], [])
        return line1, line2, head1, head2

    def update(frame):
        # Frame goes from 0 to 50 (number of edges)
        # OR-Tools Update
        if frame < len(or_tour):
            visited_indices = or_tour[:frame+1]
            x_path = [nodes[i][0] for i in visited_indices]
            y_path = [nodes[i][1] for i in visited_indices]
            line1.set_data(x_path, y_path)
            head1.set_data([x_path[-1]], [y_path[-1]])

        # RL Agent Update
        if frame < len(rl_tour):
            visited_indices = rl_tour[:frame+1]
            x_path = [nodes[i][0] for i in visited_indices]
            y_path = [nodes[i][1] for i in visited_indices]
            line2.set_data(x_path, y_path)
            head2.set_data([x_path[-1]], [y_path[-1]])
            
        return line1, line2, head1, head2

    # Create Animation
    # Frames = Number of edges + a pause at the end
    total_frames = NUM_NODES + 15 
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, 
                                  interval=100, blit=True) # 100ms per frame

    # Save
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    save_path = "tsp_comparison.mp4"
    ani.save(save_path, writer=writer)
    
    print(f"Video saved successfully to: {save_path}")
    plt.close()

if __name__ == "__main__":
    create_comparison_video()