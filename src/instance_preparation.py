import os
import random
import math
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
TRAIN_PATH = "../data/train"
VAL_PATH = "../data/val"

MIN_NODES = 20
MAX_NODES = 100 

NUM_TRAIN = 10000 
NUM_VAL = 500      

def ensure_dirs():
    os.makedirs(os.path.join(TRAIN_PATH, "instances"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_PATH, "solutions"), exist_ok=True)
    os.makedirs(os.path.join(VAL_PATH, "instances"), exist_ok=True)
    os.makedirs(os.path.join(VAL_PATH, "solutions"), exist_ok=True)

def generate_and_solve(args):
    """
    Worker function to generate one instance and solve it.
    Args packed in tuple for map: (idx, base_path, min_n, max_n)
    """
    idx, base_path, min_n, max_n = args
    
    # 1. Randomize Size
    # Each process needs its own random seed state, but standard random is thread-safe enough here
    curr_nodes = random.randint(min_n, max_n)
    
    # 2. Generate Instance
    name = f"tsp_{curr_nodes}_{idx}"
    filename = os.path.join(base_path, "instances", f"{name}.txt")
    
    nodes = []
    for i in range(1, curr_nodes + 1):
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        nodes.append((i, x, y))

    with open(filename, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {curr_nodes}\n")
        f.write("NODE_COORD_SECTION\n")
        for node_id, x, y in nodes:
            f.write(f"{node_id} {x} {y}\n")
            
    # 3. Solve with OR-Tools
    solve_tsp_ortools(nodes, name, base_path)
    
    return curr_nodes

def solve_tsp_ortools(nodes, instance_name, base_path):
    """Solves the instance and saves the tour."""
    node_ids = [n[0] for n in nodes]
    coords = {n[0]: (n[1], n[2]) for n in nodes}
    size = len(node_ids)
    
    dist_matrix = {}
    for i in node_ids:
        dist_matrix[i] = {}
        for j in node_ids:
            if i == j: dist_matrix[i][j] = 0
            else:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist_matrix[i][j] = int(math.sqrt((x1-x2)**2 + (y1-y2)**2))

    manager = pywrapcp.RoutingIndexManager(size, 1, 0) 
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[node_ids[from_node]][node_ids[to_node]]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 1 

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            tour.append(node_ids[node_idx])
            index = solution.Value(routing.NextVar(index))
        
        sol_filename = os.path.join(base_path, "solutions", f"{instance_name}_sol.txt")
        with open(sol_filename, "w") as f:
            f.write(f"COST: {solution.ObjectiveValue()}\n")
            f.write("TOUR_SECTION\n")
            for node in tour:
                f.write(f"{node}\n")

def generate_dataset_parallel(num_instances, base_path, desc):
    ensure_dirs()
    print(f"--- Generating {desc} Set ({num_instances} instances) ---")
    
    # Prepare arguments for each worker
    # We pass min/max nodes explicitly to avoid global variable issues in workers
    tasks = [(i, base_path, MIN_NODES, MAX_NODES) for i in range(num_instances)]
    
    generated_sizes = []
    
    # Determine CPUs (Colab usually has 2, sometimes 4)
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPU cores for parallel generation.")

    with Pool(processes=num_cpus) as pool:
        # imap_unordered is faster as it yields results as soon as they finish
        for size in tqdm(pool.imap_unordered(generate_and_solve, tasks), total=num_instances):
            generated_sizes.append(size)
            
    return generated_sizes

def plot_distribution():
    """
    Scans the data directories to infer instance sizes from filenames
    and plots the distribution.
    """
    print("Scanning directories to generate distribution histogram...")
    
    paths = {
        "Train": TRAIN_PATH + "/instances",
        "Val": VAL_PATH + "/instances"
    }
    
    sizes = {"Train": [], "Val": []}

    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
            
        # Filename format: tsp_{num_nodes}_{idx}.txt
        # Example: tsp_25_100.txt -> splits to ['tsp', '25', '100.txt']
        files = [f for f in os.listdir(path) if f.endswith(".txt")]
        for f in files:
            try:
                parts = f.split('_')
                if len(parts) >= 2:
                    node_count = int(parts[1])
                    sizes[key].append(node_count)
            except ValueError:
                continue # Skip files that don't match pattern

    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Define common bins for consistent x-axis
    all_sizes = sizes["Train"] + sizes["Val"]
    if not all_sizes:
        print("No data found to plot.")
        return
        
    min_bin = min(all_sizes)
    max_bin = max(all_sizes)
    bins = range(min_bin, max_bin + 2)

    # Train Plot
    plt.subplot(1, 2, 1)
    plt.hist(sizes["Train"], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Training Set (N={len(sizes['Train'])})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Count")

    # Val Plot
    plt.subplot(1, 2, 2)
    plt.hist(sizes["Val"], bins=bins, color='salmon', edgecolor='black', alpha=0.7)
    plt.title(f"Validation Set (N={len(sizes['Val'])})")
    plt.xlabel("Number of Nodes")

    plt.tight_layout()
    plt.savefig("../data/data_distribution.png")
    print(f"Histogram saved to 'data_distribution.png'")



if __name__ == "__main__":
    # Validate
    # val_sizes = generate_dataset_parallel(NUM_VAL, VAL_PATH, "Validation")
    
    # Train
    # train_sizes = generate_dataset_parallel(NUM_TRAIN, TRAIN_PATH, "Training")
    
    # Plot
    plot_distribution()
    
    print("\nData Generation Complete!")