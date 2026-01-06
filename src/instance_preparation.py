import os
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tqdm import tqdm

# --- Configuration ---
INSTANCE_FOLDER = "../tsp_instances"
SOLUTION_FOLDER = "../tsp_solutions"
NUM_INSTANCES = 500
MIN_NODES = 100
MAX_NODES = 500

# Create directories
os.makedirs(INSTANCE_FOLDER, exist_ok=True)
os.makedirs(SOLUTION_FOLDER, exist_ok=True)

def generate_tsp_instance(num_nodes, instance_id):
    """Generates random 2D coordinates and saves in TSPLIB format."""
    name = f"tsp_{instance_id}"
    filename = os.path.join(INSTANCE_FOLDER, f"{name}.txt")
    
    nodes = []
    # 1000x1000 grid
    for i in range(1, num_nodes + 1):
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        nodes.append((i, x, y))

    with open(filename, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write(f"COMMENT: Random TSP instance\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {num_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for node_id, x, y in nodes:
            f.write(f"{node_id} {x} {y}\n")
            
    return filename

def parse_tsp_instance(filepath):
    """Parses a TSPLIB formatted file."""
    nodes = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    coord_section = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            coord_section = True
            continue
        if "EOF" in line:
            break
        if coord_section:
            parts = line.strip().split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[node_id] = (x, y)
    return nodes

def solve_tsp_ortools(nodes, instance_name):
    """
    Solves TSP using OR-Tools with Guided Local Search.
    Saves solution with GAP and STATUS fields.
    """
    # 1. Create Distance Matrix
    node_ids = list(nodes.keys())
    size = len(node_ids)
    dist_matrix = {}
    
    for from_node in node_ids:
        dist_matrix[from_node] = {}
        for to_node in node_ids:
            if from_node == to_node:
                dist_matrix[from_node][to_node] = 0
            else:
                x1, y1 = nodes[from_node]
                x2, y2 = nodes[to_node]
                dist = int(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
                dist_matrix[from_node][to_node] = dist

    # 2. Setup Routing Model
    manager = pywrapcp.RoutingIndexManager(size, 1, 0) 
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[node_ids[from_node]][node_ids[to_node]]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. Search Parameters (Optimized for Quality)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 2 # Increase for better optimality on >100 nodes

    # 4. Solve
    solution = routing.SolveWithParameters(search_parameters)

    # 5. Process Result
    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            tour.append(node_ids[node_idx])
            index = solution.Value(routing.NextVar(index))
        
        # Determine Status/Gap
        # Note: Routing library doesn't give Proven Gap easily. 
        # We assume 0.0 gap for our ground truth, but mark status.
        status_id = routing.status()
        status_map = {
            0: "ROUTING_NOT_SOLVED",
            1: "ROUTING_SUCCESS", # Found a solution (usually hit time limit)
            2: "ROUTING_FAIL",
            3: "ROUTING_FAIL_TIMEOUT",
            4: "ROUTING_INVALID"
        }
        status_str = status_map.get(status_id, "UNKNOWN")
        
        # Save Solution File
        sol_filename = os.path.join(SOLUTION_FOLDER, f"{instance_name}_sol.txt")
        with open(sol_filename, "w") as f:
            f.write(f"NAME : {instance_name}_sol.txt\n")
            f.write(f"COMMENT : OR-Tools Guided Local Search\n")
            f.write(f"TYPE : TOUR\n")
            f.write(f"DIMENSION : {size}\n")
            f.write(f"COST : {solution.ObjectiveValue()}\n")
            f.write(f"STATUS : {status_str}\n")
            # Since we treat this as Ground Truth for the RL agent:
            f.write(f"GAP : 0.00% (Reference Solution)\n") 
            f.write("TOUR_SECTION\n")
            for node in tour:
                f.write(f"{node}\n")
            f.write("-1\n")
            
        return tour
    else:
        print(f"No solution found for {instance_name}")
        return None

def visualize_solution(nodes, tour, title="TSP Solution"):
    """
    Visualizes the TSP solution using NetworkX.
    Green Node = Start/End.
    """
    G = nx.DiGraph()
    pos = {}
    
    # Add nodes
    for node_id, (x, y) in nodes.items():
        G.add_node(node_id)
        pos[node_id] = (x, y)
    
    # Add edges
    edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)] 
        edges.append((u, v))
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue')
    # Highlight Start Node (first in tour)
    nx.draw_networkx_nodes(G, pos, nodelist=[tour[0]], node_size=150, node_color='green')
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5, arrowstyle='-|>', arrowsize=15)
    
    # Draw Labels (optional, can be cluttered for 100 nodes)
    # nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('on') # Show axis for coordinates
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Generating {NUM_INSTANCES} Instances ---")
    
    # 1. Generate
    for i in range(1, NUM_INSTANCES + 1):
        n_nodes = random.randint(MIN_NODES, MAX_NODES)
        generate_tsp_instance(n_nodes, i)
        if i % 50 == 0: print(f"Generated {i}/{NUM_INSTANCES}...")

    print("\n--- Solving Instances with OR-Tools ---")

    # 2. Solve
    for i in tqdm(range(1, NUM_INSTANCES + 1), desc="Solving TSP", unit="inst"):
        instance_name = f"tsp_{i}"
        instance_path = os.path.join(INSTANCE_FOLDER, f"{instance_name}.txt")
        
        nodes = parse_tsp_instance(instance_path)
        tour = solve_tsp_ortools(nodes, instance_name)
        
        # Visualize the first instance only
        # Note: plt.show() blocks execution, so the progress bar will pause 
        # until you close the window.
        if i == 1:
            # We use tqdm.write to print above the progress bar without breaking it
            tqdm.write(f"Visualizing solution for {instance_name}...")
            visualize_solution(nodes, tour, title=f"Solution: {instance_name}")

    print(f"\nDone! Solutions saved in '{SOLUTION_FOLDER}'.")