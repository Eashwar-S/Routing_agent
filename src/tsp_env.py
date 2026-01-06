import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gymnasium import spaces

class TSPEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Traveling Salesman Problem.
    
    State Space:
        - nodes: (N, 2) array of coordinates (Static)
        - visited: (N,) binary array (1 = visited, 0 = unvisited)
        - current_node: Integer index of current position
        
    Action Space:
        - Discrete(N): Index of the next node to visit.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 1}

    def __init__(self, num_nodes=20, env_size=1000):
        super(TSPEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.env_size = env_size
        
        # Actions: Choose any node index
        self.action_space = spaces.Discrete(num_nodes)
        
        # Observation: Dict containing graph info and current state
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(low=0, high=env_size, shape=(num_nodes, 2), dtype=np.float32),
            "visited": spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.int8),
            "current_node": spaces.Discrete(num_nodes)
        })
        
        # State variables
        self.nodes = None
        self.visited = None
        self.current_node = None
        self.step_count = 0
        self.total_distance = 0.0
        self.tour = [] # To store the sequence of actions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Generate new random graph (or load from options if needed)
        self.nodes = np.random.randint(0, self.env_size, size=(self.num_nodes, 2)).astype(np.float32)
        
        # 2. Reset State
        self.visited = np.zeros(self.num_nodes, dtype=np.int8)
        self.step_count = 0
        self.total_distance = 0.0
        self.tour = []
        
        # 3. Start at node 0 (Arbitrary convention, reduces search space)
        self.current_node = 0
        self.visited[0] = 1
        self.tour.append(0)
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one step in the environment.
        Reward = Negative Distance to next node.
        """
        # Check if action is valid (not visited yet)
        if self.visited[action] == 1:
            # PENALTY: Agent tried to visit an already visited city
            # We end the episode immediately with a large negative reward
            reward = -2.0 * self.env_size 
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {"error": "Invalid Action"}

        # Calculate distance
        prev_pos = self.nodes[self.current_node]
        curr_pos = self.nodes[action]
        dist = np.linalg.norm(prev_pos - curr_pos)
        
        # Update State
        self.current_node = action
        self.visited[action] = 1
        self.tour.append(action)
        self.step_count += 1
        self.total_distance += dist
        
        # Reward is negative distance (minimization problem)
        reward = -dist
        
        terminated = False
        truncated = False
        
        # Check if Tour is Complete (visited all nodes)
        if self.step_count == self.num_nodes - 1: # -1 because we started at node 0
            # Add distance back to start (Node 0) to close the loop
            start_pos = self.nodes[self.tour[0]]
            last_pos = self.nodes[self.tour[-1]]
            return_dist = np.linalg.norm(last_pos - start_pos)
            
            self.total_distance += return_dist
            reward -= return_dist # Add the return cost to the final step reward
            
            terminated = True # Episode Done
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return {
            "nodes": self.nodes,
            "visited": self.visited,
            "current_node": self.current_node
        }

    def render(self):
        """Simple Matplotlib Render"""
        if self.nodes is None: return
        
        plt.figure(figsize=(6, 6))
        # Draw Nodes
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], c='blue', s=50)
        # Highlight Start
        plt.scatter(self.nodes[0, 0], self.nodes[0, 1], c='green', s=100, label="Start")
        # Highlight Current
        plt.scatter(self.nodes[self.current_node, 0], self.nodes[self.current_node, 1], c='red', s=80, label="Current")
        
        # Draw Lines (Tour so far)
        if len(self.tour) > 1:
            tour_coords = self.nodes[self.tour]
            plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'k-', alpha=0.6)
            
        plt.legend()
        plt.title(f"Steps: {self.step_count} | Dist: {self.total_distance:.2f}")
        plt.show()

# --- Quick Test to Verify Code ---
if __name__ == "__main__":
    print("Testing TSP Environment...")
    env = TSPEnv(num_nodes=50)
    obs, info = env.reset()
    
    done = False
    print(f"Start Node: {obs['current_node']}")
    
    # Random Agent Loop
    while not done:
        # Simple Masking Logic for Random Agent (Our RL agent will learn this!)
        visited_mask = obs['visited']
        available_nodes = np.where(visited_mask == 0)[0]
        
        if len(available_nodes) > 0:
            action = np.random.choice(available_nodes)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Went to {action}, Reward: {reward:.2f}")
        else:
            break
            
    print(f"Total Tour Distance: {env.total_distance:.2f}")
    env.render()