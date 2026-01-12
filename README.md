# Agentic Vehicle Routing (Part A): Neural Combinatorial Optimization

> **Note:** This project is Part A of a series exploring autonomous routing agents. This phase focuses on building a "Neural Solver" that learns to solve the Traveling Salesman Problem (TSP) from scratch using Deep Reinforcement Learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

Traditional approaches to combinatorial optimization problems like the Traveling Salesman Problem (TSP) rely on handcrafted heuristics or industrial solvers like Google OR-Tools. This project explores a different path: **can a neural network learn the underlying logic of routing purely through trial and error?**

We build an Attention-based Deep Learning model and train it using Reinforcement Learning to construct optimal tours step-by-step. The agent observes a map of cities and learns a policy to minimize total travel distance without ever seeing a labeled "correct" answer.

### Key Features
* **Custom RL Environment:** A gymnasium-style environment that simulates TSP constraints (visiting every city exactly once).
* **Attention-Based Policy:** An Encoder-Decoder architecture using Multi-Head Attention and a Pointer mechanism to handle graph-structured data invariant to input permutation.
* **REINFORCE with Baseline:** Trains using policy gradients stabilized by a greedy rollout baseline with a "King of the Hill" update strategy.
* **OR-Tools Benchmark:** Includes an automated pipeline to generate ground-truth solutions using Google OR-Tools' Guided Local Search for measuring the optimality gap.
* **Visualization:** Scripts to generate side-by-side video comparisons of the neural agent vs. traditional solvers.

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Eashwar-S/Routing_agent.git
    cd Routing_agent
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install torch numpy matplotlib tqdm gymnasium ortools wandb imageio-ffmpeg
    ```

---

## Usage

### 1. Data Generation & Benchmarking
First, generate a fixed validation set and solve it using OR-Tools to establish ground truth for tracking the optimality gap.

```bash
python instance_preparation.py
```

### 2. Training
Start the training loop. The script generates training data on the fly. It will use Weights & Biases for logging if configured.

**Run with Weights & Biases logging but need to have an wandb account and API**
```bash
python train.py --wandb
```

**Run locally without logging**
```bash
python train.py --no-wandb
```

### 3. Visualization
Once trained, use the saved model (e.g., best_tsp_model.pth) to generate a side-by-side video comparison against OR-Tools on a new random instance.

```bash
python visualize_comparison.py
```

## Visual Comparison: RL Agent vs. OR-Tools
![](/output/tsp_comparison.gif)


## Blog
More detailed information is available in the blog [here](https://www.eashwarsathyamurthy.com/post/agentic-vehicle-routing-part-a)