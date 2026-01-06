import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TSPEncoder(nn.Module):
    """
    Encodes the static graph (city coordinates) into node embeddings 
    using a Multi-Head Attention Transformer.
    """
    def __init__(self, input_dim=2, embedding_dim=128, num_layers=3, num_heads=8):
        super(TSPEncoder, self).__init__()
        
        # 1. Initial Linear Projection (2D coords -> 128D vectors)
        self.init_embed = nn.Linear(input_dim, embedding_dim)
        
        # 2. Transformer Encoder Layers
        # We use PyTorch's built-in TransformerEncoderLayer for speed/stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True,   # Expect inputs as [Batch, Seq_Len, Dim]
            norm_first=True     # Pre-normalization (better for deep networks)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Input: x [Batch, Num_Cities, 2]
        Output: embeddings [Batch, Num_Cities, Embedding_Dim]
        """
        h = self.init_embed(x)
        h = self.transformer_encoder(h)
        return h

class AttentionModel(nn.Module):
    """
    The Full Policy Network.
    - Encodes the graph once.
    - Decodes one step at a time to choose the next city.
    """
    def __init__(self, embedding_dim=128, hidden_dim=128, num_layers=3, num_heads=8):
        super(AttentionModel, self).__init__()
        
        self.encoder = TSPEncoder(2, embedding_dim, num_layers, num_heads)
        
        # Decoder components
        # We need to project the "Context" (Graph Embedding + Current Node Embedding)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim)
        self.project_step_context = nn.Linear(embedding_dim, embedding_dim)
        
        # Attention Mechanism (Single Head for the Pointer)
        self.v = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.W_ref = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        # Uniform initialization often helps convergence in RL
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, static_input, current_node_idx, visited_mask):
        """
        Calculates the probability distribution for the next step.
        
        Args:
            static_input: [Batch, Num_Cities, 2] - The coordinates
            current_node_idx: [Batch, 1] - Index of the current city
            visited_mask: [Batch, Num_Cities] - 1 if visited, 0 if not
            
        Returns:
            probs: [Batch, Num_Cities] - Probability of visiting each city
            log_probs: [Batch, Num_Cities] - Log probabilities
        """
        batch_size, num_cities, _ = static_input.size()
        
        # 1. Encode the Graph (Context)
        # Note: In a real efficient loop, we would cache this and not re-compute every step.
        # But for simplicity in this week-long project, we re-compute or assume passed embeddings.
        node_embeddings = self.encoder(static_input) # [Batch, N, Emb]
        
        # 2. Define "Context"
        # Graph Context: Mean of all node embeddings (represents the "map")
        graph_embedding = node_embeddings.mean(dim=1) # [Batch, Emb]
        
        # Current Node Context: Embedding of the node we are currently standing on
        # Gathers the embedding at the index 'current_node_idx'
        current_node_embedding = node_embeddings.gather(
            1, 
            current_node_idx.unsqueeze(-1).expand(batch_size, 1, node_embeddings.size(-1))
        ).squeeze(1) # [Batch, Emb]
        
        # Combine contexts
        # Query = W_g * graph_emb + W_c * current_node_emb
        query = self.project_fixed_context(graph_embedding) + \
                self.project_step_context(current_node_embedding) # [Batch, Emb]
        
        # 3. Calculate Attention (Pointer Mechanism)
        # Compare Query (Context) with Keys (All Node Embeddings)
        # Scores = v^T * Tanh(W_ref * Nodes + W_q * Query)
        
        # Expand query to match number of cities
        query_expanded = query.unsqueeze(1).expand_as(node_embeddings) # [Batch, N, Emb]
        
        # Attention Score Calculation
        ref_proj = self.W_ref(node_embeddings)
        query_proj = self.W_q(query_expanded)
        
        scores = torch.sum(self.v * torch.tanh(ref_proj + query_proj), dim=-1) # [Batch, N]
        
        # 4. Masking
        # We must set the score of visited nodes to -Infinity so Softmax makes them 0
        scores = scores.masked_fill(visited_mask.bool(), float('-inf'))
        
        # 5. Probabilities
        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)
        
        return probs, log_probs

# --- Quick Test ---
if __name__ == "__main__":
    # Fake Batch of 2 instances, 10 cities each
    B, N, D = 2, 10, 2
    x = torch.randn(B, N, D)
    mask = torch.zeros(B, N)
    mask[:, 0] = 1 # Assume node 0 is visited
    curr = torch.zeros(B, 1, dtype=torch.long) # Currently at node 0
    
    model = AttentionModel()
    probs, _ = model(x, curr, mask)
    
    print("Output Shape:", probs.shape) # Should be [2, 10]
    print("Prob of visited node 0 (Should be 0):", probs[0, 0].item())
    print("Sum of probs (Should be 1):", probs[0].sum().item())