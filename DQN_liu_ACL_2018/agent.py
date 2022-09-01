import torch
import torch.nn as nn

class MLPAgent(nn.Module):
    
    def __init__(self, state_dim: int, n_action: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_action)
        )

    def forward(self, state: torch.IntTensor) -> torch.FloatTensor:
        return self.mlp(state)