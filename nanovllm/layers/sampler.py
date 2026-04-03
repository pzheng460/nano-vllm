import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = temperatures <= 1e-10
        # Clamp to avoid division by zero; greedy paths use argmax directly
        temperatures = temperatures.clamp(min=1e-10)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # For greedy (temperature=0): use direct argmax (deterministic)
        if greedy.any():
            greedy_tokens = logits.argmax(dim=-1)
            sample_tokens = torch.where(greedy, greedy_tokens, sample_tokens)
        return sample_tokens
