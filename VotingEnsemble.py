import torch
import torch.nn as nn

class VotingEnsemble(nn.Module):
    def __init__(self, models):
        super(VotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = [model(x) for model in self.models]
        avg_logits = torch.mean(torch.stack(logits), dim=0)
        return avg_logits