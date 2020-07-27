import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    """Neural network for usage in agent"""
    def __init__(self, param_dict={}):
        """ Initialize a Model object.

        Params
        ======
           param_dict(dictionary): contains size-information and dueling-label
        """
        super().__init__()

        input_size = param_dict.get("input_size", 37)
        self.output_size = param_dict.get("output_size", 4)
        hn = param_dict.get("hn", [128, 128, 64, 32])
        self.dueling = param_dict.get("dueling", True)

        if self.dueling:
            hn = [input_size] + hn + [self.output_size + 1]
        else:
            hn = [input_size] + hn + [self.output_size]

        self.hidden = nn.ModuleList()
        for k in range(len(hn)-1):
            self.hidden.append(nn.Linear(hn[k], hn[k+1]))

    def forward(self, x):
        """ Defines forward pass. Returns action-values given state x.

        Params
        ======
           x(torch.tensor): current state
        """
        for k in range(len(self.hidden)-1):
            x = F.relu(self.hidden[k](x))
        x = self.hidden[-1](x)

        if self.dueling:
            advantage, state_val = torch.split(x, self.output_size, dim=-1)
            mean = torch.mean(advantage, dim = -1).view(state_val.size())
            centered_advantage = advantage.sub(mean)
            q_vals = state_val + centered_advantage
            return q_vals
        else:
            return x
