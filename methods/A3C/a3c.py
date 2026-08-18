import torch.nn as nn
import torch.nn.functional as F
import torch
import os


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def actor(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def critic(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_critic(x)
        return v

    def save(self, path):
        checkpoint = os.path.join(path, 'a3c.pt')
        torch.save(self.state_dict(), checkpoint)

    def load(self, path):
        checkpoint = os.path.join(path, 'a3c.pt')
        self.load_state_dict(torch.load(checkpoint))