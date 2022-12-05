import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any

class ActorCritic(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Shared
        self.linear1 = nn.Linear(inp_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Divergence 
        self.actor_mean = nn.Linear(hidden_dim, out_dim)
        self.actor_std = nn.Linear(hidden_dim, out_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def gauss(self, z):
        
        mean, std = t.tanh(self.actor_mean(z)), t.nn.functional.softplus(self.actor_std(z))
        
        dist = t.distributions.normal.Normal(t.zeros(self.out_dim), t.ones(self.out_dim))

        action = (mean + std * dist.sample()).detach()

        action_logProb = -((action - mean) ** 2) / (2 * std ** 2) - t.log(std) - t.log(t.sqrt(t.tensor([2 * t.pi])))

        return action, action_logProb, mean, std


    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        z = nn.functional.relu(self.linear1(x))
        z = nn.functional.relu(self.linear2(z))
        
        action, action_logProb, mean, std = self.gauss(z)

        return action, action_logProb, mean, std, self.critic(z)


def play_loop(mdl, env):
    
    while True:
            
        state_t = t.from_numpy(env.reset()[0]).float()
        
        for _ in range(500):
            
            action, action_logProb, mean, std, crit_t = mdl(state_t)
            
            state_tt, reward_tt, terminated, truncated, info = env.step(t.clip(action, -1, 1).numpy())
            
            state_t = t.from_numpy(state_tt).float()
            
            if terminated:
                break
             

def main() -> None:
    environment = gym.make("Ant-v4", render_mode = "human")
    
    model = t.load("model_50k.torch")

    play_loop(model, environment)

    environment.close()

if __name__ == "__main__":
    main()
