import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any

LEARNING_RATE = 3e-3 
EPISODES = 10000
STEPS = 500
ENTROPY = 0.0001


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
        
        action = t.clip(t.distributions.normal.Normal(mean.detach(), std.detach()).sample(), -1, 1)

        action_logProb = -((action - mean) ** 2) / (2 * std ** 2) - t.log(std) - t.log(t.sqrt(t.tensor([2 * t.pi])))

        return action, action_logProb, mean, std


    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        z = nn.functional.relu(self.linear1(x))
        z = nn.functional.relu(self.linear2(z))
        
        action, action_logProb, mean, std = self.gauss(z)

        return action, action_logProb, mean, std, self.critic(z)


def a2c_train_loop(mdl, env):
    
    def mcr(rewards: t.Tensor, gamma = 0.99):
        returns = t.zeros_like(rewards)
        
        ret = 0
        for i in reversed(range(len(rewards))):
            returns[i] = ret = rewards[i].item() + gamma * ret
        
        returns = (returns - t.mean(returns)) / t.std(returns)

        return returns


    opti = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)
   
    for episode in range(1, EPISODES + 1):
            
        state_t = t.from_numpy(env.reset()[0]).float()
        
        rewards = []
        crits = []
        action_logProbs = []
        stds = []

        for step in range(1, STEPS + 1):
            
            action, action_logProb, mean, std, crit_t = mdl(state_t)
            
            state_tt, reward_tt, terminated, truncated, info = env.step(action.detach().numpy())
            
            crits.append(crit_t)
            rewards.append(reward_tt)
            action_logProbs.append(action_logProb)
            stds.append(std)

            state_t = t.from_numpy(state_tt).float()
            
            if terminated:
                break
             
        rewards = t.tensor(rewards).unsqueeze(1)
        crits = t.stack(crits)
        action_logProbs = t.stack(action_logProbs)
        stds = t.stack(stds)
            
        

        returns = mcr(rewards)
        advantage = returns - crits

        actor_loss = t.sum(-action_logProbs * advantage.detach()) 
        critic_loss = t.sum(advantage ** 2)
        entropy_loss = t.sum(-ENTROPY * ((0.5 * t.log(2 * t.pi * stds ** 2)) + 0.5))

        a2c_loss = critic_loss + actor_loss + entropy_loss
       
        opti.zero_grad()
        a2c_loss.backward()
        opti.step()
        
        if (episode % 1 == 0):
            print(f"EPISODE: {episode}, STEPS: {step}\n{action.detach().numpy()}\nCRIT LOSS: {critic_loss.item():.4f}, ACT LOSS: {actor_loss.item():.4f}, ENTROPY LOSS: {entropy_loss:.4f}\nMEAN MAX: {mean[t.argmax(t.abs(mean))]}, MEAN AVG: {t.mean(mean)}, STD MAX: {t.max(std)}, STD AVG: {t.mean(std)}\n", flush=True) 



def main() -> None:
    environment = gym.make("Ant-v4")
    
    model = ActorCritic(environment.observation_space.shape[0], 100, environment.action_space.shape[0])
    
    a2c_train_loop(model, environment)
    
    t.save(model, "model.torch")
    #model = t.load("model.torch")

    environment.close()

if __name__ == "__main__":
    main()
