import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any


def monte_carlo_reward(rewards: t.Tensor, crit: float, gamma: float = 0.99) -> t.Tensor:

    returns = t.zeros_like(rewards, requires_grad=False)
    
    # Override crit value, not true MC but more stable learning observed
    crit = 0.0
    for i in reversed(range(len(rewards))):
        crit = rewards[i].item() + gamma * crit
        returns[i] = crit
    
    #returns = (returns - t.mean(returns)) / t.std(returns) 

    return returns

class Actor(nn.Module):

    def __init__(self, inp_dim : int, out_dim : int) -> None:
        super().__init__()
        # Shared
        self.linear1 = nn.Linear(inp_dim, 128)
        self.linear2 = nn.Linear(128, 256)

        # Divergence 
        self.actor = nn.Linear(256, out_dim)

    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        k = nn.functional.relu(self.linear1(x))
        k = nn.functional.relu(self.linear2(k))
        return nn.functional.softmax(self.actor(k), dim=0)


class Critic(nn.Module):
    def __init__(self, inp_dim : int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(inp_dim, 128)
        self.linear2 = nn.Linear(128, 256)

        # Divergence 
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        l = nn.functional.relu(self.linear1(x))
        l = nn.functional.relu(self.linear2(l))
        return self.critic(l)



def a2c_train_loop(env : Any, episodes : int, max_steps : int = 1000, print_step : int = 100) -> None:
    actor = Actor(env.observation_space.shape[0], env.action_space.n)
    critic = Critic(env.observation_space.shape[0])
    actor_loss_func = lambda action_probs, returns, crits : -t.mean(t.log(action_probs) * (returns - crits))
    critic_loss_func = nn.MSELoss("mean")

    opti_a = optim.Adam(actor.parameters())
    opti_c = optim.Adam(critic.parameters())

    for episode in range(episodes):
        state_t = t.from_numpy(env.reset()[0].astype(np.float32, copy=False))       

        step = 0
        
        rewards = t.zeros(max_steps)
        crits = t.zeros(max_steps) # Must be here to reset graph
        action_probs = t.empty(max_steps)
        for _ in range(max_steps):
            
            action_prob = actor(state_t)
            crit_t = critic(state_t)
            
            action = t.distributions.categorical.Categorical(probs=action_prob).sample()

            state_tt, reward_tt, terminated, _, _ = env.step(action.item())

            state_tt = t.from_numpy(state_tt.astype(np.float32, copy=False))          
            crits[step] = crit_t
            rewards[step] = reward_tt
            action_probs[step] = action_prob[action]

            step += 1 
            
            state_t = state_tt
            if terminated:
                break

        
        crit_t = critic(state_t)
        
        returns = monte_carlo_reward(rewards[:step], crit_t.item())

        actor_loss = actor_loss_func(action_probs[:step], returns, crits[:step].detach());    
        critic_loss = critic_loss_func(returns, crits[:step])
       
        opti_a.zero_grad()
        opti_c.zero_grad()
    
        critic_loss.backward()
        actor_loss.backward()
        
        opti_a.step()
        opti_c.step()

        if (episode % 100 == 0):
            print(f"EPISODE: {episode}, STEPS: {step}, [{action_prob[0].item():.4f},{action_prob[1].item():.4f}], CRIT LOSS: {critic_loss.item():.4f}, ACT LOSS: {actor_loss.item():.4f}")
    
def main() -> None:
    environment = gym.make("CartPole-v1")

    a2c_train_loop(environment, 3000)

    environment.close()

if __name__ == "__main__":
    main()
