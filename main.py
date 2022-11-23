import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any


def MonteCarloReward(rewards: t.Tensor, crit: float, gamma: float = 0.99) -> t.Tensor:

    returns = t.empty_like(rewards, requires_grad=False)

    for i in reversed(range(len(rewards))):
        crit = rewards[i].item() + gamma * crit
        returns[i] = crit
    
    returns = (returns - t.mean(returns)) / t.std(returns) 

    return returns


class ActorCritic(nn.Module):
    def __init__(self, inp_dim : int, out_dim : int) -> None:
        super().__init__()
        # Shared
        self.linear1 = nn.Linear(inp_dim, 200)
        self.linear2 = nn.Linear(200, 100)
        # Divergence 
        self.actor = nn.Linear(100, out_dim)
        self.critic = nn.Linear(100, 1)

    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        z = self.linear1(x)
        z = self.linear2(z)
        return self.actor(z), self.critic(z)


def a2c_train_loop(env : Any, episodes : int, max_steps : int = 1000, print_step : int = 100) -> None:
    mdl = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    critic_loss_func = nn.MSELoss("sum")
    opti = optim.Adam(mdl.parameters())


    for episode in range(episodes):
        state_t = t.from_numpy(env.reset()[0].astype(np.float32, copy=False)) 
        step = 0
        
        rewards = t.empty(max_steps)
        crits = t.empty(max_steps) # Must be here to reset graph
        
        for _ in range(max_steps):
            _, crit_t = mdl(state_t)

            #state, reward, terminated, _, _ = env.step(action.detach().numpy())
            state_tt, reward_tt, terminated, _, _  = env.step(env.action_space.sample())
            state_tt = t.from_numpy(state_tt.astype(np.float32, copy=False))          
            crits[step] = crit_t
            rewards[step] = reward_tt

            # policy update
            # policy loss
            # Update policy

            step += 1 
            state_t = state_tt
            if terminated:
                break

        _, crit_t = mdl(state_t)

        returns = MonteCarloReward(rewards[:step], crit_t.item())
         
        opti.zero_grad()
            
        critic_loss = critic_loss_func(returns, crits[:step])
        a2c_loss = critic_loss # + actor_loss
       
        # Might have to retain graph cuz of shared params
        a2c_loss.backward()
        opti.step()

        #print(f"EPOCH: {epoch + 1}, STEP: {step + 1}, LOSS: {loss.item():.4f}")
        

def main() -> None:
    environment = gym.make("HalfCheetah-v4")

    a2c_train_loop(environment, 10)

if __name__ == "__main__":
    main()
