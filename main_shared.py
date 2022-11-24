import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any


def monte_carlo_reward(rewards: t.Tensor, crit: float, gamma: float = 0.99) -> t.Tensor:

    returns = t.zeros_like(rewards, requires_grad=False)

    for i in reversed(range(len(rewards))):
        crit = rewards[i].item() + gamma * crit
        returns[i] = crit
    
    #returns = (returns - t.mean(returns)) / t.std(returns) 

    return returns


class ActorCritic(nn.Module):
    def __init__(self, inp_dim : int, out_dim : int) -> None:
        super().__init__()
        # Shared
        self.linear1_crit = nn.Linear(inp_dim, 128)
        self.linear2_crit = nn.Linear(128, 256)

        self.linear1_act = nn.Linear(inp_dim, 128)
        self.linear2_act = nn.Linear(128, 256)

        # Divergence 
        self.actor = nn.Linear(256, out_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        k = nn.functional.relu(self.linear1_act(x))
        k = nn.functional.relu(self.linear2_act(k))

        l = nn.functional.relu(self.linear1_crit(x))
        l = nn.functional.relu(self.linear2_crit(l))
        return nn.functional.softmax(self.actor(k), dim=0), self.critic(l)


def a2c_train_loop(env : Any, episodes : int, max_steps : int = 1000, print_step : int = 100) -> None:
    mdl = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    actor_loss_func = lambda action_probs, returns, crits : -t.mean(t.log(action_probs) * (returns - crits))
    critic_loss_func = nn.MSELoss("mean")
    opti = optim.Adam(mdl.parameters(), lr=0.0001)
    
    for episode in range(episodes):
        state_t = t.from_numpy(env.reset()[0].astype(np.float32, copy=False))       

        step = 0
        
        rewards = t.zeros(max_steps)
        crits = t.zeros(max_steps) # Must be here to reset graph
        action_probs = t.empty(max_steps)
        for _ in range(max_steps):
            
            action_prob, crit_t = mdl(state_t)
            
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

        action_prob, crit_t = mdl(state_t)
        
        returns = monte_carlo_reward(rewards[:step], crit_t.item())
       
        actor_loss = actor_loss_func(action_probs[:step], returns, crits[:step].detach());    
        critic_loss = critic_loss_func(returns, crits[:step])
        a2c_loss = critic_loss + actor_loss
       
        opti.zero_grad()
        
        a2c_loss.backward()
        
        opti.step()
        
        if (episode % 1 == 0):
            print(f"EPISODE: {episode + 1}, STEPS: {step + 1}, [{action_prob[0].item():.2f},{action_prob[1].item():.2f}], CRIT LOSS: {critic_loss.item():.4f}, ACT LOSS: {actor_loss.item():.4f}")
    
def main() -> None:
    environment = gym.make("CartPole-v1")

    a2c_train_loop(environment, 500)

    environment.close()

if __name__ == "__main__":
    main()
