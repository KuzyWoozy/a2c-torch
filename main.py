import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim
from typing import Any


def monte_carlo_reward(rewards: t.Tensor, crit: float, gamma: float = 0.99) -> t.Tensor:
    returns = t.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        crit = rewards[i].item() + gamma * crit
        returns[i] = crit
    return returns


class ActorCritic(nn.Module):
    def __init__(self, inp_dim : int, out_dim : int) -> None:
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Shared
        self.linear1 = nn.Linear(inp_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        # Divergence 
        self.actor = nn.Linear(256, out_dim)
        self.critic = nn.Linear(256, 1)


    def forward(self, x : t.Tensor) -> tuple[t.Tensor, t.Tensor]:        
        self.z = nn.functional.relu(self.linear1(x))
        self.z = nn.functional.relu(self.linear2(self.z))
        
        return nn.functional.log_softmax(self.actor(self.z), dim=0), self.critic(self.z)


def a2c_train_loop(mdl : nn.Module, env : Any, episodes : int, max_steps : int = 1000) -> None:

    actor_loss_func = lambda action_logProbs, returns, crits : -t.mean(action_logProbs * (returns - crits))
    critic_loss_func = nn.MSELoss("mean")

    opti = optim.Adam(mdl.parameters(), lr=0.00001)
   
    for episode in range(episodes):
            
        state_t, info = env.reset(seed=0)
        state_t = t.tensor(list(env.decode(state_t))).float()
        
        step = 0
        
        rewards = t.zeros(max_steps)
        crits = t.zeros(max_steps) # Must be here to reset graph
        action_logProbs = t.empty(max_steps)
        
        for _ in range(max_steps):
            in_taxi = state_t[2] == 4

            state_t[0] /= 5
            state_t[1] /= 5
            state_t[2] /= 5
            state_t[3] /= 4
            
            action_logProb, crit_t = mdl(state_t)
            
            # Explore epsilon
            if t.rand(1).item() < min(0.90, episode / (episodes / 2)):
                action = t.distributions.categorical.Categorical(logits=action_logProb).sample()
            else:
                action = t.distributions.categorical.Categorical(probs=t.ones(1, mdl.out_dim)).sample()
            
            state_tt, reward_tt, terminated, _, info = env.step(action.item())
            
            state_tt = t.tensor(list(env.decode(state_tt))).float()
            
            if (t.equal(state_tt, state_t)): 
                reward_tt = -5
            
            if (not in_taxi and state_tt[2] == 4):
                reward_tt = 100
           
            reward_tt /= 100

            crits[step] = crit_t
            rewards[step] = reward_tt
            action_logProbs[step] = action_logProb[action]

            step += 1 
            state_t = state_tt

            if terminated:
                break
            
            if (not in_taxi and state_tt[2] == 4):
                break

        _, crit_t = mdl(state_t)

        returns = monte_carlo_reward(rewards[:step], crit_t.item())
        actor_loss = actor_loss_func(action_logProbs[:step], returns, crits[:step].detach());    
        critic_loss = critic_loss_func(returns, crits[:step])
        a2c_loss = critic_loss + actor_loss
       
        opti.zero_grad()
        
        a2c_loss.backward()
        
        if (episode % 10 == 0):
            param_sum = 0
            param_cnt = 0
            for param in mdl.parameters():
                param_sum += t.sum(param.data)
                param_cnt += t.prod(t.tensor([param.shape]))
            
            print(env.render())
            print(f"EPISODE: {episode}, STEPS: {step}, {[x.item() for x in action_logProb]}, CRIT LOSS: {critic_loss.item():.4f}, ACT LOSS: {actor_loss.item():.4f}, PARAM AVG: {param_sum / param_cnt}")

        opti.step()

def play_game(mdl, env):
    
    while(True):
        step = 0
        
        state_t, info = env.reset(seed=0)
        #state_t = nn.functional.one_hot(t.tensor([state_t]), num_classes=env.observation_space.n)[0].float()
        state_t = t.tensor(list(env.decode(state_t))).float()


        while(True):
            state_t[0] /= 5
            state_t[1] /= 5
            state_t[2] /= 5
            state_t[3] /= 4
            
            action_logProb, _ = mdl(state_t) 

            action = t.distributions.categorical.Categorical(logits=action_logProb).sample()

            
            state_t, reward_tt, terminated, _, _ = env.step(action.item())
            state_t = t.tensor(list(env.decode(state_t))).float()


            step += 1
            print(step, action_logProb, t.argmax(action_logProb))

            if terminated:
                break


def main() -> None:
    environment = gym.make("Taxi-v3", render_mode="ansi")
    
    model = ActorCritic(4, environment.action_space.n)
    
    a2c_train_loop(model, environment, 2000, 500)
    t.save(model, "model.torch")

    #model = t.load("model.torch")

    environment = gym.make("Taxi-v3", render_mode="human")

    play_game(model, environment)

    environment.close()

if __name__ == "__main__":
    main()
