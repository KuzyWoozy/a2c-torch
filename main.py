import time
import gym
import numpy as np
import torch as t
from torch import nn
from torch import optim


t.backends.cuda.matmul.allow_tf32 = True
t.set_default_dtype(t.float32)

ANT_SPEED = 0.2

LEARNING_RATE = 3e-3 
EPISODES = 200000
STEPS = 500
ENTROPY = 0.0001

CPU = t.device("cpu")
GPU = t.device("cuda" if t.cuda.is_available() else "cpu")


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
       
        action_logProb = -((action - mean) ** 2) / (2 * std ** 2) - t.log(std) - t.log(t.sqrt(t.tensor([2 * t.pi], device=GPU)))

        return action, action_logProb, mean, std


    def forward(self, x):
        
        z = nn.functional.relu(self.linear1(x))
        z = nn.functional.relu(self.linear2(z))
        
        action, action_logProb, mean, std = self.gauss(z)

        return action, action_logProb, mean, std, self.critic(z)


def a2c_train_loop(mdl, env):
    
    def mcr(rewards, gamma = 0.99):
        returns = t.zeros_like(rewards, device=GPU)
        
        ret = 0
        for i in reversed(range(len(rewards))):
            returns[i] = ret = rewards[i].item() + gamma * ret
        
        returns = (returns - t.mean(returns)) / t.std(returns)

        return returns


    opti = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)


    # Ensure correct broadcasting 
    rewards = t.zeros(STEPS, 1, device=GPU, requires_grad=False)
    crits = t.zeros(STEPS, 1, device=GPU)
    action_logProbs = t.zeros(STEPS, mdl.out_dim, device=GPU)
    stds = t.zeros(STEPS, mdl.out_dim, device=GPU)
   
    for episode in range(0, EPISODES):
            
        state_t = t.from_numpy(env.reset()[0]).float()
        state_t = state_t.to(GPU)    

        for step in range(0, STEPS):
            
            action, action_logProb, mean, std, crit_t = mdl(state_t)    
            action = action.to(CPU)

            state_tt, reward_tt, terminated, truncated, info = env.step(action.numpy())
            state_t = t.from_numpy(state_tt).float().to(GPU)
            
            # Undo forward reward
            forward_reward = info["reward_forward"]
            reward_tt -= forward_reward
            # Override reward, encourage walking at same speed
            reward_tt += ANT_SPEED - forward_reward if forward_reward > ANT_SPEED else forward_reward 
            
            crits[step] = crit_t
            rewards[step] = reward_tt
            action_logProbs[step] = action_logProb
            stds[step] = std

            if terminated:
                break
       
        rewards_v = rewards[:step + 1] 
        crits_v = crits[:step + 1]
        action_logProbs_v = action_logProbs[:step + 1] 
        stds_v = stds[:step + 1]

        returns = mcr(rewards_v)
        advantage = returns - crits_v
        actor_loss = t.sum(-action_logProbs_v * advantage.detach()) 
        critic_loss = t.sum(advantage ** 2)
        entropy_loss = t.sum(-ENTROPY * ((0.5 * t.log(2 * t.pi * stds_v ** 2)) + 0.5))

        a2c_loss = critic_loss + actor_loss + entropy_loss

        opti.zero_grad()
        a2c_loss.backward()
        opti.step()
        
        # Doesn't matter that we're detaching the view of the
        # original array we don't interact with the other elements
        crits = crits.detach()
        action_logProbs = action_logProbs.detach()
        stds = stds.detach()

        
        if (episode % 100 == 0):
            print(f"EPISODE: {episode}, STEPS: {step}\n{action.detach().numpy()}\nCRIT LOSS: {critic_loss.item():.4f}, ACT LOSS: {actor_loss.item():.4f}, ENTROPY LOSS: {entropy_loss:.4f}\nMEAN MAX: {mean[t.argmax(t.abs(mean))]}, MEAN AVG: {t.mean(mean)}, STD MAX: {t.max(std)}, STD AVG: {t.mean(std)}\n", flush=True) 



def main() -> None:
    environment = gym.make("Ant-v4")
    
    model = ActorCritic(environment.observation_space.shape[0], 100, environment.action_space.shape[0])
    model = model.to(GPU)
    
    start = time.time_ns()
    a2c_train_loop(model, environment)
    end = time.time_ns()

    print(f"Time taken: {(end - start) / 1e9}s")
    
    t.save(model, f"model-{round(time.time())}.torch")
    #model = t.load("model.torch")

    environment.close()

if __name__ == "__main__":
    main()
