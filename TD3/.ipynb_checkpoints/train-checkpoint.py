import numpy as np
import pandas as pd
import torch
import gym
import os

import TD3
import utils


#maximum number of episodes
max_episodes = 10
#maximum number of timesteps in each episode
max_timesteps = 2000
#how often to evaluate the policy
eval_freq = 10
#exploration noise
exp_noise = 0.1
#whether or not to save this model
save = True
save_file = "third_training_attempt/train_610_episodes"
#name of model to load (None if no model)
load = "third_training_attempt/train_600_episodes"

#setting up environment and getting environment information
env = gym.make("BipedalWalker-v3")

#getting dimensions/max action for the specific environment
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

data = utils.Data(s_dim, a_dim, int(max_action))

#creating new policy or loading a policy if provided
policy = TD3.TD3(s_dim, a_dim, max_action)
    
if load != None:
    policy.load(f"./models/{load}")

#variables containing the results
scores = []
mean_scores = []


#training
for e in range(max_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        #select action
        action = policy.select_action(state)
        #add exploration noise
        action = action + np.random.normal(0, exp_noise, size=a_dim)
        #make sure action is within range of allowed actions
        action = action.clip(-max_action, max_action)

        # take action in env:
        next_state, reward, done, info = env.step(action)
        #add new data
        data.add(state, action, next_state, reward, done)
        state = next_state
        
        total_reward += reward

        #if episode is done then re-train (updating policy)
        if done or t == (max_timesteps - 1):
            policy.train(data, t)
            break
    
    #updating and printing results
    scores.append(total_reward)
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)
    print('\rEpisode: {}/{},\tScore: {:.2f}'.format(e, max_episodes, total_reward))
    
    if (e + 1) % eval_freq == 0:
        if save: 
            policy.save(f"./models/{save_file}")

        

        
#saving results
df = pd.DataFrame(scores)
df.to_csv(f"./results/{save_file}_results.csv")
env.close()


















