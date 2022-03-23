import numpy as np
import torch
import gym
import os

import TD3
import utils

if __name__ == "__main__":
    
    #whether or not to render the environment
    render = True
    #maximum number of episodes
    max_episodes = 25
    #maximum number of timesteps in each episode
    max_timesteps = 1000000
    #name of model to load (None if no model)
    load = "third_training_attempt/train_610_episodes"

    #setting up environment and getting environment information
    env = gym.make("BipedalWalker-v3")
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0] 
    max_actions = float(env.action_space.high[0])
    
    #creating new policy or loading a policy if provided
    policy = TD3.TD3(s_dim, a_dim, max_actions)
    
    if load != None:
        policy.load_actor(f"./models/{load}")
 
    
    rewards = []

    for e in range(max_episodes):
        state = env.reset()
        e_reward = 0
    
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            e_reward += reward
            env.render()
            if done:
                break
        print('\rEpisode: {}/{},\tScore: {:.2f}'.format(e, max_episodes, e_reward))
        rewards.append(e_reward)
        

    env.close()
    rewards = np.array(rewards)
    print('average reward over {} episodes is: {}'.format(max_episodes, np.mean(rewards, dtype=np.float64)))


    
    
  