import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actor_critic import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        
        #setting up actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_goal = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_goal.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        #setting up critic 1
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_goal = Critic(state_dim, action_dim).to(device)
        self.critic_1_goal.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        
        #setting up critic 2
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_goal = Critic(state_dim, action_dim).to(device)
        self.critic_2_goal.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        #maximum action walker can take
        self.max_action = max_action
        
        #episodes are short and I want to prioritize getting the walker to stand up/walk at the beginning
        self.discount = 0.99 
        
        #using paramaters found in TD3 paper in references
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2


    def select_action(self, state):
        '''determing action given current state'''
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, data, timesteps):
        '''training models (actor and critic)'''
        
        batch_size = 100 
                
        for i in range(timesteps):
            #sample a batch of transitions from data
            state, action, reward, next_state, done = data.sample(batch_size)
            
            #convert sample into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            #calculating noise for next action
            noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            #determining next action accoring to actor goal policy
            next_action = (self.actor_goal(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            #compute goal Q-values
            goal_q1 = self.critic_1_goal(next_state, next_action)
            goal_q2 = self.critic_2_goal(next_state, next_action)
            goal_q = torch.min(goal_q1, goal_q2)
            goal_q = reward + ((1-done) * self.discount * goal_q).detach()
            
            #backward step on critic 1 to optimize it
            current_q1 = self.critic_1(state, action)
            loss_q1 = F.mse_loss(current_q1, goal_q)
            self.critic_1_optimizer.zero_grad()
            loss_q1.backward()
            self.critic_1_optimizer.step()
            
            #backward step on critic 2 to optimize it
            current_q2 = self.critic_2(state, action)
            loss_q2 = F.mse_loss(current_q2, goal_q)
            self.critic_2_optimizer.zero_grad()
            loss_q2.backward()
            self.critic_2_optimizer.step()
            
            #delayed policy updates
            if i % 2 == 0:
                #compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                #backwards step on actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                #polyak averaging update (updating frozen goal models)
                for param, goal_param in zip(self.actor.parameters(), self.actor_goal.parameters()):
                    goal_param.data.copy_( ((1-self.tau) * goal_param.data) + (self.tau * param.data))
                
                for param, goal_param in zip(self.critic_1.parameters(), self.critic_1_goal.parameters()):
                    goal_param.data.copy_( ((1-self.tau) * goal_param.data) + (self.tau * param.data))
                
                for param, goal_param in zip(self.critic_2.parameters(), self.critic_2_goal.parameters()):
                    goal_param.data.copy_( ((1-self.tau) * goal_param.data) + (self.tau * param.data))

                    
    def save(self, filename):
        '''saving the model'''
        #saving critic 1 NN
        torch.save(self.critic_1.state_dict(), filename + "_critic_1.pth")
        torch.save(self.critic_1_goal.state_dict(), filename + "_critic_1_goal.pth")
        
        #saving critic 2 NN
        torch.save(self.critic_2.state_dict(), filename + "_critic_2.pth")
        torch.save(self.critic_2_goal.state_dict(), filename + "_critic_2_goal.pth")
        
        #saving actor NN
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_goal.state_dict(), filename + "_actor_goal.pth")


    def load(self, filename):
        '''loading the model for training (training.py)'''
        
        #loading critic 1 NN
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1.pth"))
        self.critic_1_goal.load_state_dict(torch.load(filename + "_critic_1_goal.pth"))
        
        #loading critic 2 NN
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2.pth"))
        self.critic_2_goal.load_state_dict(torch.load(filename + "_critic_2_goal.pth"))

        #loading actor NN
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_goal.load_state_dict(torch.load(filename + "_actor_goal.pth"))
        
        
    def load_actor(self, path):
        '''
        loading model for display purposes (main.py), only need to load actor because this loading 
        method isn't used for training
        '''
        
        #loading actor NN
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))       
        self.actor_goal.load_state_dict(torch.load(path + "_actor_goal.pth"))

        
        
        

