import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        #creating layers for feed forward NN for actor
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        #activation function for hidden layer
        self.hidden_act = nn.ReLU()
        
        #activation function for output layer
        self.output_act = torch.tanh

        self.max_action = max_action


    def forward(self, state):
        #applying the linears and activation functions for actor model
        layer_1 = self.l1(state)
        act_1 = self.hidden_act(layer_1)
        layer_2 = self.l2(act_1)
        act_2 = self.hidden_act(layer_2)
        layer_3 = self.l3(act_2)
        act_3 = self.output_act(layer_3)
        return self.max_action * act_3


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        #critic NN
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.q1_act = nn.ReLU()


    def forward(self, state, action):
        state_and_action = torch.cat([state, action], 1)
        
        #applying the linears and activation functions for critic model
        layer_1 = self.l1(state_and_action)
        act_1 = self.q1_act(layer_1)
        layer_2 = self.l2(act_1)
        act_2 = self.q1_act(layer_2)
        layer_3 = self.l3(act_2)
        
        return layer_3
