import numpy as np
import torch


class Data(object):
    def __init__(self, s_dim, a_dim, max_size):
        self.max_size = max_size
        #information about what is currently being stored in data object
        self.pos = 0
        self.num_data = 0

        #setting up arrays to store the data for each state
        self.states = np.zeros((max_size, s_dim))
        self.actions = np.zeros((max_size, a_dim))
        self.next_state = np.zeros((max_size, s_dim))
        self.rewards = np.zeros((max_size, 1))
        self.is_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        #adding new data to arrays
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.next_state[self.pos] = next_state
        self.rewards[self.pos] = reward
        self.is_done[self.pos] = done
        
        #updating information about state of data object
        self.pos += 1
        self.num_data += 1


    def sample(self, batch_size):
        #getting list of random elements to access from each array
        rands = np.random.randint(0, self.num_data, size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        #getting elements from arrays at every randomly generated position
        for i in rands:
            state.append(self.states[i])
            action.append(self.actions[i])
            reward.append(self.rewards[i])
            next_state.append(self.next_state[i])
            done.append(self.is_done[i])

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)