import numpy as np
import torch

"""
File for initializing the replay buffer, saving transitions and sampling batches from the replay buffer.
"""

class ReplayBuffer():


    def __init__(self,cfg,device,env):
        #INITIALIZATION OF AN EMPTY REPLAY BUFFER
        self.device=device
        self.env=env
        self.cfg=cfg
        self.capacity = self.cfg.RBparam["capacity"]  
        self.state = torch.zeros((self.capacity, self.env.state_size), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((self.capacity, 1), dtype=torch.int64, device=self.device)
        self.action_c = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.next_state = torch.zeros((self.capacity, self.env.state_size), dtype=torch.float32, device=self.device)
        self.energy = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.energy_before = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((self.capacity, 1), dtype=torch.bool, device=self.device)
        self.index = torch.zeros((self.capacity, 1),  dtype=torch.int64, device=self.device)
        self.next_index = 0
        self.count = int(0)
        self.indices = None
       
    def add_transition(self, state,action,action_c,next_state,energy,energy_before,done,index):
        #add transition
        self.state[self.next_index] = state
        self.action[self.next_index] = action
        self.action_c[self.next_index] = action_c
        self.next_state[self.next_index] = next_state
        self.energy[self.next_index] = energy
        self.energy_before[self.next_index] = energy_before
        self.done[self.next_index] = done
        self.index[self.next_index] = index
        #If the replay buffer is full the oldest transition is replaced by the newest added transition
        self.next_index = (self.next_index + 1) % self.capacity
        self.count = min(int(self.count+ 1), self.capacity)
        if self.count<1:
            self.count=int(self.count[0])

    def sample_batch(self,batch_size):
        #sample indices for batch
        self.indices = torch.randint(0, self.count, size=(batch_size,), device=self.device)
        return self.state[self.indices],self.action[self.indices].squeeze(-1),self.action_c[self.indices].squeeze(-1),self.next_state[self.indices],self.energy[self.indices].squeeze(-1),self.energy_before[self.indices].squeeze(-1),self.done[self.indices].squeeze(-1),self.index[self.indices].squeeze(-1)

    def get_size(self):
        return self.count
    
