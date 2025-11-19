import numpy as np
import torch

"""
Reward functions
"""

class Rewards():

    def __init__(self,cfg,device):
        #this should be an approximation of the ground state energy
        self.cfg=cfg
        self.sigma_min=self.cfg.reward["sigma_min"]
        self.c_lin=self.cfg.reward["c_lin"]
        self.c_exp=self.cfg.reward["c_exp"]
        self.device= device
        self.E_1= {}
        self.E_2= {}

        for key in range(len(np.arange( self.cfg.characteristics["start_bond_distance"], self.cfg.characteristics["end_bond_distance"], self.cfg.characteristics["step_size_bond_distance"]))):
            self.E_1[key] = [self.cfg.characteristics["initial_energy"]]
            self.E_2[key] = [self.cfg.characteristics["initial_energy"]]

    def which_reward(self,E,E_before,done,index,tensor_form):
        if self.cfg.reward["reward"] == "exp_moving":
            reward = self.exp_moving(E, E_before,done,index, tensor_form)
        return reward

    def dynamic_values_for_reward_update(self,E,index):
        if E < self.E_1[index][np.argmax(self.E_1[index])]:
            if len(self.E_1[index]) == self.cfg.reward["mu_average"]:
                self.E_1[index][np.argmax(self.E_1[index])] = E
            else:
                self.E_1[index].append(E)
        elif E < self.E_2[index][np.argmax(self.E_2[index])]:
            if len(self.E_2[index]) == self.cfg.reward["sigma_average"]:
                self.E_2[index][np.argmax(self.E_2[index])] = E
            else:
                self.E_2[index].append(E)

    def exp_moving(self, E, E_before, done,index,tensor_form):
        #Calculation of the reward to update the neural networks
        if tensor_form:
            mu = torch.tensor([np.mean(self.E_1[i.item()]) for i in index]).to(self.device)
            sigma =torch.abs((mu- torch.tensor([np.mean(self.E_2[i.item()]) for i in index]).to(self.device)))+self.sigma_min
            ex1 = torch.exp(-1*(E - mu) / sigma)
            ex2 = torch.exp(-1*(E_before - mu) / sigma)
            lin = (E - E_before)
        #Calculation of the reward for visible output 
        else:
            mu = np.mean(self.E_1[index]) 
            sigma = np.abs(mu-np.mean(self.E_2[index]))+self.sigma_min
            ex1 = np.exp(-1*(E - mu) / sigma)
            ex2 = np.exp(-1*(E_before - mu) / sigma)
            lin =  (E - E_before)
        return self.c_exp*(ex1-ex2)-self.c_lin*lin
   
    



