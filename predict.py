# Standard
import os
from datetime import datetime
import shutil
import subprocess
import argparse
import importlib.util
import logging
import numpy as np
from QuantumStateEnv import QuantumStateEnv
from SAC import SAC_Agent
import torch
import config as cfg
from reward import Rewards
import time
import random

"""
Execution file and main operation loop of the framework.
"""

#Load config
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='The name of the configuration file (e.g. my_config.py).')
parser.add_argument('path', type=str, help='Path to the trained models.')
args = parser.parse_args()

config_path = os.path.join('config', args.config)
model_path = args.path  # <-- hier ist dein zweites Argument

if not os.path.exists(config_path):
    print(f"Configuration file {config_path} does not exist.")
cfg = load_config(config_path)


logger = logging.getLogger('dummy')
logger.addHandler(logging.NullHandler())
main_folder = '.'
# CPU can be used if GPU is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info('CUDA is available. The program is running on GPU.')
else:
    device = torch.device("cpu")
    logger.info('CUDA is not available. The program is running on CPU. This is not recommended.')

#Loop to execute training sessions
env = QuantumStateEnv(logger,cfg,main_folder, pred=True)


max_gates= cfg.training["max_gates"]                      
    
                       
layer_scale=np.round(np.linspace(1/max_gates, 1, max_gates), 3).tolist()

#prediction with step size 0.01 of the interval of interest
bond_distance_range = np.arange(cfg.characteristics["start_bond_distance"], cfg.characteristics["end_bond_distance"], 0.01)  


for session in range(0,cfg.training["runs"]):

    agent = SAC_Agent(cfg, env, device, logger,model_path,session,pred=True)   
    results=[]    
    #Loop to execute a single training session
    for k in range(len(bond_distance_range)):
        
        bond_distance = np.round(bond_distance_range[k],2)
    
        #No further training, direct prediction
        evaluation_episode = True

        #These variables need to be reseted for a new episode
        state,qustate,current_qucircuit,episode_circuit,episode_angles= env.reset(k)
        done=False
        start_episodes=False  
        E_before=cfg.characteristics["initial_energy"]
    
        #LOOP FOR PREDICTING CIRCUIT FOR ONE PARAMETER      
        for i in range(max_gates): 

                
            #chose the next action if evaluation episode deterministically, otherwise non deterministically, and for start steps random
            action,angle_action= agent.get_next_action(state, evaluation_episode, start_episodes)
        
            #track the circuit
            episode_circuit.append(action)
            episode_angles.append(angle_action)
            
                
            next_state,qustate,current_qucircuit=env.step(qustate,action,angle_action,current_qucircuit,i,layer_scale,k)
                
            if i==max_gates-1:
                done=True
                E=env.get_energy(qustate,k)
            
            state = next_state
        
        #Visible output
        print("session:",
            session,
            "parameter:",
            np.round(bond_distance, 2),
            "Predicted energy:",
            np.round(E, 4),
            flush=True)

        #Add result for the episode
        results.append((E, episode_circuit, episode_angles, bond_distance))

    model_path_save = os.path.join(
        model_path,
        f"training_session_{session}")
             
    # Save predictions
    np.save(f"{model_path_save}/unseen_prediction_{session}.npy", results)
