import patch
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
parser.add_argument('config', type=str, help='The path to the configuration file.')
args = parser.parse_args()
config_path = os.path.join('config', args.config)
if not os.path.exists(config_path):
    print(f"Configuration file {config_path} does not exist.")
cfg = load_config(config_path)

# Create folder to save the results
results_folder = cfg.training["results_folder_name"]   
timestamp = datetime.now().isoformat()
main_folder = os.path.join(results_folder, f"training_{timestamp}")
if not os.path.exists(results_folder):
    os.makedirs(results_folder,exist_ok=True)
if not os.path.exists(main_folder):
    os.makedirs(main_folder,exist_ok=True)

# Save configuration
text_file = os.path.join(main_folder, "config_info.txt")
shutil.copy2(config_path , text_file)

#Set up Logger
log_file = 'Info.log'
log_path = os.path.join(main_folder, log_file)
logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Info')

# CPU can be used if GPU is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info('CUDA is available. The program is running on GPU.')
else:
    device = torch.device("cpu")
    logger.info('CUDA is not available. The program is running on CPU. This is not recommended.')

#Loop to execute training sessions
env = QuantumStateEnv(logger,cfg,main_folder,pred=False)

for x in range(cfg.training["runs"]):

    logger.info("New training session is starting.")

    rwd = Rewards(cfg,device)
    agent = SAC_Agent(cfg, env, device, logger,model_path=None,session=None, pred=False)   

    #Get parameters from config file
    max_episodes = cfg.training["max_episodes"]               
    max_gates= cfg.training["max_gates"]                      
    train_evaluate_ratio = cfg.training["train_evaluate_ratio"]    
    random_episodes=cfg.training["random_episodes"]      
    results=[]                          
    layer_scale=np.round(np.linspace(1/max_gates, 1, max_gates), 3).tolist()
    decay=(max_episodes*max_gates)
    bond_distance_range = np.arange(cfg.characteristics["start_bond_distance"], cfg.characteristics["end_bond_distance"], cfg.characteristics["step_size_bond_distance"])  

    
    #Loop to execute a single training session
    for e in range(max_episodes):

        #For each episode, the bond distance is randomly sampled
        index = random.randint(0, len(bond_distance_range)-1)
        bond_distance = np.round(bond_distance_range[index],2)
  
        #Is this episode an evaluation episode? (In eval episodes the action is chosen according to the highest probability)
        evaluation_episode = e % train_evaluate_ratio == 0

        #These variables need to be reseted for a new episode
        state,qustate,current_qucircuit,episode_circuit,episode_angles= env.reset(index)
        done=False
        start_episodes=False  
        E_before=cfg.characteristics["initial_energy"]
      
        #The first actions are choosen randomly:
        if e<random_episodes:
            start_episodes=True
            evaluation_episode=False
        
        #LOOP FOR ONE EPISODE, i.e CONSTRUCTING ONE CIRCUIT       
        for i in range(max_gates): 

            #function for updating the target entropy  
            step=((e*max_gates)+i)
            agent.update_target_entropy(step,decay)
            
            #chose the next action if evaluation episode deterministically, otherwise non deterministically, and for start steps random
            action,angle_action= agent.get_next_action(state, evaluation_episode, start_episodes)
     
            #track the circuit
            episode_circuit.append(action)
            episode_angles.append(angle_action)
        
            
            next_state,qustate,current_qucircuit=env.step(qustate,action,angle_action,current_qucircuit,i,layer_scale,index)
        
            E=env.get_energy(qustate,index)
            
            #Update values for dynamical reward
            rwd.dynamic_values_for_reward_update(E,index)
            agent.dynamic_values_for_reward_update(E,index)
            
            if i==max_gates-1:
                done=True
                #Calculate Return
                E_before=cfg.characteristics["initial_energy"]
                episode_return=rwd.which_reward(E,E_before,done,index,tensor_form=False)
                
            #Save transition
            if not evaluation_episode:
                agent.train_on_transition(state, action, angle_action, next_state, E, E_before,  done, e,index)
                    
            
            E_before=E 
            state = next_state
    
        
        #Visible output
        print(
            "Training session:",
            x+1,"/",cfg.training["runs"],
            "episode:",
            e,"/", max_episodes,
            "energy:",
            np.round(E, 4),
            "return:",
            np.round(episode_return, 4),
            "bd:",
            np.round(bond_distance, 2),
            flush=True)

        #Add result for the episode
        results.append((e, E, evaluation_episode, episode_return, episode_circuit, episode_angles, bond_distance))

    # Save all quantities of one training session
    session_folder = os.path.join(main_folder, f"training_session_{x}")
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    rl_quantities = agent.rl_quantities()
    agent.save_policy(x, session_folder)
    results_file = os.path.join(session_folder, f"results_{x}.npy")
    rl_quantities_file = os.path.join(session_folder, f"rl_quantities_{x}.npy")
    np.save(results_file, np.array(results, dtype=object))
    np.save(rl_quantities_file, np.array(rl_quantities,dtype=object))
