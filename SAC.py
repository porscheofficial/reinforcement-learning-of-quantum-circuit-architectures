import numpy as np
import torch
import torch.nn.functional as F
from NeuralNet import Critic
from NeuralNet import Actor
import os
from ReplayBuffer import ReplayBuffer
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import copy
import time
#Use tensorboard 

from reward import Rewards


"""
Executing updates of the critic and actor networks and the temperature tuning according to SAC.
"""

class SAC_Agent():
    def __init__(self,cfg, env, device, logger,model_path,session,pred):

        #definition of the Rl quantities
        self.pred=pred
        self.model_path=model_path
        self.rwd= Rewards(cfg,device)
        self.device=device
        self.cfg=cfg
        self.environment=env
        self.replay_buffer = ReplayBuffer(self.cfg,self.device,self.environment)
        self.batch_size= self.cfg.SACparam["batch_size"]    
        self.gamma = self.cfg.SACparam["gamma"]
        self.lr_critic = self.cfg.SACparam["lr_critic"]
        self.lr_actor= self.cfg.SACparam["lr_actor"]
        self.lr_alpha_d=self.cfg.SACparam["lr_alpha_d"]
        self.lr_alpha_c=self.cfg.SACparam["lr_alpha_c"]
        self.soft_update_factor= self.cfg.SACparam["soft_update_factor"]
        self.training_update_factor=self.cfg.SACparam["training_update_factor"]
        self.target_entropy_deduction_d=self.cfg.SACparam["target_entropy_deduction_d"]
        self.target_entropy_deduction_c= self.cfg.SACparam["target_entropy_deduction_c"]
        self.target_entropy_end_value_d=self.cfg.SACparam["target_entropy_end_value_d"]
        self.target_entropy_end_value_c=self.cfg.SACparam["target_entropy_end_value_c"]
        self.max_gates= self.cfg.training["max_gates"]   
        self.train_evaluate_ratio = self.cfg.training["train_evaluate_ratio"] 
        self.environment = env
        self.state_dim = env.state_size
        self.action_dim = env.action_size
        self.decay_factor_c=self.cfg.SACparam["decay_factor_c"] 
        self.decay_factor_d=self.cfg.SACparam["decay_factor_d"] 
        self.training_count=0
        self.act_lower_bounds=-np.pi
        self.act_upper_bounds=np.pi
        self.number_tensor=torch.tensor([list(range(0,self.action_dim))]).repeat_interleave(self.batch_size,-2).view(self.batch_size*self.action_dim,1).to(self.device)
        
        #######-----NETWORKS AND OPTIMIZER INITIALIZATION----########
        
        #Two critic networks for (clipped) double Q learning, the +1 in the input dimension is for the angle that is concatenated to the state
        self.critic1 = Critic(self.cfg,input_dimension= self.state_dim+1 ,output_dimension=self.action_dim).to(self.device)
        self.critic2 = Critic(self.cfg,input_dimension= self.state_dim+1 ,output_dimension=self.action_dim).to(self.device)
     
        self.critic_optimiser1 = torch.optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic_optimiser2 = torch.optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
    
        #target networks are updated more slowly than the normal critic network to increase stability (as unhelpful changes will take longer to be recognizable in the target net)
        self.critic_target1 = Critic(self.cfg,input_dimension= self.state_dim+1 ,output_dimension=self.action_dim).to(self.device)
        self.critic_target2 = Critic(self.cfg,input_dimension= self.state_dim+1 ,output_dimension=self.action_dim).to(self.device)
    

        #initialize policy network
        self.actor =Actor(self.cfg,input_dimension=self.state_dim,output_dimension=self.action_dim,output_activation=torch.nn.Softmax(dim=1)).to(self.device)
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        if self.pred:
            model_path_actor = os.path.join(
                self.model_path,
                f"training_session_{session}",
                f"actor_model_{session}.pt"
            )
            self.actor.load_state_dict(torch.load(model_path_actor, map_location=self.device))
        
    
        ################################################################
       
        #######-------Initialize temperature tuning------###############

        #DISCRETE ALPHA (log to make sure final alpha is non negative)
        self.log_alpha_d = torch.tensor([0.]).to(self.device)
        self.log_alpha_d.requires_grad_(True)
        self.log_alpha_optimiser_d = torch.optim.SGD([self.log_alpha_d], lr=self.lr_alpha_d)
        self.alpha_d=self.log_alpha_d.exp()
       
        #CONTINUOUS ALPHA
        self.log_alpha_c = torch.tensor([0.]).to(self.device)
        self.log_alpha_c.requires_grad_(True)
        self.log_alpha_optimiser_c = torch.optim.SGD([self.log_alpha_c], lr=self.lr_alpha_c)
        self.alpha_c=self.log_alpha_c.exp()
        
        ################################################################

        #Tracking of the RL quantities
        self.rl_results=[]
        
    
    ################---------------CHOOSE ACTIONS--------------------##############
    def get_next_action(self, state, evaluation_episode=False, start_episodes=False):
        """
        Method to get the next action from the policy network.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma, action_probabilities = self.actor(state)
        
        #NON DETERMINISTICALLY during training: discrete action is sampled from the Categorical Distribution, contninuous action is sampled from a Normal Distribution with corresponding mu and sigma
        if not evaluation_episode and not start_episodes:
            action_probabilities_numpy = action_probabilities.squeeze(0).detach().cpu().numpy()
            # Ensure probabilities are non-negative and sum to 1
            action_probabilities_numpy = np.abs(action_probabilities_numpy)
            action_probabilities_numpy = action_probabilities_numpy / np.sum(action_probabilities_numpy)
            #get discrete action
            action = np.random.choice(range(self.action_dim), p=action_probabilities_numpy)
       
            #get corresponding continous action (=angle)
            pi_distribution = Normal(mu.squeeze(0)[action], sigma.squeeze(0)[action])
            angle_action=pi_distribution.sample() # No need to squeeze, already 1D
        
        #DETERMINISTICALLY during evaluation: discrete action with highest prob is chosen and for the corresponding continuous distribution the angle is chosen as mu
        elif evaluation_episode:
            action_probabilities_numpy = action_probabilities.squeeze(0).detach().cpu().numpy()
            #get discrete action
            action = np.argmax(action_probabilities_numpy)
          
            #get corresponding continous action (=angle)
            angle_action=mu.squeeze(0)[action]

        #RANDOMLY during start_episodes:
        else: # start_episodes == True
            action = np.random.choice(range(self.action_dim))
        
            #get corresponding continous action (=angle)
            choices = np.arange(-100,101)
            num = np.random.choice(choices)
            angle_action=torch.tensor(np.pi*(num/100), device=self.device)

        angle_action = torch.tanh(angle_action)
        angle_action = self.act_lower_bounds + 0.5 * (angle_action + 1.0) * (self.act_upper_bounds - self.act_lower_bounds)
        
        return action, angle_action.detach().cpu().numpy()

    #############################################################################

    
    
    #SAVE TRANSITIONS TO REPLAY BUFFER AND UPDATE THE NETWORKS
    def train_on_transition(self, state, action, action_c, next_state, energy, energy_before, done, e,index):
        
        self.training_count=self.training_count+1+1/self.train_evaluate_ratio  
        #Saving transition to replay buffer
        state_tensor = torch.tensor(np.array(state)).float().to(self.device)
        action_tensor = torch.tensor(np.array(action)).to(self.device)
        action_c_tensor = torch.tensor(np.array(action_c)).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_state)).float().to(self.device)
        energy_tensor = torch.tensor(np.array(energy)).float().to(self.device)
        energy_before_tensor = torch.tensor(np.array(energy_before)).float().to(self.device)
        done_tensor = torch.tensor(np.array(done)).to(self.device)
        index_tensor = torch.tensor(np.array(index)).to(self.device)
        self.replay_buffer.add_transition(state_tensor,action_tensor,action_c_tensor,next_state_tensor,energy_tensor,energy_before_tensor,done_tensor,index_tensor)
        
        #Is it time to update? (Updates are only made if replay buffer has enough transitions and only all 'training_update_factor'-times)

        if self.replay_buffer.get_size() >=1024 and int(self.training_count)%self.training_update_factor==0:
            self.train_networks(e)

    def train_networks(self, e):
        
        #if it is time to update, the update is made 'training_update_factor'-times in a row
        for i in range(self.training_update_factor):

            #SAMPLE BATCH FROM REPLAY BUFFER
            
            states_tensor,actions_tensor,actions_c_tensor,next_states_tensor,energy_tensor,energy_before_tensor, done_tensor,index_tensor= self.replay_buffer.sample_batch(self.batch_size)
            #print(self.E_learn,self.E_mins,self.E_mins_2)
            rewards_tensor=self.rwd.which_reward(energy_tensor,energy_before_tensor,done_tensor ,index_tensor,tensor_form=True)

            #print(rewards_tensor)
            #######--------CRITIC LOSS-------######
            critic_loss1, critic_loss2 = self.critic_loss(states_tensor, actions_tensor,actions_c_tensor, next_states_tensor, rewards_tensor, done_tensor)
    
            #Update critic networks
            self.critic_optimiser1.zero_grad()
            self.critic_optimiser2.zero_grad()
            critic_loss1.backward()
            critic_loss2.backward()
            self.critic_optimiser1.step()
            self.critic_optimiser2.step()
            ########################################
           
            #freeze Q params for policy update
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False
            
            #######--------POLICY LOSS-------#######
            policy_loss= self.actor_loss(states_tensor)
            
        
            #Update the actor network
            
            self.actor_optimiser.zero_grad()
            policy_loss.backward()
            self.actor_optimiser.step()

            #unfreeze Q params 
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True
            
            ########################################

            #####-----TEMPERATURE TUNING-----####
            alpha_loss_d, real_entropy_d, alpha_loss_c,real_entropy_c=self.get_alpha_loss_and_entropy(states_tensor)
            
            #Update alpha discrete
            self.log_alpha_optimiser_d.zero_grad()
            alpha_loss_d.backward()
            self.log_alpha_optimiser_d.step()
            self.alpha_d = self.log_alpha_d.exp()
            #Update alpha continuous
            self.log_alpha_optimiser_c.zero_grad()
            alpha_loss_c.backward()
            self.log_alpha_optimiser_c.step()
            self.alpha_c = self.log_alpha_c.exp()
            #print(self.alpha_d,self.alpha_c)
            #####################################
            
            #Soft update of the target networks
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)
            ##################################

            #Save the results
            Q_loss=1/2*(critic_loss1+critic_loss1)
            
        self.rl_results.append((e,policy_loss.detach().cpu().numpy(),Q_loss.detach().cpu().numpy(),self.alpha_d.detach().cpu().numpy(), self.alpha_c.detach().cpu().numpy(),real_entropy_d.detach().cpu().numpy(),real_entropy_c.detach().cpu().numpy()))
    

    def critic_loss(self, states_tensor, actions_tensor,actions_c_tensor, next_states_tensor,rewards_tensor, done_tensor):
       
        with torch.no_grad(): #dont keep track of the gradients, it is not needed
            
            action_probabilities, log_action_probabilities,angle_action,log_angle_action = self.get_pi_and_logpi(next_states_tensor)
  
            q_values_target1  = (self.critic_target1(torch.cat((next_states_tensor.repeat_interleave(self.action_dim,-2), angle_action.view(self.batch_size*self.action_dim,1)),dim=1))).gather(1,self.number_tensor).view(self.batch_size,self.action_dim)
            q_values_target2 = (self.critic_target2(torch.cat((next_states_tensor.repeat_interleave(self.action_dim,-2),  angle_action.view(self.batch_size*self.action_dim,1)),dim=1))).gather(1,self.number_tensor).view(self.batch_size,self.action_dim)
            
           
            #Clipped double Q-trick: Use the smaller Q-value
            q_values_targets_min=torch.min(q_values_target1, q_values_target2)
          
            #Calculate the soft state value function
            soft_state_value_function = (action_probabilities * (q_values_targets_min - self.alpha_d * log_action_probabilities- self.alpha_c*log_angle_action)).sum(dim=1)
           
            #Calculate the bellman backup operator
            bellman_backup_operator = rewards_tensor + ~done_tensor * self.gamma*soft_state_value_function

       
        Q_values1 = self.critic1(torch.cat((states_tensor, actions_c_tensor.unsqueeze(1)),dim=1)).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        Q_values2 = self.critic2(torch.cat((states_tensor,actions_c_tensor.unsqueeze(1)),dim=1)).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
       

        #Calculate MSE of the critic networks
        critic_MSE_1=1/2*(Q_values1-bellman_backup_operator)**2
        critic_MSE_2=1/2*(Q_values2- bellman_backup_operator)**2
       
        #Calculate the critic losses
        critic_loss1 = critic_MSE_1.mean()
        critic_loss2 = critic_MSE_2.mean()

        return critic_loss1, critic_loss2

    def actor_loss(self, states_tensor):
        
        action_probabilities, log_action_probabilities,angle_action,log_angle_action = self.get_pi_and_logpi(states_tensor)
       

        #with torch.no_grad():

        #Get Q-values from the critic networks
        q_values1  = (self.critic1(torch.cat((states_tensor.repeat_interleave(self.action_dim,-2), angle_action.view(self.batch_size*self.action_dim,1)),dim=1))).gather(1,self.number_tensor).view(self.batch_size,self.action_dim)
        q_values2 = (self.critic2(torch.cat((states_tensor.repeat_interleave(self.action_dim,-2),  angle_action.view(self.batch_size*self.action_dim,1)),dim=1))).gather(1,self.number_tensor).view(self.batch_size,self.action_dim)
        #print(q_values1)
            
        
        #Calculate the policy loss
        policy_loss =(1/self.batch_size)*((action_probabilities * (self.alpha_d * log_action_probabilities+self.alpha_c*log_angle_action - torch.min(q_values1, q_values2))).sum(dim=1)).sum()
  
        return policy_loss

    def get_alpha_loss_and_entropy(self,states_tensor):

        action_probabilities, log_action_probabilities,angle_action,log_angle_action = self.get_pi_and_logpi(states_tensor)
       
        #Discrete distribution
        alpha_loss_d = -(1/self.batch_size)*(self.log_alpha_d.exp() * action_probabilities.detach()*(log_action_probabilities + self.target_entropy_d).detach()).sum()
        real_entropy_d=-(1/self.batch_size)*(action_probabilities * log_action_probabilities).sum()
        
        #Continuous distributions
        alpha_loss_c = -(1/self.batch_size)*(self.log_alpha_c.exp() * action_probabilities.detach()*(log_angle_action + self.target_entropy_c).detach()).sum()
        real_entropy_c=-(1/self.batch_size)*(action_probabilities * log_angle_action).sum()
        #print(real_entropy_d,real_entropy_c)
        return alpha_loss_d, real_entropy_d, alpha_loss_c,real_entropy_c

    
        

        
    """
    action probabilities and log in the form needed to perform updates, i.e
    from the neural network one gets vec(p),vec(mu),vec(sigma) and here it is further processed
    to obtain vec(p),vec(log(p)),vec(sampled continuous action),vec(log of sampled continuous action)
    *vectors contain each quantity for each discrete action
    """
    def get_pi_and_logpi(self, states_tensor):
        #DISCRETE
        mu, std, action_probabilities = self.actor.forward(states_tensor)
        z=1e-10
        log_action_probabilities = torch.log(action_probabilities + z)#.to(self.device)
        #CONTINUOUS
        pi_distributions=Normal(mu,std)
        angle_action=pi_distributions.rsample()
        logp_pi = pi_distributions.log_prob(angle_action)
        logp_pi -= (2*(np.log(2) - angle_action - F.softplus(-2*angle_action)))#.sum(axis=1, keepdim=True) 
        angle_action=torch.tanh(angle_action)
        angle_action= self.act_lower_bounds + 0.5*(angle_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)
       
        return action_probabilities, log_action_probabilities,angle_action,logp_pi

    #Function for the soft update of the target networks
    def soft_update(self, target_model, origin_model):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(self.soft_update_factor * local_param.data + (1 - self.soft_update_factor) * target_param.data)

    def rl_quantities(self):
        return self.rl_results
   
    def dynamic_values_for_reward_update(self,E,index):
        self.rwd.dynamic_values_for_reward_update(E,index)


    #update the target entropy
    def update_target_entropy(self,step,decay):
        

        h_start_d=np.log(self.environment.action_size)-self.target_entropy_deduction_d
        h_end_d=self.target_entropy_end_value_d
        h_start_c=np.log(2)-self.target_entropy_deduction_c
        h_end_c=self.target_entropy_end_value_c

        self.target_entropy_d=h_end_d+(h_start_d-h_end_d)*np.exp(-self.decay_factor_d*(step/decay))
        self.target_entropy_c=h_end_c+(h_start_c-h_end_c)*np.exp(-self.decay_factor_c*(step/decay))


    def save_policy(self, x, session_folder):
        actor_model_path = os.path.join(session_folder, f"actor_model_{x}.pt")
        torch.save(self.actor.state_dict(), actor_model_path)