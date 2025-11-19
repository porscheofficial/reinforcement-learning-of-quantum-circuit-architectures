import torch

"""
File for setting up the neural network architecture.
"""
class Critic(torch.nn.Module):

    """
    Q-networks: q(s, continous action=c)--> vec(q value for each discrete action)=(q(s,a_1,c),...q(s,a_|A|,c))
    if the q-value is needed for a sepcific action, it is is gathered afterwards in SAC.py
    """

    def __init__(self, cfg, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
            super(Critic, self).__init__()
            self.hidden_layers=cfg.NeuralNet["hidden_layers_critic"] 
            # First layer
            self.layers = torch.nn.ModuleList()
            self.layers.append(torch.nn.Linear(input_dimension, self.hidden_layers[0]))

            
            for i in range(1, len(self.hidden_layers)):
                self.layers.append(torch.nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))

            #Output layer
            self.output_layer = torch.nn.Linear(self.hidden_layers[-1], output_dimension)
            self.output_activation = output_activation

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        qvalues = self.output_activation(self.output_layer(x))
        return qvalues




class Actor(torch.nn.Module):


    """
    Policy-networks: pi(s)-->(vec(p)=(p1,..,p_|A|), vec(mu)=,(mu1,..,mu_|A|), vec(sigma)=(sigma1,...,sigma_|A|)), 
    i.e outputs for each possible action a probability to take the
    discrete action plus a mu and sigma for the corresponding continous/angle distribution
    """
    def __init__(self,cfg, input_dimension,  output_dimension, output_activation=torch.nn.Identity()):
        super(Actor, self).__init__()

        #Constrcut Network architecture
        self.hidden_layers=cfg.NeuralNet["hidden_layers_actor"] 
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dimension, self.hidden_layers[0]))

        for i in range(1, len(self.hidden_layers)):
            self.layers.append(torch.nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))

        #Three heads: Discrete probability, mu, sigma
        last_hidden_size = self.hidden_layers[-1]
        self.discrete_probs_layer = torch.nn.Linear(last_hidden_size, output_dimension)
        self.mu_layer = torch.nn.Linear(last_hidden_size, output_dimension)
        self.sigma_layer = torch.nn.Linear(last_hidden_size, output_dimension)
        self.output_activation = output_activation

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))

        # Output p, mu, log_sigma
        discrete_probs = self.output_activation(self.discrete_probs_layer(x))
        mu = self.mu_layer(x)
        log_std = self.sigma_layer(x)
        
        # Clamp log_std for stability and exponentiate to get std
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mu, std, discrete_probs



