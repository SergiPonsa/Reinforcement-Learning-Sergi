import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module): #Neural net to choose the action taking into acount the state
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=600 , fc2_units = 300): #original only 1 hidden layer 256 neurons
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__() # It's a class of neural net, evolved from there
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)# Always will have the states as input,the number of hidden layersit's 
                                                        #choosen by us,more layers more complex better results, but slower
        self.fc2 = nn.Linear(fc1_units,fc2_units)   
        self.fc3 = nn.Linear(fc2_units, action_size)# Always will have the number of actions as output 
        
        self.bn1 = nn.BatchNorm1d(fc1_units); # Normalize the input increasing the speed
        self.bn2 = nn.BatchNorm1d(fc2_units);
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
        #With batch normalization (doesn't work)
        #x = F.relu((self.bn1(self.fc1(state)))) # alow only positive
        #x = F.relu((self.bn2(self.fc2(x))))
        
        #without batch normaization
        x = F.relu(self.fc1(state)) # alow only positive
        x = F.relu(self.fc2(x))
            #return F.tanh(self.fc2(x))# alow positive and negative functions from -1 to 1
        return torch.tanh(self.fc3(x))# alow positive and negative functions from -1 to 1


class Critic(nn.Module): # Evaluate the selected action for that state
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=600, fc2_units=300, fc3_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__() # it's a class of neural net
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units) # first layer the actual state
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units) # second layer features of the actual state + the chossen action
        self.fc3 = nn.Linear(fc2_units, fc3_units) # third layer taking into acount the state and the chosen action for that 
                                                    #state
        self.fc4 = nn.Linear(fc3_units, 1) #Returns a single value the expected reward from that state and action taken, the Q
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # With batch normalization (don't work)
        #xs = F.leaky_relu((self.bn1(self.fcs1(state)))) # Allow mainly positive values for the states, but some negative too
        
        #Without batch normalization
        xs = F.leaky_relu(self.fcs1(state)) # Allow mainly positive values for the states, but some negative too
        x = torch.cat((xs, action), dim=1)  # Put together the action input with the output of the previous layer
        x = F.leaky_relu(self.fc2(x)) # Allow mainly positive values for the states, but some negative too
        x = F.leaky_relu(self.fc3(x)) # Allow mainly positive values for the states, but some negative too
        return self.fc4(x)
