import numba
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

@numba.njit()
def calculate_gae(values,rewards,dones,gamma,gae_lambda):
    if len(values.shape) != 1:
        return None,None
    
    length = values.shape[0]
    dones[length-1] = True
    advantages = np.zeros(length, dtype=np.float32)
    returns = np.zeros(length, dtype=np.float32)
    
    last_gae = 0.0
    for index in range(length-1,-1,-1):
        if dones[index]:
            delta = rewards[index] - values[index]
            last_gae = delta
        else:
            delta = rewards[index] + gamma * values[index+1] - values[index]
            last_gae = delta + gamma * gae_lambda * last_gae
            
        advantages[index] = last_gae
        returns[index] = last_gae + values[index]
                                         
    return advantages, returns

class MaskedCategorical:
    def __init__(self, logits):
        self.origin_logits = logits
        self.probs = F.softmax(logits,dim=-1)
        self.dist = Categorical(self.probs)
            
    def update_masks(self,masks,device = 'cpu'):
        if masks is None:
            return self
        probs = torch.lerp(self.origin_logits, torch.tensor(-1e+10).to(device), 1.0 - masks)
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
        return self
    
    def update_bias_masks(self,masks,device = 'cpu'):
        if masks is None:
            return self
        probs = self.origin_logits + torch.log(masks)
        self.probs = F.softmax(probs,dim=-1)
        self.dist = Categorical(self.probs)
        return self
    
    def sample(self):
        actions = self.dist.sample()
        return actions
    
    def log_prob(self,actions):
        return self.dist.log_prob(actions)
    
    def entropy(self):
        return self.dist.entropy()
    
    def argmax(self):
        return torch.argmax(self.probs,dim=-1)
    
    def argmin(self):
        return torch.argmin(self.probs,dim=-1)
