import torch
import numpy as np
from collections import deque

def argmax(arr):
    """ Custom argmax to return all args that maximize arr

    Params
    ======
        arr(array): Array whose argmax is calculated
    """ 
    return np.array([i for i, a in enumerate(arr) if a == max(arr)])

def epsilon_greedy(q_vals, epsilon):
    """ Returns chosen action and probabilites according to epsilon-greedy strategy 

    Params
    ======
        q_vals(torch.Tensor): action-values
        epsilon(float): epsilon in epsilon-greedy 
    """
    probs = np.ones(q_vals.shape) * epsilon
    max_inds = argmax(q_vals)
    probs[max_inds] += (1.-epsilon)/max_inds.shape[0]
    if np.random.uniform() < epsilon:
        return np.random.randint(q_vals.shape[0]), probs
    else:
        return torch.argmax(q_vals).item(), probs

def greedy(q_vals):
    """ Returns chosen action and probabilites according to greedy strategy 

    Params
    ======
        q_vals(torch.Tensor): action-values
    """
    probs = np.zeros(q_vals.shape)
    max_inds = argmax(q_vals)
    probs[max_inds] += 1./max_inds.shape[0]
    return max_inds[0], probs


def target(reward, gamma, q_vals_next, target_policy, done):
    """ Returns general TD-target 

    Params
    ======
        reward(float): current reward
        gamma(float): discount factor
        q_vals_next(torch.Tensor): action values of next state
        target_policy(torch.Tensor -> (float, array)): target policy returning chosen action and probabilites
        done(bool): flag indicating terminal state 
    """
    probs = torch.from_numpy(target_policy(q_vals_next)[1]).float()
    return (reward + gamma * (torch.dot(q_vals_next.cpu(), probs).float()) * (1.-done)).float()


def copy_model(model_from, model_to, tau=1.0):
    """ Soft update of model weights weighted with tau 

    Params
    ======
        model_from(Model): model to copy from
        model_to(Model): model to copy to
        tau(float): weight factor for soft update 
    """
    for target_param, policy_param in zip(model_to.parameters(), model_from.parameters()):
        target_param.data = (1-tau)*target_param.data + tau * policy_param.data

def prefixsum(arr):
    """ Returns prefix-sum of arr  

    Params
    ======
        arr(list): array to calculate refix-sum of
    """
    presum = np.zeros(len(arr))
    for i in range(len(arr)):
        presum[i] = presum[i-1] + arr[i]
    return presum
    

class ReplayBuffer():
    """ Replaybuffer for usage in Agent.""" 

    def __init__(self, size, batchsize):
        """ Initialize a ReplayBuffer object 

        Params
        ======
            size(int): size of the replaybuffer
            batchsize(int): size of a sample from the replaybuffer
        """
        self.buffer = deque(maxlen=size)
        self.batchsize = batchsize

    def append(self, state, action, reward, next_state, done):
        """ Append new experience 

        Params
        ======
            state(torch.Tensor): current state
            action(int): chosen action
            reward(float): observed reward
            next_state(torch.Tensor): next state
            done(bool): flag indicating terminal state
        """
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        """ Returns sample of memories from replaybuffer """
        indices = np.random.randint(low = 0, high = len(self.buffer), size=1000)
        states_t = []
        actions = []
        rewards = []
        next_states_t = []
        dones = []
        for ind in indices:
            state_t, action, reward, next_state_t, done = self.buffer[ind]
            states_t.append(state_t)
            actions.append(action)
            rewards.append(reward)
            next_states_t.append(next_state_t)
            dones.append(done)

        return states_t, actions, rewards, next_states_t, dones
    
    def __len__(self):
        """ Returns current length of replaybuffer """
        return self.buffer.__len__()




class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replaybuffer for usage in Agent."""
    def __init__(self, size, batchsize, epsilon=0.0):
        """ Initialize a ReplayBuffer object 

        Params
        ======
            size(int): size of the replaybuffer
            batchsize(int): size of a sample from the replaybuffer
            epsilon(float): regularizer for priorization
        """
        super().__init__(size, batchsize)
        self.weights = deque(maxlen=size)
        self.epsilon = abs(epsilon)

    def append(self, state, action, reward, next_state, done, weight):
        """ Append new experience 

        Params
        ======
            state(torch.Tensor): current state
            action(int): chosen action
            reward(float): observed reward
            next_state(torch.Tensor): next state
            done(bool): flag indicating terminal state
            weight(float): weight for priorization
        """
        self.buffer.append([state, action, reward, next_state, done])
        self.weights.append(abs(weight)+self.epsilon)

    def sample(self):
        """ Returns sample of memories from replaybuffer """
        p = np.array(self.weights)/np.sum(self.weights)
        indices = np.random.choice(len(self.weights), size=self.batchsize, p = p)
        states_t = []
        actions = []
        rewards = []
        next_states_t = []
        dones = []
        for ind in indices:
            state_t, action, reward, next_state_t, done = self.buffer[ind]
            states_t.append(state_t)
            actions.append(action)
            rewards.append(reward)
            next_states_t.append(next_state_t)
            dones.append(done)

        return states_t, actions, rewards, next_states_t, dones
    
    def __len__(self):
        """ Returns current length of replaybuffer """
        return self.buffer.__len__()


