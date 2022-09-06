import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from data_handling.network_dataset import NetworkDataset

class StateTransformer():
    def __init__(self, num_pipes, num_nodes):

        self.num_pipes = num_pipes
        self.num_nodes = num_nodes

        self.min_flow_rate = 1e8
        self.max_flow_rate = -1e8

        self.min_head = 1e8
        self.max_head = -1e8

    def partial_fit(self, state):
        flow_rate = state[:, 0:self.num_pipes]
        head = state[:, self.num_pipes:self.num_pipes+self.num_nodes]

        if torch.any(flow_rate < self.min_flow_rate):
            self.min_flow_rate = torch.min(flow_rate)
        if torch.any(flow_rate > self.max_flow_rate):
            self.max_flow_rate = torch.max(flow_rate)

        if torch.any(head < self.min_head):
            self.min_head = torch.min(head)
        if torch.any(head > self.max_head):
            self.max_head = torch.max(head)

    def transform_state(self, state):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        flow_rate = state[:, 0:self.num_pipes]
        head = state[:, self.num_pipes:self.num_pipes+self.num_nodes]

        flow_rate = (flow_rate - self.min_flow_rate) / (self.max_flow_rate - self.min_flow_rate)
        head = (head - self.min_head) / (self.max_head - self.min_head)
        
        state = torch.cat((flow_rate, head), dim=1).squeeze(0)
        return state

    def inverse_transform_states(self, states):
        flow_rate = states[:, 0:self.num_pipes]
        head = states[:, self.num_pipes:self.num_pipes+self.num_nodes]

        flow_rate = flow_rate * (self.max_flow_rate - self.min_flow_rate) + self.min_flow_rate
        head = head * (self.max_head - self.min_head) + self.min_head

        state = torch.cat((flow_rate, head), dim=1).squeeze(0)
        return state

