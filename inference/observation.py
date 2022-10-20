import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from utils.label_to_idx import LabelToIndex

class ObservationOperator():
    def __init__(
            self,
            pipe_labels,
            node_labels,
            pipe_observation_labels,
            node_observations_labels
    ):

        self.pipe_labels = pipe_labels
        self.node_labels = node_labels
        self.pipe_observation_labels = pipe_observation_labels
        self.node_observations_labels = node_observations_labels
        self.num_pipes = len(pipe_labels)
        self.num_nodes = len(node_labels)

        self.LabelToIndex = LabelToIndex(self.pipe_labels, self.node_labels)
        self.pipe_obs_idx = [self.LabelToIndex.pipe_label_to_index_dict.get(key) \
                             for key in self.pipe_observation_labels]
        self.node_obs_idx = [self.LabelToIndex.node_label_to_index_dict.get(key) \
                             for key in self.node_observations_labels]

    def get_observations(self, state):
        if len(state.shape) == 1:
            flow_rate = state[0:self.num_pipes]
            head = state[self.num_pipes:self.num_pipes+self.num_nodes]

            flow_rate_obs = flow_rate[self.pipe_obs_idx]
            head_obs = head[self.node_obs_idx]

            obs = torch.cat((flow_rate_obs, head_obs))
        elif len(state.shape) == 2:
            flow_rate = state[:, 0:self.num_pipes]
            head = state[:, self.num_pipes:self.num_pipes+self.num_nodes]

            flow_rate_obs = flow_rate[:, self.pipe_obs_idx]
            head_obs = head[:, self.node_obs_idx]

            obs = torch.cat((flow_rate_obs, head_obs), dim=1)

        return obs