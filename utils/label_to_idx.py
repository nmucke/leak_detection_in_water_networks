import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb


class LabelToIndex():
    def __init__(
            self,
            pipe_labels,
            node_labels,
    ):

        self.pipe_labels = pipe_labels
        self.node_labels = node_labels

        self.pipe_label_to_index_dict = self.pipe_label_to_index()
        self.index_to_pipe_label_dict = self.index_to_pipe_label()

        self.node_label_to_index_dict = self.node_label_to_index()
        self.index_to_node_label_dict = self.index_to_node_label()


    def pipe_label_to_index(self):

        pipe_index = [self.pipe_labels.index(pipe) for pipe in self.pipe_labels]
        pipe_label_to_index_dict = dict(zip(self.pipe_labels, pipe_index))

        return pipe_label_to_index_dict

    def index_to_pipe_label(self):

        pipe_index = list(self.pipe_label_to_index_dict.values())
        pipe_labels = list(self.pipe_label_to_index_dict.keys())
        index_to_pipe_label_dict = dict(zip(pipe_index, pipe_labels))

        return index_to_pipe_label_dict


    def node_label_to_index(self):

        node_index = [self.node_labels.index(node) for node in self.node_labels]
        node_label_to_index_dict = dict(zip(self.node_labels, node_index))

        return node_label_to_index_dict

    def index_to_node_label(self):

        node_index = list(self.node_label_to_index_dict.values())
        node_labels = list(self.node_label_to_index_dict.keys())
        index_to_node_label_dict = dict(zip(node_index, node_labels))

        return index_to_node_label_dict
