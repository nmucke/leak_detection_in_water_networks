import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class GraphAttention(nn.Module):
    def __init__(
        self, 
        in_node_features=10, 
        out_node_features=10, 
        in_edge_features=10,
        out_edge_features=10,
        adj_matrix=None,
        num_nodes=10,
        num_egdes=10,
        ):
        super().__init__()

        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.in_edge_features = in_edge_features
        self.out_edge_features = out_edge_features
        self.adj_matrix = torch.tensor(adj_matrix)
        self.adj_matrix = self.adj_matrix + torch.eye(num_nodes)
        self.num_nodes = num_nodes
        self.num_egdes = num_egdes
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.adj_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))

        self.edge_coords = torch.where(self.adj_matrix==1)

        self.W = nn.Parameter(torch.Tensor(self.in_node_features, self.out_node_features), requires_grad=True)
        self.W.data.normal_(0, 0.1)
        self.a = nn.Parameter(torch.Tensor(2*self.out_node_features, 1), requires_grad=True)
        self.a.data.normal_(0, 0.1)

    def forward(self, x):
        edges = x[:, 0:self.num_egdes].unsqueeze(-1)
        nodes = x[:, self.num_egdes:].unsqueeze(-1)

        h = torch.matmul(nodes, self.W)

        h_src_nodes = h[:, self.edge_coords[0]]
        h_dst_nodes = h[:, self.edge_coords[1]]

        alpha = torch.cat((h_src_nodes, h_dst_nodes), dim=2)
        alpha = torch.matmul(alpha, self.a).squeeze(-1)
        alpha = self.leaky_relu(alpha)
        alpha = self.softmax(alpha)

        self.adj_matrix_scaled = self.adj_matrix_zeros.repeat(x.shape[0], 1, 1)
        self.adj_matrix_scaled[:, self.edge_coords[0], self.edge_coords[1]] = alpha
        
        h_prime = torch.matmul(self.adj_matrix_scaled, h)
        
        return self.leaky_relu(h_prime)

class MultiHeadGraphAttention(nn.Module):
    def __init__(
        self, 
        in_node_features=10, 
        out_node_features=10, 
        in_edge_features=10,
        out_edge_features=10,
        adj_matrix=None,
        num_heads=1,
        num_nodes=10,
        num_egdes=10,
        ):
        super().__init__()

        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.attention_heads.append(
                GraphAttention(
                    in_node_features=in_node_features, 
                    out_node_features=out_node_features, 
                    in_edge_features=in_edge_features, 
                    out_edge_features=out_edge_features, 
                    adj_matrix=adj_matrix, 
                    num_nodes=num_nodes, 
                    num_egdes=num_egdes
                    )
                )
        
    def forward(self, x):
        h = torch.cat(
            [attention_head(x) for attention_head in self.attention_heads],
             dim=-1
             )
        return h




class Encoder(nn.Module):
    def __init__(self, latent_dim=32, state_dim=128, hidden_neurons=[64, 32, 16]):
        super().__init__()

        self.activation = nn.LeakyReLU()
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim

        self.dense_in = nn.Linear(in_features=state_dim,
                              out_features=self.hidden_neurons[0])
        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=hidden_neurons[i],
                        out_features=hidden_neurons[i + 1]
                ) for i in range(len(hidden_neurons)-1)]
        )
        self.dense_out = nn.Linear(in_features=hidden_neurons[-1],
                                   out_features=latent_dim,
                                   bias=False
                                   )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_neurons[i])
                 for i in range(1, len(self.hidden_neurons))]
        )
        self.normalize = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.activation(x)
        x = self.batch_norm_in(x)

        for dense_layer, batch_norm in zip(self.dense_layers,
                                           self.batch_norm_layers):
            x = dense_layer(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = self.dense_out(x)
        x = self.normalize(x)

        return x

class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim=32,
            state_dim=128,
            hidden_neurons=[],
            pars_dim=119,
            pars_embedding_dim=32
    ):
        super().__init__()

        self.activation = nn.LeakyReLU()
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dim = pars_dim

        self.pars_embedding = nn.Embedding(
                num_embeddings=pars_dim,
                embedding_dim=pars_embedding_dim
        )

        self.dense_in = nn.Linear(in_features=latent_dim,
                              out_features=self.hidden_neurons[0])
        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.hidden_neurons[0] += pars_embedding_dim
        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=hidden_neurons[i],
                        out_features=hidden_neurons[i + 1]
                ) for i in range(len(hidden_neurons)-1)]
        )
        self.dense_out = nn.Linear(in_features=hidden_neurons[-1],
                                   out_features=state_dim,
                                   bias=False
                                   )

        self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_neurons[i])
                 for i in range(1, len(self.hidden_neurons))]
        )

    def forward(self, x, pars):

        x = self.dense_in(x)
        x = self.activation(x)
        x = self.batch_norm_in(x)

        pars = self.pars_embedding(pars[:, 0])

        x = torch.cat((x, pars), dim=1)

        for dense_layer, batch_norm in zip(self.dense_layers,
                                           self.batch_norm_layers):

            x = dense_layer(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = self.dense_out(x)
        return x

class Critic(nn.Module):
    def __init__(self, latent_dim=32, hidden_neurons=[]):
        super().__init__()

        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_neurons = hidden_neurons

        self.dense_in = nn.Linear(in_features=latent_dim,
                              out_features=self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=hidden_neurons[i],
                        out_features=hidden_neurons[i + 1]
                ) for i in range(len(hidden_neurons)-1)]
        )
        self.dense_out = nn.Linear(in_features=hidden_neurons[-1],
                                   out_features=latent_dim,
                                   bias=False
                                   )

    def forward(self, x):

        x = self.dense_in(x)
        x = self.activation(x)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.activation(x)

        x = self.dense_out(x)
        return self.sigmoid(x)

class AutoEncoder():
    def __init__(self, latent_dim=32, input_dim=128,
                 encoder_params={}, decoder_params={}):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim,
                               input_dim=input_dim,
                               encoder_params=encoder_params)
        self.decoder = Decoder(latent_dim=latent_dim,
                               input_dim=input_dim,
                               decoder_params=decoder_params)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

if __name__ == '__main__':


    def out_size(in_size, stride, padding, kernel_size, out_pad):
        return (in_size-1)*stride-2*padding+1*(kernel_size-1)+out_pad+1

    stride = [2, 2, 2, 2, 2]
    padding = [0, 0, 0, 0, 0, 0]
    out_pad = [0, 1, 1, 0, 0]
    kernel_size = [7,7,7,7,4]

    in_size = 3
    for i in range(len(stride)):
        in_size = out_size(in_size, stride[i], padding[i], kernel_size[i],
                           out_pad[i])
        print(in_size)

