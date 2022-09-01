import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time
import networkx as nx

def get_edge_from_pair_of_nodes(inc_matrix, node1, node2):
    """
    Get the edge between two nodes.
    """
    edge_index = np.where(inc_matrix[node1, :] * inc_matrix[node2, :] == 1)
    edge_index = edge_index[0][0]

    return edge_index

def get_all_edge_for_node_pairs(inc_matrix, node_pairs):
    """
    Get all the edges for a list of node pairs.
    """
    edge_list = []
    for node1, node2 in zip(node_pairs[0], node_pairs[1]):
        edge_list.append(get_edge_from_pair_of_nodes(inc_matrix, node1, node2))
        
    return torch.tensor(np.asarray(edge_list))


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class EdgeGraphAttention(nn.Module):
    def __init__(
        self, 
        in_node_features=10, 
        out_node_features=10, 
        in_edge_features=10,
        out_edge_features=10,
        graph=None,
        num_nodes=10,
        num_edges=10,
        ):
        super().__init__()
        
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.in_edge_features = in_edge_features
        self.out_edge_features = out_edge_features

        self.adj_matrix = nx.adjacency_matrix(graph).toarray()
        self.adj_matrix = torch.tensor(self.adj_matrix)
        self.adj_matrix = self.adj_matrix# + torch.eye(num_nodes)
        self.adj_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))
        self.node_pair_idx = torch.where(self.adj_matrix==1)

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.inc_matrix = nx.incidence_matrix(graph).toarray()
        self.inc_matrix = torch.tensor(self.inc_matrix)
        self.inc_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))
        self.edge_node_pair_idx = get_all_edge_for_node_pairs(
            self.inc_matrix, 
            self.node_pair_idx
            )


        self.dense_node_for_edge = nn.Linear(
            self.in_node_features, 
            self.out_edge_features,
            bias=False
            )
        self.dense_edge = nn.Linear(
            self.in_edge_features, 
            self.out_edge_features,
            bias=False
            )
        self.dense_node_edge_combine = nn.Linear(
            2*self.out_node_features + self.out_edge_features, 
            self.out_node_features,
            bias=True
            )
        self.attn = nn.Linear(
            self.out_edge_features,
            1,
            bias=False
            )

        self.dense_node = nn.Linear(
            self.in_node_features, 
            self.out_node_features,
            bias=True
            )

        
        #self.W_node = nn.Parameter(torch.Tensor(self.in_node_features, self.out_node_features), requires_grad=True)
        #self.W_node.data.normal_(0, 0.1)
        
        #self.bias_node = nn.Parameter(torch.Tensor(self.out_node_features), requires_grad=True)
        #self.bias_node.data.zero_()

        #self.W_edge = nn.Parameter(torch.Tensor(self.in_edge_features, self.out_edge_features), requires_grad=True)
        #self.W_edge.data.normal_(0, 0.1)

        #self.bias_edge = nn.Parameter(torch.Tensor(self.out_edge_features), requires_grad=True)
        #self.bias_edge.data.zero_()

        #self.a = nn.Parameter(torch.Tensor(
        #    self.out_edge_features + 2*self.out_node_features, 1),
        #    requires_grad=True
        #    )
        #self.a.data.normal_(0, 0.1)

        #self.bias_attention = nn.Parameter(torch.Tensor(self.num_egdes), requires_grad=True)
        #self.bias_attention.data.zero_()

    def forward(self, x):

        edges = x[:, 0:self.num_edges]
        nodes = x[:, self.num_edges:]

        #f = torch.matmul(edges, self.W_edge)
        #h = torch.matmul(nodes, self.W_node)
        #h_src_nodes = h[:, self.node_pair_idx[0]]
        #h_dst_nodes = h[:, self.node_pair_idx[1]]
        #f_edges = f[:, self.edge_node_pair_idx]
        #f_out = torch.matmul(f_out, self.a).squeeze(-1) + self.bias_attention
        #f_out = self.leaky_relu(f_out)
        #alpha = self.softmax(f_out)


        f = self.dense_edge(edges)
        h = self.dense_node(nodes)

        h_src_nodes = h[:, self.node_pair_idx[0]]
        h_dst_nodes = h[:, self.node_pair_idx[1]]
        f_edges = f[:, self.edge_node_pair_idx]

        f_out = torch.cat((h_src_nodes, f_edges, h_dst_nodes), dim=2)
        f_out = self.dense_node_edge_combine(f_out)
        f_out = self.leaky_relu(f_out)

        alpha = self.attn(f_out)
        alpha = self.softmax(alpha)

        h_out = self.dense_node(nodes)

        self.adj_matrix_scaled = self.adj_matrix_zeros.repeat(x.shape[0], 1, 1)
        self.adj_matrix_scaled[:, self.node_pair_idx[0], self.node_pair_idx[1]] = alpha.squeeze(-1)
        
        h_out = torch.matmul(self.adj_matrix_scaled, h_out)
        #h_out = self.leaky_relu(h_out)
        
        x = torch.cat((f_out, h_out), dim=1)
        return x


class GraphAttention(nn.Module):
    def __init__(
        self, 
        in_node_features=10, 
        out_node_features=10, 
        in_edge_features=10,
        out_edge_features=10,
        graph=None,
        num_nodes=10,
        num_edges=10,
        ):
        super().__init__()

        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.in_edge_features = in_edge_features
        self.out_edge_features = out_edge_features

        self.adj_matrix = nx.adjacency_matrix(graph).toarray()
        self.adj_matrix = torch.tensor(self.adj_matrix)
        self.adj_matrix = self.adj_matrix + torch.eye(num_nodes)
        self.adj_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))
        self.node_pair_idx = torch.where(self.adj_matrix==1)

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.W_node = nn.Parameter(torch.Tensor(self.in_node_features, self.out_node_features), requires_grad=True)
        self.W_node.data.normal_(0, 0.1)

        self.W_edge = nn.Parameter(torch.Tensor(self.in_edge_features, self.out_edge_features), requires_grad=True)
        self.W_edge.data.normal_(0, 0.1)

        self.a = nn.Parameter(torch.Tensor(2*self.out_node_features, 1), requires_grad=True)
        self.a.data.normal_(0, 0.1)

    def forward(self, x):
        edges = x[:, 0:self.num_edges].unsqueeze(-1)
        nodes = x[:, self.num_edges:].unsqueeze(-1)

        h = torch.matmul(nodes, self.W_node)

        h_src_nodes = h[:, self.node_pair_idx[0]]
        h_dst_nodes = h[:, self.node_pair_idx[1]]

        alpha = torch.cat((h_src_nodes, h_dst_nodes), dim=2)
        alpha = torch.matmul(alpha, self.a).squeeze(-1)
        alpha = self.leaky_relu(alpha)
        alpha = self.softmax(alpha)

        self.adj_matrix_scaled = self.adj_matrix_zeros.repeat(x.shape[0], 1, 1)
        self.adj_matrix_scaled[:, self.node_pair_idx[0], self.node_pair_idx[1]] = alpha
        
        h_prime = torch.matmul(self.adj_matrix_scaled, h)
        
        return self.leaky_relu(h_prime)

class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(
        self, 
        in_features=10, 
        hidden_features=10,
        out_features=10, 
        graph=None,
        num_heads=1,
        num_nodes=10,
        num_edges=10,
        ):
        super().__init__()

        self.num_heads = num_heads
        self.num_edges = num_edges
        self.num_nodes = num_nodes

        self.leaky_relu = nn.LeakyReLU()

        self.attention_heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.attention_heads.append(
                EdgeGraphAttention(
                    in_node_features=in_features, 
                    out_node_features=hidden_features, 
                    in_edge_features=in_features, 
                    out_edge_features=hidden_features, 
                    graph=graph,
                    num_nodes=num_nodes, 
                    num_edges=num_edges
                    )
                )
        self.dense_out1 = nn.Linear(
            num_heads*hidden_features, 
            hidden_features,
            bias=True
            )
        self.dense_out2 = nn.Linear(
            hidden_features, 
            out_features,
            bias=False
            )
        
    def forward(self, x):
        out = torch.cat(
            [attention_head(x) for attention_head in self.attention_heads],
            dim=-1
            )
        out = self.dense_out1(out)
        out = self.leaky_relu(out)
        out = self.dense_out2(out)

        return out

class GraphReduction(nn.Module):
    def __init__(
        self, 
        pivotal_nodes=10,
        num_edges=10,
        num_nodes=10,
        ):
        super().__init__()

        self.pivotal_nodes = pivotal_nodes
        self.num_pivotal_nodes = len(pivotal_nodes)
        self.num_edges = num_edges
        self.num_nodes = num_nodes

        #self.pivotal_nodes = torch.randint(
        #    low=0, 
        #    high=self.num_pivotal_nodes,
        #    size=(self.num_pivotal_nodes,)
        #)
    
    def forward(self, x):
        return x[:, self.num_edges + self.pivotal_nodes]


class GraphRecovery(nn.Module):
    def __init__(
        self, 
        num_features=10,
        pivotal_nodes=None,
        num_edges=10,
        num_nodes=10,
        ):
        super().__init__()

        self.pivotal_nodes = pivotal_nodes
        self.num_features = num_features
        self.num_pivotal_nodes = len(pivotal_nodes)
        self.num_edges = num_edges
        self.num_nodes = num_nodes

        '''
        self.activation = nn.LeakyReLU()

        self.dense1 = nn.Linear(
            self.num_features, 
            num_nodes + num_edges,
            bias=True
            )
        self.dense2 = nn.Linear(
            num_nodes + num_edges, 
            num_nodes + num_edges,
            bias=False
            )
        '''
    
    def forward(self, x):
        
        out = torch.zeros(x.shape[0], self.num_nodes + self.num_edges, self.num_features)
        out[:, self.num_edges + self.pivotal_nodes, :] = x

        return out


class GraphEncoder(nn.Module):
    def __init__(
        self, 
        latent_dim=10,
        pivotal_nodes=None,
        num_attention_layers=1,
        attention_layer_params=None,
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.pivotal_nodes = pivotal_nodes
        self.num_pivotal_nodes = len(pivotal_nodes)
        self.num_attention_layers = num_attention_layers
        self.attention_layer_params = attention_layer_params

        self.activation = nn.LeakyReLU()

        self.dense_in = nn.Linear(
            in_features=1,
            out_features=attention_layer_params['in_features'],
            bias=True
            )

        self.attention_layers = nn.ModuleList()
        for i in range(self.num_attention_layers):
            self.attention_layers.append(
                MultiHeadGraphAttentionLayer(**attention_layer_params)
                )
        self.graph_reduction = GraphReduction(
            pivotal_nodes=self.pivotal_nodes,
            num_edges=attention_layer_params['num_edges'],
            num_nodes=attention_layer_params['num_nodes'],
            )
        self.dense1 = nn.Linear(
            self.num_pivotal_nodes*attention_layer_params['out_features'], 
            self.latent_dim*2,
            bias=True
            )
        self.dense2 = nn.Linear(
            self.latent_dim*2, 
            self.latent_dim,
            bias=False
            )
    def forward(self, x):
        x = self.dense_in(x.unsqueeze(-1))
        x = self.activation(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        x = self.graph_reduction(x)
        
        x = x.view(-1, self.num_pivotal_nodes*self.attention_layer_params['out_features'])

        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)

        return x



class GraphDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim=10,
        pivotal_nodes=None,
        num_attention_layers=1,
        attention_layer_params=None,
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.pivotal_nodes = pivotal_nodes
        self.num_pivotal_nodes = len(pivotal_nodes)
        self.num_attention_layers = num_attention_layers
        self.attention_layer_params = attention_layer_params

        self.activation = nn.LeakyReLU()

        self.dense_in1 = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.latent_dim*2,
            bias=True
            )
        self.dense_in2 = nn.Linear(
            in_features=self.latent_dim*2,
            out_features=self.num_pivotal_nodes*attention_layer_params['out_features'],
            bias=True
            )

        self.graph_recovery = GraphRecovery(
            num_features=attention_layer_params['out_features'],
            pivotal_nodes=pivotal_nodes,
            num_edges=attention_layer_params['num_edges'],
            num_nodes=attention_layer_params['num_nodes'],
            )

        self.attention_layers = nn.ModuleList()
        for i in range(self.num_attention_layers):
            self.attention_layers.append(
                MultiHeadGraphAttentionLayer(**attention_layer_params)
                )
        self.dense_out = nn.Linear(
            attention_layer_params['out_features'], 
            1,
            bias=False
            )
    def forward(self, x):
        x = self.dense_in1(x)
        x = self.activation(x)
        x = self.dense_in2(x)
        x = x.view(-1, self.num_pivotal_nodes, self.attention_layer_params['out_features'])

        x = self.graph_recovery(x)

        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        x = self.dense_out(x)

        return x.squeeze(-1)

        



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

