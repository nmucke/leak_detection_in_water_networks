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
            pars_dim=(119),
            pars_embedding_dim=32
    ):
        super().__init__()

        self.activation = nn.LeakyReLU()
        self.hidden_neurons = hidden_neurons
        self.latent_dim = latent_dim
        self.pars_dim = pars_dim

        self.pars_embedding_layers = nn.ModuleList()
        for i in range(len(self.pars_dim)):
            self.pars_embedding_layers.append(
                nn.Embedding(
                    num_embeddings=pars_dim[i],
                    embedding_dim=pars_embedding_dim
                )
            )

        self.dense_in = nn.Linear(in_features=latent_dim,
                              out_features=self.hidden_neurons[0])
        self.batch_norm_in = nn.BatchNorm1d(self.hidden_neurons[0])

        self.hidden_neurons[0] += pars_embedding_dim * len(self.pars_dim)
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
        pars = [emb_layer(pars[:, i])
            for i, emb_layer in enumerate(self.pars_embedding_layers)]
        pars = torch.cat(pars, 1)

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

