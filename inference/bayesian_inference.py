import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import hamiltorch as ht
import sys
import os
from torchquad import MonteCarlo, set_up_backend
import ray

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class ImportanceSampling():
    def __init__(
            self,
    ):

        self.prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(10),
                covariance_matrix=torch.eye(10)
            )

    def integrate(self, integrand_fun, dim=10, N=10, seed=1, ):

        self.prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(dim),
                covariance_matrix=torch.eye(dim)
            )
        
        sample_values = self.prior.sample((N,))
        log_prob = self.prior.log_prob(sample_values)
        prob = torch.exp(log_prob).unsqueeze(1)

        integrand_values = integrand_fun(sample_values)

        integral = torch.sum(integrand_values / prob, dim=0) / N

        return integral

class BayesianInference():
    def __init__(
            self,
            observation_operator,
            decoder,
            encoder,
            obs_std=0.1,
    ):

        self.observation_operator = observation_operator
        self.decoder = decoder
        self.encoder = encoder
        self.latent_dim = decoder.latent_dim
        self.loss_function = nn.MSELoss()#nn.L1Loss()#
        self.num_pipes = observation_operator.num_pipes
        self.num_nodes = observation_operator.num_nodes
        self.pi = torch.tensor(3.14159265359)

        self.obs_std = obs_std

    def integrand(self, observations, latent_state, pars):

        dot_prod = torch.bmm(latent_state.unsqueeze(1), latent_state.unsqueeze(2))
        dot_prod = dot_prod.squeeze(1)
        prior = 1/torch.sqrt((2*self.pi)**self.latent_dim) * torch.exp(-0.5 * dot_prod)

        pred = self.decoder(latent_state, pars)
        pred_obs = self.observation_operator.get_observations(pred)

        diff = observations - pred_obs
        diff = torch.bmm(diff.unsqueeze(1), diff.unsqueeze(2))
        diff = diff.squeeze(1)

        likelihood = 1/torch.sqrt((2*self.pi*self.obs_std)**observations.shape[0]) * \
                     torch.exp(-0.5 * diff/self.obs_std/self.obs_std)

        return prior * likelihood

    def compute_p_y_given_c(
            self,
            observations,
            pars,
            num_samples=10000,
        ):

        #mc = MonteCarlo()
        mc = ImportanceSampling()


        # Compute the function integral by sampling 10000 points over domain
        p_y_given_c = 0
        for i in range(10):

            integrand_fun = lambda z: self.integrand(
                    observations=observations,
                    latent_state=z,
                    pars=pars.repeat(num_samples//10, 1)
            )

            p_y_given_c += mc.integrate(
                    integrand_fun,
                    dim=self.latent_dim,
                    N=num_samples//10,
                    #integration_domain=[[-2., 2.]] * self.latent_dim,
                    seed=i,
                    #backend="torch",
            ).detach().item()

        return p_y_given_c/10

    @ray.remote
    def compute_leak_location_posterior(
            self,
            leak_location,
            time_stamp,
            observations,
            num_samples=10000,
        ):

        pars = torch.tensor([[leak_location, time_stamp]], dtype=torch.int32)
        #pars = torch.tensor([[leak_location]], dtype=torch.int32)


        posterior = self.compute_p_y_given_c(
                observations=observations,
                pars=pars,
                num_samples=num_samples,
        )

        return posterior