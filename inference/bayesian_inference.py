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
import hamiltorch

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Hamiltonian_MC():
    def __init__(
            self,
            HMC_params=None
    ):

        if HMC_params is None:
            self.HMC_params = {
                'step_size': 1.,
                'num_steps_per_sample': 5,
                'integrator': hamiltorch.Integrator.IMPLICIT,
                'sampler': hamiltorch.Sampler.HMC_NUTS,
                'desired_accept_rate': 0.3
                }
        else:
            self.HMC_params = HMC_params

    def integrate(self, prior, log_likelihood, num_samples, num_burn_in_samples):

        self.HMC_params['num_samples'] = num_samples
        self.HMC_params['burn'] = num_burn_in_samples

        log_posterior = lambda z: prior.log_prob(z).sum() + log_likelihood(z).sum()

        z_init = prior.sample()

        z_samples = hamiltorch.sample(
            log_prob_func=log_posterior,
            params_init=z_init,
            **self.HMC_params
            )
        z_samples = torch.stack(z_samples)

        log_posterior_samples = log_posterior(z_samples)
        
        return np.mean(z_samples)



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
        #log_prob = self.prior.log_prob(sample_values)
        #prob = torch.exp(log_prob).unsqueeze(1)

        integrand_values = integrand_fun(sample_values)
        integral = torch.mean(integrand_values)
        #integral = torch.sum(integrand_values / prob, dim=0) / N

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

        self.num_obs = len(observation_operator.pipe_obs_idx) + \
            len(observation_operator.node_obs_idx)

        self.likelihood_prob = torch.distributions.Normal(
            torch.zeros(self.num_obs),
            obs_std * torch.ones(self.num_obs)
        )

        self.prior = torch.distributions.Normal(
            torch.zeros(self.latent_dim),
            torch.ones(self.latent_dim)
        )

    def integrand(self, observations, latent_state, pars):

        #dot_prod = torch.bmm(latent_state.unsqueeze(1), latent_state.unsqueeze(2))
        #dot_prod = dot_prod.squeeze(1)
        #prior = 1/torch.sqrt((2*self.pi)**self.latent_dim) * torch.exp(-0.5 * dot_prod)

        pred = self.decoder(latent_state, pars)
        pred_obs = self.observation_operator.get_observations(pred)

        diff = observations - pred_obs
        diff = torch.bmm(diff.unsqueeze(1), diff.unsqueeze(2))
        diff = diff.squeeze(1)

        likelihood = 1/torch.sqrt((2*self.pi*self.obs_std)**observations.shape[0]) * \
                     torch.exp(-0.5 * diff/self.obs_std/self.obs_std)

        return likelihood#prior * likelihood

    def log_likelihood(self, z, pars, observations):
        pred = self.decoder(z.unsqueeze(0), pars)
        pred_obs = self.observation_operator.get_observations(pred)
        diff = observations - pred_obs
        log_prob = self.likelihood_prob.log_prob(diff.squeeze(0))
        return log_prob

    def compute_p_y_given_c(
            self,
            observations,
            pars,
            num_samples=10000,
            integration_method="monte_carlo",
        ):

        if integration_method == "monte_carlo":
            mc = MonteCarlo()
        if integration_method == "importance_sampling":
            mc = ImportanceSampling()
        if integration_method == "hamiltonian":
            mc = Hamiltonian_MC()


        # Compute the function integral by sampling 10000 points over domain
        p_y_given_c = 0
        num_batches = 5
        for i in range(num_batches):

            integrand_fun = lambda z: self.integrand(
                    observations=observations,
                    latent_state=z,
                    pars=pars.repeat(num_samples//num_batches, 1)
            )

            if integration_method == "monte_carlo":
                p_y_given_c += mc.integrate(
                    integrand_fun,
                    dim=self.latent_dim,
                    N=num_samples//num_batches,
                    integration_domain=[[-4., 4.]] * self.latent_dim,
                    seed=i,
                    backend="torch",
                ).detach().item()
            if integration_method == "importance_sampling":
                p_y_given_c += mc.integrate(
                    integrand_fun,
                    dim=self.latent_dim,
                    N=num_samples//num_batches,
                    seed=i,
                ).detach().item()
            if integration_method == "hamiltonian":
                log_likelihood = lambda z: self.log_likelihood(
                    z=z,
                    pars=pars,
                    observations=observations
                    )
                num_burn_in_samples = num_samples//num_batches//2
                p_y_given_c += mc.integrate(
                    prior=self.prior,
                    log_likelihood=log_likelihood,
                    num_samples=num_samples//num_batches + num_burn_in_samples,
                    num_burn_in_samples=num_burn_in_samples,
                ).detach().item()
        return p_y_given_c/num_batches

@ray.remote
def compute_leak_location_posterior(
        bayesian_inference_object,
        leak_location,
        time_stamp,
        observations,
        num_samples=10000,
        integration_method="monte_carlo",
    ):

    if time_stamp is None:
        pars = torch.tensor([[leak_location]], dtype=torch.int32)
    else:
        pars = torch.tensor([[leak_location, time_stamp]], dtype=torch.int32)
    #pars = torch.tensor([[leak_location]], dtype=torch.int32)

    posterior = bayesian_inference_object.compute_p_y_given_c(
            observations=observations,
            pars=pars,
            num_samples=num_samples,
            integration_method=integration_method,
    )

    return posterior