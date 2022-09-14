import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import hamiltorch as ht
import sys
import os
import ray
from torchquad import MonteCarlo, set_up_backend

#set_up_backend("torch", data_type="float32")


@ray.remote
def compute_reconstruction_error(
        leak_location,
        true_state,
        variational_inference,
        variational_minumum=True,
        HMC=False
    ):

    pars = torch.tensor([[leak_location]], dtype=torch.int64)

    if HMC:
        reconstruction, reconstruction_std, latent_state, latent_std = \
            variational_inference.compute_HMC_statistics(
                observations=observation_operator.get_observations(true_state),
                pars=torch.tensor([[i]], dtype=torch.int64),
                num_samples=1000,
        )
        reconstruction = reconstruction.unsqueeze(0)
        reconstruction_std = reconstruction_std.unsqueeze(0)
        latent_state = latent_state.unsqueeze(0)

    else:
        if variational_minumum:
            latent_state, reconstruction = variational_inference.compute_variational_minimum(
                    observations=observation_operator.get_observations(true_state),
                    pars=pars,
                    num_iterations=2500
            )
        else:
            latent_state, reconstruction = variational_inference.compute_encoder_decoder_reconstruction(
                    true_state=true_state.unsqueeze(0),
                    pars=pars,
            )

    log_posterior = variational_inference.log_posterior(
            observations=observation_operator.get_observations(true_state),
            predicted_observations=observation_operator.get_observations(reconstruction[0]),
            latent_state=latent_state
    )


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





class VariationalInference():
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


    def compute_HMC_statistics(
            self,
            observations,
            pars,
            num_samples=100,
        ):

        HMC_params = {'num_samples': 2*num_samples,
                      'step_size': 1.,
                      'num_steps_per_sample': 5,
                      'burn': num_samples,
                      'integrator': ht.Integrator.IMPLICIT,
                      'sampler': ht.Sampler.HMC_NUTS,
                      'desired_accept_rate': 0.3}

        posterior = lambda z: self.compute_log_posterior(
                observations=observations,
                latent_state=z.unsqueeze(0),
                pars=pars
        )
        latent_state, _ = self.compute_variational_minimum(
                observations=observations,
                pars=pars,
                num_iterations=2500,
                tol=1e-4
        )

        blockPrint()
        z_samples = ht.sample(
                log_prob_func=posterior,
                params_init=latent_state.squeeze(0),
                **HMC_params
        )
        enablePrint()

        z_samples = torch.stack(z_samples)

        state_samples = self.decoder(z_samples, pars.repeat(z_samples.shape[0], 1))

        return state_samples.mean(dim=0).detach(), state_samples.std(dim=0).detach(),\
               z_samples.mean(dim=0).detach(), z_samples.std(dim=0).detach()

    def compute_log_posterior(self, observations, latent_state, pars):

        reconstruction = self.decoder(latent_state, pars)
        reconstruction_observations = self.observation_operator.get_observations(
                reconstruction[0]
        )

        log_pos = self.log_posterior(
                observations=observations,
                predicted_observations=reconstruction_observations,
                latent_state=latent_state
        )

        return log_pos



    def log_posterior(self, observations, predicted_observations, latent_state):

        log_likelihood = torch.distributions.Normal(
                predicted_observations,
                torch.ones_like(predicted_observations) * self.obs_std
        ).log_prob(observations).sum()

        log_prior = torch.distributions.Normal(
                torch.zeros_like(latent_state),
                torch.ones_like(latent_state)
        ).log_prob(latent_state).sum()

        return log_likelihood + log_prior

    def compute_relative_error(self, state_1, state_2):

        if state_1.shape[0] == self.num_pipes + self.num_nodes:
            flow_rate_1 = state_1[:self.num_pipes]
            flow_rate_2 = state_2[:self.num_pipes]
            head_1 = state_1[self.num_pipes:]
            head_2 = state_2[self.num_pipes:]
        else:
            flow_rate_1 = state_1[:len(self.observation_operator.pipe_observation_labels)]
            flow_rate_2 = state_2[:len(self.observation_operator.pipe_observation_labels)]
            head_1 = state_1[-len(self.observation_operator.node_observations_labels):]
            head_2 = state_2[-len(self.observation_operator.node_observations_labels):]


        e_head = torch.linalg.norm(head_1 - head_2) / torch.linalg.norm(head_1)
        e_flow_rate = torch.linalg.norm(flow_rate_1 - flow_rate_2) / torch.linalg.norm(flow_rate_1)

        return e_head + e_flow_rate

    def compute_variational_minimum(
            self,
            observations,
            pars,
            num_iterations=100,
            tol=1e-5
    ):

        z_min = torch.zeros(
                (1, self.latent_dim),
                dtype=torch.get_default_dtype(),
                requires_grad=True
        )
        optimizer = optim.Adam([z_min], lr=.1)
        #optimizer = optim.LBFGS([z_min], lr=1.)
        dz = 1e8
        '''
        def closure():
            optimizer.zero_grad()
            reconstruction = self.decoder(z_min, pars)
            reconstruction_observations = \
                self.observation_operator.get_observations(
                    reconstruction[0]
                )
            loss = self.loss_function(
                    reconstruction_observations, observations
            )
            loss.backward()
            return loss

        for i in range(num_iterations):
            z_old = z_min.clone()
            optimizer.step(closure)
            dz = torch.norm(z_min - z_old)


        '''
        for i in range(num_iterations):
            z_old = z_min.clone()

            optimizer.zero_grad()
            reconstruction = self.decoder(z_min, pars)
            reconstruction_observations = self.observation_operator.get_observations(reconstruction[0])
            #loss = self.loss_function(
            #    reconstruction_observations, observations
            #)

            loss = -self.log_posterior(observations, reconstruction_observations, z_min)

            loss.backward()
            optimizer.step()

            dz = torch.norm(z_min - z_old)
            if dz < tol:
                #print(f"Converged after {i} iterations")
                break

        reconstruction = self.decoder(z_min, pars)

        return z_min.detach(), reconstruction.detach()

    def compute_encoder_decoder_reconstruction(self, true_state, pars):

        z = self.encoder(true_state)
        reconstructed_state = self.decoder(z, pars)

        return z.detach(), reconstructed_state.detach()


class VariationalInferenceGAN():
    def __init__(
            self,
            observation_operator,
            generator,
    ):

        self.observation_operator = observation_operator
        self.generator = generator
        self.latent_dim = generator.latent_dim
        self.loss_function = nn.MSELoss()
        self.num_pipes = observation_operator.num_pipes
        self.num_nodes = observation_operator.num_nodes

    def compute_relative_error(self, state_1, state_2):

        if state_1.shape[0] == self.num_pipes + self.num_nodes:
            flow_rate_1 = state_1[:self.num_pipes]
            flow_rate_2 = state_2[:self.num_pipes]
            head_1 = state_1[self.num_pipes:]
            head_2 = state_2[self.num_pipes:]
        else:
            flow_rate_1 = state_1[:len(self.observation_operator.pipe_observation_labels)]
            flow_rate_2 = state_2[:len(self.observation_operator.pipe_observation_labels)]
            head_1 = state_1[-len(self.observation_operator.node_observations_labels):]
            head_2 = state_2[-len(self.observation_operator.node_observations_labels):]


        e_head = torch.linalg.norm(head_1 - head_2) / torch.linalg.norm(head_1)
        e_flow_rate = torch.linalg.norm(flow_rate_1 - flow_rate_2) / torch.linalg.norm(flow_rate_1)

        return e_head + e_flow_rate

    def compute_variational_minimum(
            self,
            observations,
            num_iterations=100,
            tol=1e-6
    ):

        z_min = torch.zeros((1, self.latent_dim), requires_grad=True)
        optimizer = optim.Adam([z_min], lr=1.)
        #optimizer = optim.LBFGS([z_min], lr=1.)
        '''
        dz = 1e8
        def closure():
            optimizer.zero_grad()
            reconstruction = self.generator(z_min)
            reconstruction = reconstruction[:, :self.num_pipes+self.num_nodes]
            reconstruction_observations = \
                self.observation_operator.get_observations(
                    reconstruction[0])
            loss = self.loss_function(
                    reconstruction_observations, observations
            )
            loss.backward()
            return loss

        for i in range(num_iterations):
            z_old = z_min.clone()
            optimizer.step(closure)
            dz = torch.norm(z_min - z_old)

            #if dz < tol:
            #    print("Converged after {} iterations".format(i))
            #    break

        '''
        for i in range(num_iterations):
            z_old = z_min.clone()

            optimizer.zero_grad()
            reconstruction = self.generator(z_min)
            reconstruction_observations = reconstruction[:, :self.num_pipes+self.num_nodes]
            reconstruction_observations = self.observation_operator.get_observations(reconstruction_observations[0])
            loss = self.loss_function(
                reconstruction_observations, observations
            )
            loss.backward()
            optimizer.step()

        reconstruction = self.generator(z_min)

        return z_min.detach(), reconstruction.detach()
