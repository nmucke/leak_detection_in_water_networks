import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import copy


class TrainAdversarialAE():
    def __init__(self, encoder, decoder, critic,
                 encoder_optimizer, decoder_optimizer,
                 critic_optimizer,
                 with_adversarial_training,
                 latent_dim=32, n_critic=5, gamma=10, save_string='AdvAE',
                 n_epochs=100, L1_regu=None,
                 device='cpu'):

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.enc_opt = encoder_optimizer
        self.dec_opt = decoder_optimizer
        self.cri_opt = critic_optimizer
        scheduler_step_size = 5
        scheduler_gamma = 0.95

        self.enc_opt_scheduler = optim.lr_scheduler.StepLR(
                self.enc_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )
        self.dec_opt_scheduler = optim.lr_scheduler.StepLR(
                self.dec_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )
        self.cri_opt_scheduler = optim.lr_scheduler.StepLR(
                self.cri_opt,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
        )

        self.with_adversarial_training = with_adversarial_training
        self.n_epochs = n_epochs
        self.save_string = save_string

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gamma = gamma

        self.reconstruction_loss_function = nn.MSELoss()
        self.critic_loss_function = nn.BCELoss()

        self.L1_regu = L1_regu

    def train(self, dataloader, val_dataloader=None, patience=50):
        """Train adversarial autoencoder"""

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
            }, self.save_string)

        best_val_loss = np.inf
        num_epochs_no_improvement = 0
        best_encoder = copy.deepcopy(self.encoder.state_dict())
        best_decoder = copy.deepcopy(self.decoder.state_dict())
        best_critic = copy.deepcopy(self.critic.state_dict())

        recon_loss_list = []
        critic_loss_list = []
        enc_loss_list = []
        self.teacher_forcing_rate = 1.
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            recon_loss, critic_loss, enc_loss, gp = self.train_epoch(dataloader)
            self.teacher_forcing_rate = self.teacher_forcing_rate * 0.98

            # Save loss
            recon_loss_list.append(recon_loss)
            critic_loss_list.append(critic_loss)
            enc_loss_list.append(enc_loss)

            # Save generator and critic weights
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
                'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
                'critic_optimizer_state_dict': self.cri_opt.state_dict(),
                }, self.save_string)


            if val_dataloader is not None:
                val_loss = self.compute_val_loss(val_dataloader)
                print(f'val loss: {val_loss:0.5f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    best_encoder = copy.deepcopy(self.encoder.state_dict())
                    best_decoder = copy.deepcopy(self.decoder.state_dict())
                    best_critic = copy.deepcopy(self.critic.state_dict())

                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1
                    if num_epochs_no_improvement >= patience:

                        self.encoder.load_state_dict(best_encoder)
                        self.decoder.load_state_dict(best_decoder)
                        self.critic.load_state_dict(best_critic)

                        torch.save({
                            'encoder_state_dict': best_encoder,
                            'decoder_state_dict': best_decoder,
                            'critic_state_dict': best_critic,
                            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
                            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
                            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
                        }, self.save_string)
                        break

        # Save generator and critic weights

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_optimizer_state_dict': self.enc_opt.state_dict(),
            'decoder_optimizer_state_dict': self.dec_opt.state_dict(),
            'critic_optimizer_state_dict': self.cri_opt.state_dict(),
            }, self.save_string)

        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        return recon_loss_list, critic_loss_list, enc_loss_list

    def train_epoch(self, dataloader):
        """Train generator and critic for one epoch"""

        pbar = tqdm(
                enumerate(dataloader),
                total=int(len(dataloader.dataset)/dataloader.batch_size),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
        recon_loss = 0
        for bidx, (real_state, real_pars) in pbar:
            self.iii = bidx

            batch_size = real_state.size(0)
            num_steps = real_state.size(1)
            num_states = real_state.size(2)
            num_pars = real_pars.size(2)  

            real_state = real_state.reshape(
                batch_size*num_steps,
                num_states
            )
            real_pars = real_pars.reshape(
                batch_size*num_steps,
                num_pars
            )

            real_state = real_state.to(self.device)
            real_pars = real_pars.to(self.device)

            if self.with_adversarial_training:
                self.encoder.eval()
                critic_loss, gp = self.critic_train_step(
                        state=real_state,
                        pars=real_pars
                )
                self.encoder.train()
            else:
                critic_loss = 0


            #if bidx % self.n_critic == 0:
            self.critic.eval()
            recon_loss += self.train_step(
                    real_state=real_state,
                    real_pars=real_pars
            )
            self.critic.train()

            pbar.set_postfix({
                    'recon_loss': recon_loss/(bidx+1),
                    'critic_loss': critic_loss/(bidx+1),
                    'epoch': self.epoch,
                    }
            )

        self.enc_opt_scheduler.step()
        self.dec_opt_scheduler.step()
        if self.with_adversarial_training:
            self.cri_opt_scheduler.step()

        return recon_loss/(bidx+1), 1, 1,1#enc_loss, gp

    def critic_train_step(self, state, pars):
        """Train critic one step"""

        batch_size = state.size(0)

        self.cri_opt.zero_grad()

        generated_latent_data = self.encoder(state)
        true_latent_data = self.sample(batch_size)

        critic_real = self.critic(true_latent_data)
        critic_generated = self.critic(generated_latent_data)

        target_real = torch.ones_like(critic_real)
        target_generated = torch.zeros_like(critic_generated)

        cri_loss = 0.5 * self.critic_loss_function(critic_real, target_real) \
                   + 0.5 * self.critic_loss_function(critic_generated, target_generated)

        cri_loss.backward()
        self.cri_opt.step()



        '''
        a = list(self.critic.parameters())[0].clone()
        self.cri_opt.step()
        b = list(self.critic.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
        pdb.set_trace()
        '''

        return cri_loss.detach().item(),  1

    def train_step(self, real_state, real_pars):

        # Encode state
        real_latent = self.encoder(real_state)

        loss = 0

        if self.with_adversarial_training:
            # Compute critic loss
            loss_critic = self.critic_loss_function(
                    self.critic(real_latent),
                    torch.ones_like(self.critic(real_latent))
            )
            loss = loss + 1e-3*loss_critic

        # Decode state
        #decoder_input = torch.cat([real_latent, real_pars], dim=1)
        state_recon = self.decoder(real_latent, real_pars)

        # Compute reconstruction loss
        loss_recon = self.reconstruction_loss_function(state_recon, real_state)

        loss = loss + loss_recon

        if self.L1_regu is not None:
            l1_regu_encoder = 0.
            l1_regu_decoder = 0.
            for param in self.encoder.parameters():
                l1_regu_encoder += param.abs().sum()
            for param in self.decoder.parameters():
                l1_regu_decoder += param.abs().sum()

            loss = loss + self.L1_regu*(l1_regu_encoder + l1_regu_decoder)

        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)

        self.enc_opt.step()
        self.dec_opt.step()

        return loss_recon.detach().item()

    def gradient_penalty(self, data, generated_data):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device)
        epsilon = epsilon.expand_as(data)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data
        interpolation = torch.autograd.Variable(interpolation,
                                                requires_grad=True)

        interpolation_critic_score = self.critic(interpolation)

        grad_outputs = torch.ones(interpolation_critic_score.size(),
                                  device=self.device)

        gradients = torch.autograd.grad(outputs=interpolation_critic_score,
                                        inputs=interpolation,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)[0]
        gradients_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def sample(self, n_samples):
        """Generate n_samples fake samples"""
        return torch.randn(n_samples, self.latent_dim).to(self.device)

    def compute_val_loss(self, val_dataloader):
        """Compute validation loss"""
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0
        for batch_idx, (state, pars) in enumerate(val_dataloader):

            batch_size = state.size(0)
            num_steps = state.size(1)
            num_states = state.size(2)
            num_pars = pars.size(2)  

            state = state.reshape(
                batch_size*num_steps,
                num_states
            )
            pars = pars.reshape(
                batch_size*num_steps,
                num_pars
            )

            state = state.to(self.device)
            pars = pars.to(self.device)
            val_loss += self.reconstruction_loss_function(
                    self.decoder(self.encoder(state), pars), state).detach().item()
        self.encoder.train()
        self.decoder.train()
        return val_loss/(batch_idx+1)
