import torchvision.transforms as transforms
import torch
import pdb
from tqdm import tqdm
import os
import wntr
import matplotlib.pyplot as plt

class TrainGAN():
    def __init__(self, generator, critic, generator_optimizer, critic_optimizer,
                 latent_dim=100, n_critic=5, gamma=10, n_epochs=100,
                 save_string=None, physics_loss=None, device='cpu', transformer=None):

        self.device = device
        self.generator = generator
        self.critic = critic
        self.G_opt = generator_optimizer
        self.C_opt = critic_optimizer

        self.generator.train(mode=True)
        self.critic.train(mode=True)

        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.save_string = save_string

        self.physics_loss = physics_loss

        self.transformer = transformer


    def physics_loss_function(self, data):

        leak_demand = data[:, 66]
        demand_pred = torch.matmul(-self.incidence_mat.T,
                                   data[:, 0:34].T)
        #loss = leak_demand + demand_pred.sum(dim=0)

        return demand_pred.T

    def get_demand(self, data):
        demand_pred = torch.matmul(-self.incidence_mat.T,
                                   data[:, 0:34].T)
        return demand_pred.T

    def train(self, dataloader):
        """Train generator and critic"""

        generator_loss = []
        critic_loss = []
        gradient_penalty = []
        for epoch in range(1, self.n_epochs + 1):

            # Train one step
            g_loss, c_loss, grad_penalty = self.train_epoch(dataloader)

            print(f'Epoch: {epoch}, g_loss: {g_loss:.3f},', end=' ')
            print(f'c_loss: {c_loss:.3f}, grad_penalty: {grad_penalty:.3f}')

            # Save loss
            generator_loss.append(g_loss)
            critic_loss.append(c_loss)
            gradient_penalty.append(grad_penalty)

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.G_opt.state_dict(),
                'critic_optimizer_state_dict': self.C_opt.state_dict(),
                }, self.save_string)

        # Save generator and critic weights
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer_state_dict': self.G_opt.state_dict(),
            'critic_optimizer_state_dict': self.C_opt.state_dict(),
            }, self.save_string)

        self.generator.train(mode=False)
        self.critic.train(mode=False)

        return generator_loss, critic_loss, gradient_penalty

    def train_epoch(self, data_loader):
        """Train generator and critic for one epoch"""

        for bidx, (real_data, pars) in tqdm(enumerate(data_loader),
                 total=int(len(data_loader.dataset)/data_loader.batch_size)):
            current_batch_size = len(real_data)

            real_data = torch.cat([real_data, pars], dim=1)
            real_data = real_data.to(self.device)

            c_loss, grad_penalty = self.critic_train_step(real_data)

            if bidx % self.n_critic == 0:
                g_loss = self.generator_train_step(current_batch_size)

        return g_loss, c_loss, grad_penalty

    def critic_train_step(self, data):
        """Train critic one step"""

        self.C_opt.zero_grad()
        batch_size = data.size(0)
        generated_data = self.sample(batch_size)

        if self.physics_loss:

            generated_demand = self.get_demand(generated_data)
            generated_critic_input = torch.cat(
                    [generated_data, generated_demand], dim=1)

            real_demand = self.get_demand(data)
            real_critic_input = torch.cat([data, real_demand], dim=1)

            #data = self.transformer.min_max_inverse_transform(data.detach().cpu())
            #lol = torch.matmul(self.incidence_mat.cpu().T,data[0,-34:])
            #pdb.set_trace()

            grad_penalty = self.gradient_penalty(real_critic_input,
                                                 generated_critic_input)

            c_loss = self.critic(generated_critic_input).mean() \
                     - self.critic(real_critic_input).mean() + grad_penalty
        else:
            grad_penalty = self.gradient_penalty(data,
                                                 generated_data)
            c_loss = self.critic(generated_data).mean() \
                     - self.critic(data).mean() + grad_penalty
        c_loss.backward()
        self.C_opt.step()

        return c_loss.detach().item(), grad_penalty.detach().item()

    def generator_train_step(self, batch_size):
        """Train generator one step"""

        self.G_opt.zero_grad()
        generated_data = self.sample(batch_size)
        if self.physics_loss:

            generated_demand = self.get_demand(generated_data)
            generated_critic_input = torch.cat([generated_data, generated_demand], dim=1)

            g_loss = - self.critic(generated_critic_input).mean()

        else:
            g_loss = - self.critic(generated_data).mean()
        g_loss.backward()
        self.G_opt.step()

        return g_loss.detach().item()

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

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def sample(self, n_samples):
        """Generate n_samples fake samples"""
        return self.generator(torch.randn(n_samples,
                              self.latent_dim).to(self.device))





