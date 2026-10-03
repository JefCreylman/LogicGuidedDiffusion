import torch
from torch import nn

class NoiseEncoder:
    def __init__(self, sigmas, num_timesteps=-1):
        self.num_timesteps = num_timesteps if num_timesteps > 0 else len(sigmas)
        self.sigmas = sigmas

    def forward(self, x, l):
        noise = torch.randn_like(x)
        
        sigma_t = self.sigmas[l]

        if isinstance(l, torch.Tensor):               #GEMINI: should work for both batched and non-batched now?
            sigma_t = sigma_t.reshape(-1, 1, 1, 1)
        else:
            sigma_t = sigma_t.reshape(1, 1, 1, 1)

        x = x + sigma_t * noise

        return x, noise, sigma_t
            


class QuadNISTDiffusion(nn.Module):

    def __init__(self, sigmas, num_timesteps=-1):
        super().__init__()

        self.num_timesteps = num_timesteps if num_timesteps > 0 else len(sigmas)

        if self.num_timesteps > len(sigmas):
            raise Exception('Not enough sigmas for this number of timesteps')

        self.sigmas = sigmas

        self.encoder = NoiseEncoder(sigmas=self.sigmas, num_timesteps=self.num_timesteps)

    def forward(self, x, l):
        pass

    def calculate_loss(self, x):
        batch_size = x.shape[0]
        l = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

        noisy_x, noise, sigma_t = self.encoder(x, l)

        #Taking lambda = sigma**2 as stated in Generative Modeling by Estimating Gradients of the Data Distribution gives following short form
        loss =  0.5 * torch.mean((self(noisy_x, l) * sigma_t + noise) ** 2)

        return loss

    def sample(self, etas, budgets, noise):
        x = noise
        for l in reversed(range(self.num_timesteps)):
            for n in range(budgets[l]):
                e = torch.randn_like(x)
                x = x + etas[l]*self(x, l) + torch.sqrt(2*etas[l])*e                #TODO: Change to add CG!
        return x
