import torch.nn.functional as F
import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, d_hidden, d_out,num_step):
        super().__init__()
        self.num_step = num_step
        # 时间步嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        # 主干网络
        self.mlp = nn.Sequential(
            nn.Linear(d_in + d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x, timesteps):
        # 时间步处理
        t = timesteps.expand(x.size(0)).float().unsqueeze(-1)  / self.num_step
        t_emb = self.time_embed(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.mlp(x)

class GaussianMultinomialDiffusion(nn.Module):
    def __init__(self, num_numerical_features, denoise_fn, num_timesteps, device='cpu'):
        super(GaussianMultinomialDiffusion, self).__init__()
        self.num_numerical_features = num_numerical_features
        self.denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        alphas = 1. - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(alphas, dim=0)).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - torch.cumprod(alphas, dim=0)).to(device)

    def extract(self, arr, timesteps, broadcast_shape):
        res = arr.gather(-1, timesteps).float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise

    def forward(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        x_noisy = self.q_sample(x, t, noise)
        model_out = self.denoise_fn(x_noisy, t)
        return model_out, noise

    def compute_loss(self, x, t):
        model_out, noise = self.forward(x, t)
        return F.mse_loss(model_out, noise)

    def sample(self, num_samples):
        samples = torch.randn((num_samples, self.num_numerical_features)).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            samples = self.reverse_diffusion_step(samples, t)
        return samples

    def extract2(self, arr, timesteps, broadcast_shape):

        res = arr.gather(0, timesteps)  # 将timesteps改为张量形式传入
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def reverse_diffusion_step(self, x, t):
        t = torch.tensor([t], device=x.device)
        betas_t = self.extract2(self.betas, t, x.shape)
        one_minus_alphas_bar_sqrt_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        eps_theta = self.denoise_fn(x, t)
        coeff = betas_t / one_minus_alphas_bar_sqrt_t
        mean = (1 / (1 - betas_t).sqrt()) * (x - coeff * eps_theta)
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        sigma_t = betas_t.sqrt()
        return mean + sigma_t * z






