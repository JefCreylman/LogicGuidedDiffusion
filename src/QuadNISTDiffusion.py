import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NoiseEncoder(nn.Module):
    def __init__(self, sigmas, num_timesteps=-1):
        super().__init__()
        self.num_timesteps = num_timesteps if num_timesteps > 0 else len(sigmas)
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))

    def forward(self, x, l):
        noise = torch.randn_like(x)
        
        sigma_t = self.sigmas[l]

        if isinstance(l, torch.Tensor):               #GEMINI: should work for both batched and non-batched now?
            sigma_t = sigma_t.reshape(-1, 1, 1, 1)
        else:
            sigma_t = sigma_t.reshape(1, 1, 1, 1)

        x = x + sigma_t * noise

        return x, noise, sigma_t
            
'''Gemini'''
class ConditionalInstanceNorm2d(nn.Module):
    """Injects noise-level scale and shift parameters into the feature maps."""
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        
        # Initialize embeddings stably
        self.embed.weight.data[:, :num_features].normal_(1.0, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, l):
        out = self.instance_norm(x)
        gamma, beta = self.embed(l).chunk(2, dim=-1)
        gamma = gamma.reshape(-1, self.num_features, 1, 1)
        beta = beta.reshape(-1, self.num_features, 1, 1)
        return out * gamma + beta

'''Gemini'''
class ConditionalResBlock(nn.Module):
    """Residual Block with Conditional Instance Normalization for score matching."""
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalInstanceNorm2d(out_channels, num_classes)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, l):
        h = F.elu(self.norm1(self.conv1(x), l))
        h = self.norm2(self.conv2(h), l)
        return F.elu(h + self.shortcut(x))

'''Gemini'''
class MNISTScoreNet(nn.Module):
    """A lightweight U-Net architecture built for MNIST score matching."""
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        # Encoder / Downsampling
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = ConditionalResBlock(64, 64, num_classes)
        self.down2 = ConditionalResBlock(64, 128, num_classes)
        self.down3 = ConditionalResBlock(128, 256, num_classes)
        
        # Middle / Bottleneck
        self.mid = ConditionalResBlock(256, 256, num_classes)
        
        # Decoder / Upsampling
        self.up3 = ConditionalResBlock(256 + 256, 128, num_classes)
        self.up2 = ConditionalResBlock(128 + 128, 64, num_classes)
        self.up1 = ConditionalResBlock(64 + 64, 64, num_classes)
        
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, l):
        # Initial projection
        h1 = self.init_conv(x) 
        
        # Downward paths (with pooling)
        d1 = self.down1(h1, l) 
        pool1 = F.avg_pool2d(d1, 2) 
        
        d2 = self.down2(pool1, l) 
        pool2 = F.avg_pool2d(d2, 2) 

        d3 = self.down3(pool2, l)
        pool3 = F.avg_pool2d(d3, 2)
        
        # Bottleneck
        m = self.mid(pool3, l)
        
        # Upward paths (with interpolation and skip connections)
        up_3 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)   # Out: 14x14
        u3 = self.up3(torch.cat([up_3, d3], dim=1), l)
                      
        up_2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([up_2, d2], dim=1), l) 
        
        up_1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([up_1, d1], dim=1), l) 
        
        return self.final_conv(u1)


class QuadNISTDiffusion(nn.Module):
    config = {}

    def __init__(self, sigmas, num_timesteps=-1):
        super().__init__()

        self.num_timesteps = num_timesteps if num_timesteps > 0 else len(sigmas)

        if self.num_timesteps > len(sigmas):
            raise Exception('Not enough sigmas for this number of timesteps')

        self.sigmas = sigmas

        self.encoder = NoiseEncoder(sigmas=self.sigmas, num_timesteps=self.num_timesteps)

        self.score_net = MNISTScoreNet(num_classes=self.num_timesteps, in_channels=1)

    '''Gemini'''
    def forward(self, x, l):
        if not isinstance(l, torch.Tensor):
            l = torch.tensor([l], device=x.device, dtype=torch.long)
        elif l.dim() == 0:
            l = l.unsqueeze(0)

        l = l.to(x.device)


        # Standard NCSN output scaling: score = net(x, l) / sigma
        sigma_t = self.encoder.sigmas[l]

        # FIXED: Safer dimension evaluation checks for batch dimensions vs. sing
        if sigma_t.dim() == 1 and sigma_t.numel() > 1:
            sigma_t = sigma_t.reshape(-1, 1, 1, 1)
        else:
            sigma_t = sigma_t.reshape(1, 1, 1, 1)

        return self.score_net(x, l) / sigma_t

    def calculate_loss(self, x):
        '''
        Calculate loss without cluttering training script
        '''
        batch_size = x.shape[0]
        l = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

        noisy_x, noise, sigma_t = self.encoder(x, l)

        # Use score_net directly to circumvent the unnecessary if/else and device checks in forward() and reduce training times.
        # It should safe to do so as l is constructed above to already pass all checks
        score_pred = self.score_net(noisy_x, l) / sigma_t
        #Taking lambda = sigma**2 as stated in Generative Modeling by Estimating Gradients of the Data Distribution gives following short form
        loss =  0.5 * torch.mean((score_pred * sigma_t + noise) ** 2)

        return loss

    def sample(self, etas, budgets, noise, classifier=None, omega=1, target=1):
        '''
        Sample the diffusion model using LD

        Args:
            etas:
                List of stepsizes per noise level
            budgets:
                Amount of steps per noise level
            noise:
                The seed noise for sampling
            classifier (nn.Module | None):
                Classifier used for CG. If None, no guidance is used
            omega:
                Scalar modulating the influence of the classifier. Unused if classifier = None
            target:
                Guidance target. Unused if classifier = None

        '''
        x = noise

        guided = isinstance(classifier, nn.Module)

        print("Guided" if guided else "Unguided")
        noiseAware = True

        for l in range(self.num_timesteps):
            sigma_tensor = torch.full((x.shape[0], 1), self.sigmas[l], dtype=torch.float, device=x.device)
            l_tensor = torch.full((x.shape[0],), l, dtype=torch.long, device=x.device)
            for _ in range(budgets[l]):
                e = torch.randn_like(x)
                if guided:
                    x.requires_grad_(True)

                    with torch.enable_grad():
                        try:
                            loss = classifier.calculate_loss(x.flatten(1), target, sigma=sigma_tensor, reduction="sum")
                        except:
                            noiseAware = False
                            loss = classifier.calculate_loss(x.flatten(1), target, reduction="sum")

                    grad = torch.autograd.grad(loss, x)[0]
                    x = x.detach()

                    score = self(x, l_tensor) - omega * grad
                    x = x + etas[l]*score + np.sqrt(2*etas[l])*e
                else:
                    x = x + etas[l]*self(x, l_tensor) + np.sqrt(2*etas[l])*e

        if not noiseAware:
            print("Had to use noise unaware implementation")
        return x

    def get_checkpoint(self):
        '''
        Get correct checkpoint layout for saving
        '''
        return {
            "state_dict": self.state_dict(),
            "sigmas": self.sigmas,
            "num_timesteps": self.num_timesteps
            }

    def load_model_from_checkpoint(pt):
        '''
        Load a model from a loaded checkpoint file
        '''
        model = QuadNISTDiffusion(pt["sigmas"], pt["num_timesteps"])
        model.load_state_dict(pt["state_dict"])
        return model
