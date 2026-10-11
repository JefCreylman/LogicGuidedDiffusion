import torch
from torch import nn

class QuadNISTClassifier(nn.Module):
    '''
    QuadNISTClassifier implementation adapted from https://medium.com/@amdnewaz/in-this-article-well-walk-through-building-a-simple-neural-network-using-pytorch-to-classify-211b33ba5c62
    '''

    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(28*28*4, 512)
        self.relu = nn.ReLU()
        self.outputD = nn.Linear(512, 10)
        self.outputQ = nn.Linear(512, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.flatten(1)

        x = self.hidden(x)
        x = self.relu(x)
        
        d = self.outputD(x)
        q = self.outputQ(x)

        d = self.softmax(d)
        q = self.softmax(q)

        return {'digit': d, 'quadrant': q}

    def calculate_loss(self, x, label, reduction="mean"):
        '''
        Calculate loss without cluttering training script
        '''
        output = self(x)
        loss_fn = nn.NLLLoss(reduction=reduction)

        loss_d = loss_fn(output['digit'], label['digit'])
        loss_q = loss_fn(output['quadrant'], label['quadrant'] - 1)

        total_loss = loss_d + loss_q

        return total_loss

    def get_checkpoint(self):
        '''
        Get correct checkpoint layout for saving
        '''
        return {
            "state_dict": self.state_dict(),
            }

    def load_model_from_checkpoint(pt):
        '''
        Load a model from a loaded checkpoint file
        '''
        model = QuadNISTClassifier()
        model.load_state_dict(pt["state_dict"])
        return model


class NoiseAwareQuadNISTClassifier(nn.Module):
    '''
    QuadNISTClassifier implementation adapted from https://medium.com/@amdnewaz/in-this-article-well-walk-through-building-a-simple-neural-network-using-pytorch-to-classify-211b33ba5c62
    
    Args:
        sigmas (None | list):
            You should add sigmas that were used to train here, just to make sure the diffusion and classifier sigmas correspond.
        num_timesteps (None | int):
            You should add sigmas that were used to train here, just to make sure the diffusion and classifier sigmas correspond. None uses len(sigmas) if sigmas was given.
    '''

    def __init__(self, sigmas=None, num_timesteps=None):
        super().__init__()
        
        self.hidden = nn.Linear(28*28*4, 512)
        self.relu = nn.ReLU()

        self.embed_sigma = nn.Sequential(nn.Linear(1,128), nn.ReLU(), nn.Linear(128,512), nn.ReLU())

        self.outputD = nn.Linear(512, 10)
        self.outputQ = nn.Linear(512, 4)
        self.softmax = nn.LogSoftmax(dim=1)

        self.sigmas = sigmas
        self.num_timesteps = num_timesteps if num_timesteps != None or sigmas == None else len(sigmas)

    def forward(self, x, sigma=None):
        x = x.flatten(1)

        x = self.hidden(x)
        x = self.relu(x)

        if sigma is not None:
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(1)

            sigma = torch.log(sigma.float() + 1e-8)     #add 1e-8 to avoid log(0)
            sigma = self.embed_sigma(sigma)
            x = x + sigma
        
        d = self.outputD(x)
        q = self.outputQ(x)

        d = self.softmax(d)
        q = self.softmax(q)

        return {'digit': d, 'quadrant': q}

    def calculate_loss(self, x, label, sigma=None, reduction="mean"):
        '''
        Function for calculating losses without cluttering training script
        '''
        output = self(x, sigma)
        loss_fn = nn.NLLLoss(reduction=reduction)

        loss_d = loss_fn(output['digit'], label['digit'])
        loss_q = loss_fn(output['quadrant'], label['quadrant'] - 1)

        total_loss = loss_d + loss_q

        return total_loss

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
        model = NoiseAwareQuadNISTClassifier(pt["sigmas"], pt["num_timesteps"])
        model.load_state_dict(pt["state_dict"])
        return model