import torch
from torch import nn

class LargeLeNet5(nn.Module):
    '''
    Adaptation of LeNet5 for 56x56 images
    '''
    def __init__(self, num_classes=10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # C1: [N, 6, 52, 52]
            nn.Conv2d(3, 6, 5),
            nn.Tanh(),

            # C2: [N, 6, 26, 26]
            nn.AvgPool2d(2),
            
            # C3: [N, 16, 22, 22]
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),

            # S4: [N, 16, 11, 11]
            nn.AvgPool2d(2),

            # This is kinda C5
            # For the original implementation the below linear version and Conv2D are equivalent.
            # This is not the case 56x56 images, but I'll leave it this way anyway since it is more flexible methinks
            nn.Flatten(),
            nn.Linear(16*11*11, 120),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            # F6
            nn.Linear(120, 84),
            nn.Tanh(),

            # Output
            nn.Linear(84, num_classes)
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits

    def calculate_loss(self, x, label, reduction="mean"):
        '''
        Calculate loss without cluttering training script.
        Only does a digit loss cause I feel like this makes more sense 'image realism'-wise, as that's what we're doing with this anyway.
        '''
        output = self(x)
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)

        loss_d = loss_fn(output, label['digit'])

        total_loss = loss_d

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
        model = LargeLeNet5()
        model.load_state_dict(pt["state_dict"])
        return model