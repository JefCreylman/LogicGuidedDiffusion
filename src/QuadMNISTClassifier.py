import torch
from torch import nn

class QuadMNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        '''
        https://medium.com/@amdnewaz/in-this-article-well-walk-through-building-a-simple-neural-network-using-pytorch-to-classify-211b33ba5c62
        '''
        self.hidden = nn.Linear(28*28*4, 512)
        self.relu = nn.ReLU()
        self.outputD = nn.Linear(512, 10)
        self.outputQ = nn.Linear(512, 4)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        
        d = self.outputD(x)
        q = self.outputQ(x)

        #d = self.sigmoid(d)
        #q = self.sigmoid(q)

        d = self.softmax(d)
        q = self.softmax(q)

        return {'digit': d, 'quadrant': q}

    def calculate_loss(self, x, label):
        output = self(x)
        loss_fn = nn.NLLLoss()

        loss_d = loss_fn(output['digit'], label['digit'])
        loss_q = loss_fn(output['quadrant'], label['quadrant'] - 1)

        total_loss = loss_d + loss_q

        return total_loss
