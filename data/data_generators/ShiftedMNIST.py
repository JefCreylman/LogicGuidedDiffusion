import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class ShiftedMNIST(Dataset):
    '''
    Modified MNIST Datset with digits placed ad random into a 56x56 images.
    The original MNIST images are uniformly shifted 0 to 27 pixels horizontaly and 0 to 27 verticaly towards the bottom right.
    
    Args:
        root (str or pathlib.Path):
            Root directory of dataset where MNIST/raw/train-images-idx3-ubyte and MNIST/raw/t10k-images-idx3-ubyte exist.
        stochastic (bool):
            If True, each item has a randomly generated shift.
            If False, each MNIST item is copied 28*28 times, once for each shift combination 
        length (int):
            Allows for the dataset length selection if stochastic = True. Otherwise, does nothing.
        
        train (bool, optional):
            If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
        '''        
    def __init__(self, root, train = True, stochastic = False, length=240000):
        self.mnist = datasets.MNIST(
            root = root,
            train = train,
            download=True,
            transform=transforms.ToTensor()
        )
        if not stochastic:
            self.length = len(self.mnist)*28*28
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
        Returns item with given id

        Args:
            idx (int):
                id of requested item

        Returns:
            image (Tensor):
                tensor of pixel values
            label (Tensor):
                tensor of segmentation targets
        '''

        if self.stochastic:
            original_image, original_label = self.mnist[idx%len(self.mnist)]

            image = torch.zeros((56, 56))

            row = torch.randint(0, 28)
            column = torch.randint(0, 28)

            image[row:row+28, column:column+28] = original_image
        else:
            original_image, original_label = self.mnist[idx//(28*28)]

            image = torch.zeros((56, 56))

            row = (idx//28)%28
            column = idx%28
            
            image[row:row+28, column:column+28] = original_image


        label = torch.zeros(10, image.shape[0], image.shape[1])
        label[original_label] = image
 
        return image, label
