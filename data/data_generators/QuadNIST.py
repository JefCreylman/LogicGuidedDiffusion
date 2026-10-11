import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class QuadNIST(Dataset):
    '''
    Modified MNIST Datset with digits in one of four quadrants of the image
    
    Args:
        root (str or pathlib.Path):
            Root directory of dataset where MNIST/raw/train-images-idx3-ubyte and MNIST/raw/t10k-images-idx3-ubyte exist.
        
        train (bool, optional):
            If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
        '''        
    def __init__(self, root, train = True):
        self.mnist = datasets.MNIST(
            root = root,
            train = train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return 4*len(self.mnist)

    def __getitem__(self, idx):
        '''
        Returns item with given id

        Args:
            idx (int):
                id of requested item

        Returns:
            image (Tensor):
                tensor of pixel values
            label (dict):
                contains 'digit', the original mnist label, and 'quadrant', the quadrant the digit is in
        '''
        original_image, original_label = self.mnist[idx//4]
        quadrant = idx % 4
        
        match quadrant:
            case 0:
                quadrantTensor = torch.tensor([[0., 1.],[0., 0.]])
            case 1:
                quadrantTensor = torch.tensor([[1., 0.],[0., 0.]])
            case 2:
                quadrantTensor = torch.tensor([[0., 0.],[1., 0.]])
            case 3:
                quadrantTensor = torch.tensor([[0., 0.],[0., 1.]])
            case _:
                print(f"Case not allowed: {quadrant} from {idx}")

        image = torch.kron(quadrantTensor, original_image)
        label = {'digit': original_label, 'quadrant': quadrant + 1}
 
        return image, label
