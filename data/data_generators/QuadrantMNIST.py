import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class QuadrantMNIST(Dataset):
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
