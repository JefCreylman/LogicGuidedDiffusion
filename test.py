import torch
from data.data_generators.QuadrantMNIST import QuadrantMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

train_dataset = QuadrantMNIST(train=True, root='./data')

print(len(train_dataset))

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
data_iterator = iter(dataloader)

for i in range(10):
    image, label = next(data_iterator)

    label = tuple(map(lambda x: x.item(), label))

    imageGrid = image.squeeze()

    plt.imshow(imageGrid, cmap='gray')
    plt.title(True if label is None else f"Label: {label}")
    plt.axis('off')
    plt.show()