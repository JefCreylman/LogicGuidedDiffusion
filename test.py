import torch
from data.data_generators.QuadrantMNIST import QuadrantMNIST
from torch.utils.data import DataLoader
from src.QuadMNISTClassifier import QuadMNISTClassifier
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

PATH = './models/QuadNISTClassif/20261003_110845/epoch_19.pt'

validation_dataset = QuadrantMNIST(train=False, root='./data')

model = QuadMNISTClassifier()
model.load_state_dict(torch.load(PATH))
model = model

print(len(validation_dataset))

dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


with torch.no_grad():
    data_iterator = iter(dataloader)
    for i in range(5):
        image, label = next(data_iterator)

        image = image.squeeze()

        label = model(image.reshape(1, -1))
        label = (torch.argmax(label['digit']).item(), torch.argmax(label['quadrant']).item()+1)

        plt.imshow(image, cmap='gray')
        plt.title(True if label is None else f"Label: {label}")
        plt.axis('off')
        plt.show()

   
    dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
    
    total_samples = 0
    correct_digits = 0
    correct_quads = 0
    correct_both = 0

    for batch_idx, (images, gt_labels) in enumerate(dataloader):
            images = images.reshape(images.shape[0], -1)

            labels = model(images)

            pred_digits = torch.argmax(labels['digit'], dim = 1)
            pred_quads = torch.argmax(labels['quadrant'], dim = 1) + 1

            total_samples += images.size(0)

            digit_matches = torch.eq(pred_digits, gt_labels['digit'])
            quad_matches = torch.eq(pred_quads, gt_labels['quadrant'])

            correct_digits += digit_matches.sum().item()
            correct_quads += quad_matches.sum().item()

            correct_both += (digit_matches & quad_matches).sum().item()

            print(f"\nEvaluation Results across {total_samples} samples:")
            print(f"Correct Digits: {correct_digits}/{total_samples} ({100 * correct_digits / total_samples:.2f}%)")
            print(f"Correct Quadrants: {correct_quads}/{total_samples} ({100 * correct_quads / total_samples:.2f}%)")
            print(f"Perfect Matches (Both): {correct_both}/{total_samples} ({100 * correct_both / total_samples:.2f}%)")
    
