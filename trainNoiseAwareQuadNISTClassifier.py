import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader

from src.QuadNISTClassifier import NoiseAwareQuadNISTClassifier
from data.data_generators.QuadNIST import QuadNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5

'''
Following parameters should correspond to those of the diffusion model you are using
'''
LOGSIGMA1 = 1.45
NUM_TIMESTEPS = 206
sigmas = torch.logspace(start=0, end=-2, steps=NUM_TIMESTEPS).tolist()

#Load datasets
training_set = QuadNIST(train=True, root='./data')
validation_set = QuadNIST(train=False, root='./data')

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

#Prepare model and optimizer
model = NoiseAwareQuadNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def train_one_epoch(epoch_index):
    '''
    Functionality for one training epoch

    Passes all training data once and logs loss every 1000 baches

    Args:
        epoch_index (int):
            Index of current epoch (for logging purposes)

    Returns:
        last_loss (float):
            Loss of the last 1000 consecutive baches
    '''
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), {'digit': labels['digit'].to(device), 'quadrant': labels['quadrant'].to(device)}

        optimizer.zero_grad()

        batch_size = inputs.shape[0]

        sigmas_t = torch.tensor(sigmas, dtype=torch.float, device=device)
        random_indices = torch.randint(0, len(sigmas), (batch_size,), device=device)
        batch_sigmas = sigmas_t[random_indices].unsqueeze(1)

        inputs = inputs.reshape(inputs.shape[0], -1)

        loss = model.calculate_loss(inputs, labels, batch_sigmas)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i%1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

#Prepare timestamp for logging and saving model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

best_vloss = 1_000_000.     #Approximation for +infinity (loss should always be way lower)

no_improvement = 0          #Tracker for PATIENCE

for epoch in range(EPOCHS):
    '''
    Per epoch:
        1. Train one epoch
        2. Check validation loss
        3. Check for validation loss improvement. If there is none, don't save model (Should probably be changed for crashes with high PATIENCE)
        4. If no improvement for longer than PATIENCE, stop training (i.e. Early stopping with patience)
    '''
    print(f'EPOCH {epoch_number + 1}:')

    model.train(True)
    last_loss = train_one_epoch(epoch_number)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), {'digit': vlabels['digit'].to(device), 'quadrant': vlabels['quadrant'].to(device)}
            vinputs = vinputs.reshape(vinputs.shape[0], -1)
            vloss = model.calculate_loss(vinputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i+1)
    print(f'LOSS train {last_loss} valid {avg_vloss}')

    if avg_vloss < best_vloss:
        no_improvement = 0
        best_vloss = avg_vloss
        model_path = f'./models/NoiseAwareQuadNISTClassif/{timestamp}'
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_path, f'epoch_{epoch_number}.pt'))
    elif PATIENCE >= 0: 
        no_improvement += 1
        print(f'LOSS did not improve for epoch {epoch_number + 1}')
        if no_improvement > PATIENCE:
            print(f'LOSS did not improve for {no_improvement} epochs, breaking due to early stopping')
            break

    print("",flush=True)

    epoch_number += 1
