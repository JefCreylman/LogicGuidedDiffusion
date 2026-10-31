import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader

from src.QuadNISTDiffusion import QuadNISTDiffusion
from data.data_generators.QuadNIST import QuadNIST


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
        avg_loss (float):
            Average loss over the entire epoch
    '''
    running_loss = 0.
    total_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, _ = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        loss = model.calculate_loss(inputs)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        if i%1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    avg_loss = total_loss / len(training_loader)

    return last_loss, avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    '''
    TRAINING PARAMETERS:
    '''
    BATCH_SIZE = 128
    EPOCHS = 50
    PATIENCE = 5
    NUM_WORKERS = 0         

    '''
    MODEL PARAMETERS:
        Sigma1 Changed back to 1, following comment explains how it should be chosen according to NCSNv2    
            LOGSIGMA1 is obtained using Technique 1 from Improved Techniques for Training Score-Based Generative Models:
                (Initial noise scale). Choose sigma1 to be as large as the maximum Euclidean distance
                between all pairs of training data points.

            I approximate the maximum by following logic: 
                At least 2 quadrant always stay the same, so at most 2*28*28 pixels change.
                Assume at most half of the pixels of a quadrant are lit up maximum. As such maximum 28*28 pixels change
                Pixel values are ranged [0, 1], so maximum change is 1
                As such sigma1 should be sqrt(28*28*1) = 28

        NUM_TIMESTEPS is gotten from the parameterFinder for NCSNv2, which approximates C = 0.5
    '''
    LOGSIGMA1 = 0
    NUM_TIMESTEPS = 120

    sigmas = torch.logspace(start=LOGSIGMA1, end=-2, steps=NUM_TIMESTEPS).tolist()   

    #Load datasets
    training_set = QuadNIST(train=True, root='./data')
    validation_set = QuadNIST(train=False, root='./data')

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    #Prepare model and optimizer
    model = QuadNISTDiffusion(sigmas).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

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
        last_loss, avg_loss = train_one_epoch(epoch_number)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, _ = vdata
                vinputs = vinputs.to(device)
                vloss = model.calculate_loss(vinputs)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i+1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        if avg_vloss < best_vloss:
            no_improvement = 0
            best_vloss = avg_vloss
            model_path = f'./models/QuadNISTDiffusion/{timestamp}'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.get_checkpoint(), os.path.join(model_path, f'epoch_{epoch_number}.pt'))
        elif PATIENCE >= 0: 
            no_improvement += 1
            print(f'LOSS did not improve for epoch {epoch_number + 1}')
            if no_improvement > PATIENCE:
                print(f'LOSS did not improve for {no_improvement} epochs, breaking due to early stopping')
                break

        print("",flush=True)

        epoch_number += 1