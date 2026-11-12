import torch
import numpy as np
import os

from datetime import datetime

from torch.utils.data import DataLoader

from src.ShiftedMNISTSegmenter import SimpleUNet
from data.data_generators.ShiftedMNIST import ShiftedMNIST

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def train_x_steps(x, data_iterator):
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

    iterator_end = False

    for i in range(x):

        try:
            data = next(data_iterator)
        except StopIteration:
            iterator_end = True
            break

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss = model.calculate_loss(inputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        if i%1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}', flush=True)
            running_loss = 0.

    avg_loss = total_loss / x

    return last_loss, avg_loss, iterator_end


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    '''
    TRAINING PARAMETERS:
    '''
    STEP_BEFORE_SAVE = 10_000
    BATCH_SIZE = 128
    AMOUNT_SAVES = None         #Put at None to pass entire dataset once
    PATIENCE = -1
    NUM_WORKERS = 0

    ONE_PASS = True

    PATH = None
    START_STEP = 0
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
    training_set = ShiftedMNIST(train=True, root='./data')
    validation_set = ShiftedMNIST(train=False, root='./data', stochastic=True, length=10000)

    saves = int(np.ceil(len(training_set)/STEP_BEFORE_SAVE/BATCH_SIZE)) if AMOUNT_SAVES == None else AMOUNT_SAVES
        

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)



    #Prepare model and optimizer
    
    model = SimpleUNet(1, 11).to(device)
    if PATH != None:
        model.load_state_dict(torch.load(PATH))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    #Prepare timestamp for logging and saving model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_vloss = 1_000_000.     #Approximation for +infinity (loss should always be way lower)

    no_improvement = 0          #Tracker for PATIENCE

    data_iterator = iter(training_loader)

    if PATH != None:
        for i in range(START_STEP):
             data, index = next(data_iterator)              # TODO: Make this WAY MORE EFFICIENT cause this implementation is terrible

        print(f"Starting training at step {index+2}", flush=True)    


    iterator_end = False
    end_check = False

    for i in range(saves):
        '''
        Per x steps:
            1. Train x steps
            2. Check validation loss
            3. Check for validation loss improvement. If there is none, don't save model (Should probably be changed for crashes with high PATIENCE)
            4. If no improvement for longer than PATIENCE, stop training (i.e. Early stopping with patience)
        '''
        print(f'Steps {i*STEP_BEFORE_SAVE+1}-{(i+1)*STEP_BEFORE_SAVE}:')

        model.train(True)

        if iterator_end:
            end_check = True
            data_iterator = iter(training_loader)

        last_loss, avg_loss, iterator_end = train_x_steps(STEP_BEFORE_SAVE, data_iterator)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for v, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                vloss = model.calculate_loss(vinputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (v+1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        if avg_vloss < best_vloss:
            no_improvement = 0
            best_vloss = avg_vloss
            model_path = f'./models/ShiftedMNISTSegmenter/{timestamp}'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, f'step_{(i+1)*STEP_BEFORE_SAVE}.pt'))
        elif PATIENCE >= 0: 
            no_improvement += 1
            print(f'LOSS did not improve for steps {i*STEP_BEFORE_SAVE+1}-{(i+1)*STEP_BEFORE_SAVE}')
            if no_improvement > PATIENCE:
                print(f'LOSS did not improve for {no_improvement} times {STEP_BEFORE_SAVE} steps, breaking due to early stopping')
                break

        print("",flush=True)

        if end_check and ONE_PASS:
            break
        elif end_check:
            end_check = False