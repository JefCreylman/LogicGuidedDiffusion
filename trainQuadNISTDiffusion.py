import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.QuadNISTDiffusion import QuadNISTDiffusion
from data.data_generators.QuadNIST import QuadNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 1

training_set = QuadNIST(train=True, root='./data')
validation_set = QuadNIST(train=False, root='./data')

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

model = QuadNISTDiffusion().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
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
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        avg_loss = total_loss / len(training_loader)

    return last_loss, avg_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/QuadNIST_trainer_{timestamp}')
epoch_number = 0

best_vloss = 1_000_000.

no_improvement = 0

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    model.train(True)
    last_loss, avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, _ = vdata
            vinputs = vinputs.to(device)
            vloss = model.calculate_loss(vinputs)
            running_vloss += vloss

    avg_vloss = running_vloss / (i+1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    writer.add_scalars('Training vs. Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch_number + 1)

    writer.flush()

    if avg_vloss < best_vloss:
        no_improvement = 0
        best_vloss = avg_vloss
        model_path = f'./models/QuadNISTClassif/{timestamp}'
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