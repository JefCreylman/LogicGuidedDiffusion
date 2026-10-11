import torch
import numpy as np

from data.data_generators.QuadNIST import QuadNIST
from data.data_generators.ShiftedMNIST import ShiftedMNIST
from torch.utils.data import DataLoader
from src.QuadNISTClassifier import QuadNISTClassifier, NoiseAwareQuadNISTClassifier
from src.QuadNISTDiffusion import QuadNISTDiffusion
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

CLASSIF_PATH = './models/NoiseAwareQuadNISTClassif/20261010_152904_T10/epoch_6.pt'
DIFFUSION_PATH = './models/QuadNISTDiffusion/20261011_111601/epoch_26_might_be_best.pt'

def testQuadNIST():
    NUM_SAMPLES = 5

    validation_dataset = QuadNIST(train=False, root='./data')

    model = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    model = model

    print(len(validation_dataset))

    dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(NUM_SAMPLES):
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

        for i, (images, gt_labels) in enumerate(dataloader):
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

                if total_samples == len(validation_dataset) or i%100 == 99:
                    print(f"\nEvaluation Results across {total_samples} samples:")
                    print(f"Correct Digits: {correct_digits}/{total_samples} ({100 * correct_digits / total_samples:.2f}%)")
                    print(f"Correct Quadrants: {correct_quads}/{total_samples} ({100 * correct_quads / total_samples:.2f}%)")
                    print(f"Perfect Matches (Both): {correct_both}/{total_samples} ({100 * correct_both / total_samples:.2f}%)")

def testShiftedMNIST():
    NUM_SAMPLES = 5

    validation_dataset = ShiftedMNIST(train=False, root='./data')

    model = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    model = model

    print(len(validation_dataset))

    dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(NUM_SAMPLES):
            image, label = next(data_iterator)

            image = image.squeeze()

            label = model(image.reshape(1, -1))
            label = (torch.argmax(label['digit']).item(), torch.argmax(label['quadrant']).item()+1)

            plt.imshow(image, cmap='gray')
            plt.title(True if label is None else f"Label: {label}")
            plt.axis('off')
            plt.show()

def testQuadNISTDiffusion():
    NUM_SAMPLES = 16        # Number of diffusion samples to display
    CHANNELS = 1            # Number of channels per sample
    HEIGHT, WIDTH = 56, 56  # Image height and width

    ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    # Instantiate model from checkpoint file
    print(f"Loading Diffusion weights from: {DIFFUSION_PATH}")
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Calculate LD step sizes from given sigmas (from NCSNv2)
    epsilon = 1e-7      # Epsilon, should be calculated with Technique 4 from NCSNv2
    etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    budgets = [200] * model.num_timesteps  # Refinement steps per noise level block

    # Generate images
    print("Executing backward diffusion sample generation...")
    with torch.no_grad():
        # Start from noise
        seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)
        generated_images = model.sample(etas=etas, budgets=budgets, noise=seed_noise)
        
        # Gemini: Norm and clamp output boundary distributions smoothly
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        classifier = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH)).to(device)

        labels = classifier(generated_images.reshape(generated_images.shape[0], -1))


    # Visualize outputs
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
    
    # Gemini: If generating only 1 sample, wrap axes array for indexing safety
    if NUM_SAMPLES == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx in range(NUM_SAMPLES):
        # Gemini: Squeeze down channel index to form clean grayscale images
        img = generated_images[idx].squeeze().cpu().numpy()
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"Sample {idx + 1}: ({torch.argmax(labels['digit'][idx]).item()}, Q{torch.argmax(labels['quadrant'][idx]).item()+1})")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

def testGuidedQuadNISTDiffusion():
    NUM_SAMPLES = 16        # Number of diffusion samples to display
    CHANNELS = 1            # Number of channels per sample
    HEIGHT, WIDTH = 56, 56  # Image height and width

    ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    # Instantiate model from checkpoint file
    print(f"Loading Diffusion weights from: {DIFFUSION_PATH}")
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    classifier = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    classifier = classifier.to(device)
    classifier.eval()

    # Calculate LD step sizes from given sigmas (from NCSNv2)
    epsilon = 1e-7
    etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    budgets = [200] * model.num_timesteps  # 200 refinement steps per noise level block

    # Generate images
    print("Executing backward diffusion sample generation...")
    with torch.no_grad():
        # Start from noise
        seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)
        target = {'digit': torch.full((NUM_SAMPLES,), 4, dtype=torch.long, device=device), 'quadrant': torch.full((NUM_SAMPLES,), 3, dtype=torch.long, device=device)}
        generated_images = model.sample(etas=etas, budgets=budgets, noise=seed_noise, classifier=classifier, omega=2, target=target)
        
        # Gemini: Norm and clamp output boundary distributions smoothly
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        labels = classifier(generated_images.reshape(generated_images.shape[0], -1))


    # Visualize outputs
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
    
    # Gemini: If generating only 1 sample, wrap axes array for indexing safety
    if NUM_SAMPLES == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx in range(NUM_SAMPLES):
        # Gemini: Squeeze down channel index to form clean grayscale images
        img = generated_images[idx].squeeze().cpu().numpy()
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"Sample {idx + 1}: ({torch.argmax(labels['digit'][idx]).item()}, Q{torch.argmax(labels['quadrant'][idx]).item()+1})")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # testQuadNIST()
    # testShiftedMNIST()
    # testQuadNISTDiffusion()
    testGuidedQuadNISTDiffusion()