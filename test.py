import torch
import numpy as np

from data.data_generators.QuadNIST import QuadNIST
from data.data_generators.ShiftedMNIST import ShiftedMNIST
from torch.utils.data import DataLoader
from src.QuadNISTClassifier import QuadNISTClassifier, NoiseAwareQuadNISTClassifier
from src.QuadNISTDiffusion import QuadNISTDiffusion
from src.ShiftedMNISTSegmenter import SimpleUNet
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# SEED = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

CLASSIF_PATH = './models/NoiseAwareQuadNISTClassif/20261012_122331_T206/epoch_14.pt'
SEGMENTER_PATH = './models/ShiftedMNISTSegmenter/20261028_135658/step_20000.pt'
DIFFUSION_PATH = './models/ShiftedMNISTDiffusion/20261025_083548/step_200000.pt'

def testQuadNIST():
    NUM_SAMPLES = 16        # Number of diffusion samples to display

    ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    validation_dataset = QuadNIST(train=False, root='./data')

    model = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    model = model

    print(len(validation_dataset))

    dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)


    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
        
        # Gemini: If generating only 1 sample, wrap axes array for indexing safety
    with torch.no_grad():
        if NUM_SAMPLES == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        
        data_iterator = iter(dataloader)
        for idx in range(NUM_SAMPLES):
            image, label = next(data_iterator)

            # Gemini: Squeeze down channel index to form clean grayscale images
            image = image.squeeze()

            label = model(image.reshape(1, -1))

            label = f"{torch.argmax(label['digit']).item()}: {torch.exp(torch.max(label['digit'])).item():.2f}, {torch.argmax(label['quadrant']).item()+1}: {torch.exp(torch.max(label['quadrant'])).item():.2f}"
            
            axes[idx].imshow(image, cmap='gray')
            axes[idx].set_title(True if label is None else f"Label: {label}")
            axes[idx].axis('off')

        plt.tight_layout()
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
    NUM_SAMPLES = 15        # Number of diffusion samples to display

    ROWS = NUM_SAMPLES//3
    COLS = 3*3

    validation_dataset = ShiftedMNIST(train=False, root='./data')

    model = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    model = model

    print(len(validation_dataset))

    dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
        
        # Gemini: If generating only 1 sample, wrap axes array for indexing safety
    with torch.no_grad():
        if NUM_SAMPLES == 1:
            axes = np.expand_dims(axes, axis=0)

        
        data_iterator = iter(dataloader)
        for idx in range(NUM_SAMPLES):
            tdig = 0
            pdig = 0
            # while tdig == pdig:
            imaget, label = next(data_iterator)

            r = idx // 3

            c = idx % 3 * 3

            # Gemini: Squeeze down channel index to form clean grayscale images
            image = imaget.squeeze().numpy()
            label = label.squeeze().numpy()

            axes[r, c].imshow(image, cmap='gray')
            axes[r, c].set_title(f"Sample {idx+1}: Input Image")
            axes[r, c].axis('off')

            axes[r, c+1].imshow(label, cmap='tab20', vmin=0, vmax=10)
            
            unique_classes = np.unique(label)
            detected_digits = [int(c) for c in unique_classes if c != 10]
            digit_str = str(detected_digits[0]) if detected_digits else "None"
            tdig = detected_digits[0]
            
            axes[r, c+1].set_title(f"Target Mask (Digit: {digit_str})")
            axes[r, c+1].axis('off')

            segm = SimpleUNet(1,11).to(device)

            segm.load_state_dict(torch.load(SEGMENTER_PATH))


            pred = torch.max(segm(imaget.to(device)), 1)[1]

            pred = pred.squeeze().cpu().numpy()

            axes[r, c+2].imshow(pred, cmap='tab20', vmin=0, vmax=10)
                        
            unique_classes, counts = np.unique(pred, return_counts=True)
            sorted_indices = np.argsort(-counts)
            unique_classes = unique_classes[sorted_indices]
            
            detected_digits = [int(c) for c in unique_classes if c != 10]
            digit_str = f"{detected_digits[0]}" if detected_digits else "None"
            pdig = detected_digits[0]
            
            axes[r, c+2].set_title(f"Pred. Mask (Digit: {digit_str})")
            axes[r, c+2].axis('off')

        plt.tight_layout()
        plt.show()


        # for i, (images, gt_labels) in enumerate(dataloader):
            # ADD CODE TO CHECK ACCURACY OF THE SEGMENTATION MODEL HERE (just by majority non bg pixels)




def testQuadNISTDiffusion():
    NUM_SAMPLES = 45        # Number of diffusion samples to display
    CHANNELS = 1            # Number of channels per sample
    HEIGHT, WIDTH = 56, 56  # Image height and width

    # ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    ROWS = 5
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    EQUAL_NOISE = True

    # Instantiate model from checkpoint file
    print(f"Loading Diffusion weights from: {DIFFUSION_PATH}")
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Calculate LD step sizes from given sigmas (from NCSNv2)
    epsilon = 5e-7      # Epsilon, should be calculated with Technique 4 from NCSNv2
    etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    budgets = [200] * model.num_timesteps  # Refinement steps per noise level block

    # Generate images
    print("Executing backward diffusion sample generation...")
    with torch.no_grad():
        print("max noise: ", model.sigmas[0])
        # Start from noise
        seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)
        print(seed_noise[0].shape)
        if EQUAL_NOISE:
            for i in range(NUM_SAMPLES-1):
                seed_noise[i+1] = seed_noise[0]
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
        label = f"{torch.argmax(labels['digit'][idx]).item()}: {torch.exp(torch.max(labels['digit'][idx])).item():.2f}, {torch.argmax(labels['quadrant'][idx]).item()+1}: {torch.exp(torch.max(labels['quadrant'][idx])).item():.2f}"
        axes[idx].set_title(label)
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
    epsilon = 5e-7
    etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    budgets = [200] * model.num_timesteps  # 200 refinement steps per noise level block

    # Generate images
    print("Executing backward diffusion sample generation...")
    with torch.no_grad():
        print("max noise: ", model.sigmas[0])
        # Best: omega 200 and epsilon 1e-6
        # Start from noise
        seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)
        target = {'digit': torch.full((NUM_SAMPLES,), 4, dtype=torch.long, device=device), 'quadrant': torch.full((NUM_SAMPLES,), 3, dtype=torch.long, device=device)}
        generated_images = model.sample(etas=etas, budgets=budgets, noise=seed_noise, classifier=classifier, omega=500, target=target, omega_scaling=False)
        
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
        label = f"{torch.argmax(labels['digit'][idx]).item()}: {torch.exp(torch.max(labels['digit'][idx])).item():.2f}, {torch.argmax(labels['quadrant'][idx]).item()+1}: {torch.exp(torch.max(labels['quadrant'][idx])).item():.2f}"
        axes[idx].set_title(label)
        axes[idx].axis('off')

    plt.tight_layout(wspace=1)
    plt.show()

if __name__ == "__main__":
    # testQuadNIST()
    # testShiftedMNIST()
    testQuadNISTDiffusion()
    # testGuidedQuadNISTDiffusion()