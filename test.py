import os
import time
import csv

import torch
import numpy as np



from torcheval.metrics import FrechetInceptionDistance
from data.data_generators.QuadNIST import QuadNIST, FilteredQuadNIST
from data.data_generators.ShiftedMNIST import ShiftedMNIST
from torch.utils.data import DataLoader, Dataset
from src.QuadNISTClassifier import QuadNISTClassifier, NoiseAwareQuadNISTClassifier
from src.QuadNISTDiffusion import QuadNISTDiffusion
from src.ShiftedMNISTSegmenter import SimpleUNet
from src.MNISTLeNet import LargeLeNet5
from src.Logics import FuzzyLogic
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# SEED = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

CLASSIF_PATH = './models/NoiseAwareQuadNISTClassif/20261012_122331_T206/epoch_14.pt'
SEGMENTER_PATH = './models/ShiftedMNISTSegmenter/20261028_135658/step_20000.pt'
# DIFFUSION_PATH = './models/ShiftedMNISTDiffusion/20261025_083548/step_200000.pt'
DIFFUSION_PATH = './models/QuadNISTDiffusion/20261104_163409/epoch_195.pt'
LENET_PATH = './models/QuadNISTLeNet/20261104_104125/epoch_49.pt'

class DiffusionDatasetGenerator(Dataset):
    '''
    Use to create a Dataset from a diffusion model for easier use
    '''
    def __init__(self, model, len=240000, batch_size=1, channels=1, noise_height=56, noise_width=56, epsilon=5e-6, budgets=20, classifier=None, omega=1, target=1, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.len = len
        self.batch_size = batch_size
        self.channels = channels
        self.noise_height = noise_height
        self.noise_width = noise_width

        self.classifier = classifier
        self.omega = omega
        if callable(target):
            self.target = target
        else:
            self.target = {'digit': torch.full((self.batch_size,), target['digit'], dtype=torch.long, device=device), 'quadrant': torch.full((self.batch_size,), target['quadrant'], dtype=torch.long, device=device)}

        self.device = device
        
        self.etas = [epsilon * (s ** 2) / (self.model.sigmas[-1] ** 2) for s in self.model.sigmas]
        self.budgets = [budgets] * self.model.num_timesteps


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seed_noise = torch.randn(self.batch_size, self.channels, self.noise_height, self.noise_width, device=self.device)

        if self.classifier == None:
            res = self.model.sample(etas=self.etas, budgets=self.budgets, noise=seed_noise)
        else:
            res = self.model.sample(etas=self.etas, budgets=self.budgets, noise=seed_noise, classifier=self.classifier, omega=self.omega, target=self.target)


        return res.cpu()


def testQuadNIST():
    NUM_SAMPLES = 16        # Number of diffusion samples to display

    ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    validation_dataset = QuadNIST(train=False, root='./data')

    model = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    model = model

    # print(len(validation_dataset))

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


def testQuadNISTLeNet():
    validation_dataset = QuadNIST(train=False, root='./data')

    model = LargeLeNet5.load_model_from_checkpoint(torch.load(LENET_PATH))
    
    with torch.no_grad():
        dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
        
        total_samples = 0
        correct_digits = 0

        for i, (images, gt_labels) in enumerate(dataloader):
                labels = model(images)

                pred_digits = torch.argmax(labels, dim = 1)

                total_samples += images.size(0)

                digit_matches = torch.eq(pred_digits, gt_labels['digit'])

                correct_digits += digit_matches.sum().item()

                if total_samples == len(validation_dataset) or i%100 == 99:
                    print(f"\nEvaluation Results across {total_samples} samples:")
                    print(f"Perfect Matches (Both): {correct_digits}/{total_samples} ({100 * correct_digits / total_samples:.2f}%)")


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
    BATCH_SIZE = 128        # Number of samples to generate per batch
    CHANNELS = 1            # Number of channels per sample
    HEIGHT, WIDTH = 56, 56  # Image height and width

    # ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    ROWS = 5
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    EQUAL_NOISE = False

    # Instantiate model from checkpoint file
    print(f"Loading Diffusion weights from: {DIFFUSION_PATH}")
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Calculate LD step sizes from given sigmas (from NCSNv2)
    # epsilon = 5e-7      # Epsilon, should be calculated with Technique 4 from NCSNv2
    # etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    # budgets = [200] * model.num_timesteps  # Refinement steps per noise level block

    # Generate images
    # print("Executing backward diffusion sample generation...")
    # with torch.no_grad():
    #     print("max noise: ", model.sigmas[0])
    #     # Start from noise
    #     seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)
    #     print(seed_noise[0].shape)
    #     if EQUAL_NOISE:
    #         for i in range(NUM_SAMPLES-1):
    #             seed_noise[i+1] = seed_noise[0]
    #     start = time.time()
    #     generated_images = model.sample(etas=etas, budgets=budgets, noise=seed_noise)
    #     end = time.time()

    #     print(f"Base took {end-start}s")
    #     # Gemini: Norm and clamp output boundary distributions smoothly
    #     generated_images = torch.clamp(generated_images, 0.0, 1.0)

    with torch.no_grad():
        start = time.time()
        generated_dataset = DiffusionDatasetGenerator(model, batch_size=BATCH_SIZE, device=device)
        generated_images = generated_dataset[0].to(device)
        end = time.time()
        print(f"Generation took {end-start}s, that is {(end-start)/BATCH_SIZE}s per sample")
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
    NUM_SAMPLES = 45        # Number of diffusion samples to display
    CHANNELS = 1            # Number of channels per sample
    HEIGHT, WIDTH = 56, 56  # Image height and width

    # ROWS = int(np.ceil(np.sqrt(NUM_SAMPLES)))
    ROWS = 5
    COLS = int(np.ceil(NUM_SAMPLES/ROWS))

    TAR_DIG = 4
    TAR_QUAD = 3
    def fuzzy_rule(output):
                    log_p_dig4 = output['digit'][:, 4]
                    log_p_dig6 = output['digit'][:, 6]
                    log_p_quad3 = output['quadrant'][:, 2]
                    log_p_quad1 = output['quadrant'][:, 0]
                    return FuzzyLogic.log_or(FuzzyLogic.log_and(log_p_dig4, log_p_quad3), FuzzyLogic.log_and(log_p_dig6, log_p_quad1))
                    # return FuzzyLogic.log_and(log_p_dig4, log_p_quad3)

    # Instantiate model from checkpoint file
    print(f"Loading Diffusion weights from: {DIFFUSION_PATH}")
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    classifier = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH))
    classifier = classifier.to(device)
    classifier.eval()

    # Calculate LD step sizes from given sigmas (from NCSNv2)
    epsilon = 5e-6
    etas = [epsilon * (s ** 2) / (model.sigmas[-1] ** 2) for s in model.sigmas]
    budgets = [20] * model.num_timesteps  # 20 refinement steps per noise level block

    # Generate images
    print("Executing backward diffusion sample generation...")
    with torch.no_grad():
        print("max noise: ", model.sigmas[0])
        # Start from noise
        seed_noise = model.sigmas[0]*torch.randn(NUM_SAMPLES, CHANNELS, HEIGHT, WIDTH, device=device)

        # target = {'digit': torch.full((NUM_SAMPLES,), TAR_DIG, dtype=torch.long, device=device), 'quadrant': torch.full((NUM_SAMPLES,), TAR_QUAD, dtype=torch.long, device=device)}
    
        target = fuzzy_rule

        start = time.time()
        generated_images = model.sample(etas=etas, budgets=budgets, noise=seed_noise, classifier=classifier, omega=50, target=target, omega_scaling=False)
        end = time.time()
        print(f"Generation took {end-start}s, that is {(end-start)/NUM_SAMPLES}s per sample")

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

    correct = 0

    for idx in range(NUM_SAMPLES):
        pred_dig = torch.argmax(labels['digit'][idx]).item()
        pred_quad = torch.argmax(labels['quadrant'][idx]).item() + 1
        if (pred_dig == 4 and pred_quad == 3) or (pred_dig == 6 and pred_quad == 1):
            correct += 1
        # Gemini: Squeeze down channel index to form clean grayscale images
        img = generated_images[idx].squeeze().cpu().numpy()
        
        axes[idx].imshow(img, cmap='gray')
        label = f"{pred_dig}: {torch.exp(torch.max(labels['digit'][idx])).item():.2f}, {pred_quad}: {torch.exp(torch.max(labels['quadrant'][idx])).item():.2f}"
        axes[idx].set_title(label)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print(f'{correct/NUM_SAMPLES*100:.2f}% of samples corresponded to target')


def testQuadNISTDiffFID():
    SAMPLES_PER_FID = 2560
    BATCH_SIZE = 128
    
    TAR_DIG = 4
    TAR_QUAD = 3
    def fuzzy_rule(output):
        log_p_dig4 = output['digit'][:, 4]
        log_p_dig6 = output['digit'][:, 6]
        log_p_quad3 = output['quadrant'][:, 2]
        log_p_quad1 = output['quadrant'][:, 0]
        return FuzzyLogic.log_or(FuzzyLogic.log_and(log_p_dig4, log_p_quad3), FuzzyLogic.log_and(log_p_dig6, log_p_quad1))

    # OMEGAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    OMEGAS = [200, 500, 1000, 30, 40, 60, 70]

    CSV_FILE = "./metrics/logicGuidedDiffusionMetrics.csv"

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    lenet = LargeLeNet5.load_model_from_checkpoint(torch.load(LENET_PATH)).to(device)
    model = QuadNISTDiffusion.load_model_from_checkpoint(torch.load(DIFFUSION_PATH, map_location=device)).to(device)
    classifier = NoiseAwareQuadNISTClassifier.load_model_from_checkpoint(torch.load(CLASSIF_PATH)).to(device)

    lenet.eval()
    model.eval()
    classifier.eval()

    target = {'digit': TAR_DIG, 'quadrant': TAR_QUAD}
    
    # target = fuzzy_rule

    with torch.no_grad():
        for omega in OMEGAS:
            print(f'OMEGA = {omega}')
            print(f'===============')
            '''Might want to change this so the real images only get added once'''
            full_fid = FrechetInceptionDistance(lenet.feature_extractor, 120, device)
            target_fid = FrechetInceptionDistance(lenet.feature_extractor, 120, device)

            # validation_dataset = ShiftedMNIST(train=False, root='./data')
            validation_dataset = QuadNIST(train=False, root='./data')
            validation_dataloader = DataLoader(validation_dataset, BATCH_SIZE, shuffle=True)
            validation_iterator = iter(validation_dataloader)

            target_dataset = FilteredQuadNIST(train=False, root='./data', combinations=[(4,3)])
            target_dataloader = DataLoader(target_dataset, BATCH_SIZE, shuffle=True)
            target_iterator = iter(target_dataloader)

            diff_dataset = DiffusionDatasetGenerator(model, batch_size=BATCH_SIZE, device=device, classifier=classifier, omega = omega, target=target)


            # print(diff_dataset[0].shape)

            correct = 0

            for i in range(SAMPLES_PER_FID//BATCH_SIZE):
                print(f"BATCH {i+1}/{SAMPLES_PER_FID//BATCH_SIZE}", flush=True)
                res = diff_dataset[i].to(device)
                full_fid.update(next(validation_iterator)[0].to(device).expand(-1, 3, -1, -1), True)
                target_fid.update(next(target_iterator)[0].to(device).expand(-1,3,-1,-1), True)

                expanded = res.expand(-1, 3, -1, -1)
                full_fid.update(expanded, False)
                target_fid.update(expanded, False)

                res = torch.clamp(res, 0.0, 1.0)
                labels = classifier(res.reshape(res.shape[0], -1))
                for j in range(BATCH_SIZE):
                    pred_dig = torch.argmax(labels['digit'][j]).item()
                    pred_quad = torch.argmax(labels['quadrant'][j]).item() + 1
                    if pred_dig == TAR_DIG and pred_quad == TAR_QUAD:
                        correct += 1

            full_fid_score = full_fid.compute().item()
            target_fid_score = target_fid.compute().item()
            alignment_rate = correct/SAMPLES_PER_FID*100

            
            print(f'Full FID: {full_fid_score}')
            print(f'Target FID: {target_fid_score}')
            print(f'Target allignment rate: {alignment_rate:.2f}%\n', flush=True)

            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([omega, full_fid_score, target_fid_score, f"{alignment_rate:.2f}"])



if __name__ == "__main__":
    # testQuadNIST()
    # testQuadNISTLeNet()
    # testShiftedMNIST()
    # testQuadNISTDiffusion()
    testGuidedQuadNISTDiffusion()
    # testQuadNISTDiffFID()