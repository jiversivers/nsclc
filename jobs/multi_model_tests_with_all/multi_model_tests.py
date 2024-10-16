# Import packages
import os

import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
import random

from pretrainedmodels import inceptionresnetv2, xception

from my_modules.custom_models import *
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import set_seed
from my_modules.nsclc.nsclc_dataset import NSCLCDataset


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################
    # Prepare data #
    ################
    data = NSCLCDataset('NSCLC_Data_for_ML', ['fad', 'nadh', 'shg', 'orr', 'intensity'], device=torch.device('cpu'),
                        label='Metastases', mask_on=True)
    data.augment()
    data.normalize_method = 'preset'
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.to(device)

    # Random split datasets
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(data.patient_count))
    idx = [i for i in subsampler]

    # Get the image indices for all patients as nested lists
    patient_subsets = [data.get_patient_subset(i) for i in idx]

    # Find and remove any patients with no image indices
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
    for ix in idx_for_removal:
        idx.remove(ix)

    # Get labels for all remaining patients
    labels = [data.get_patient_label(i).item() for i in idx]
    image_counts = [0, 0]
    for i, label in zip(idx, labels):
        image_counts[int(label)] += len(data.get_patient_subset(i))

    # Separate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]
    print(f'Total non-metastatic patients: {len(shuffled_ones)} with {image_counts[1]} images')
    print(f'Total metastatic patients: {len(shuffled_zeros)} with {image_counts[0]} images')

    # Split train and test sets
    # Test set from first three patients (of shuffled groups) for each class
    test_pts = shuffled_zeros[0:3] + shuffled_ones[0:3]
    test_idx = [data.get_patient_subset(i) for i in test_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)
    image_counts = [0, 0]
    for idx in test_pts:
        label = data.get_patient_label(idx)
        image_counts[int(label)] += len(data.get_patient_subset(idx))
    print(f'Testing set\n'
          f'___________\n'
          f'Non-metastatic: {len(shuffled_ones[0:3])} with {image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[0:3])} with {image_counts[0]} images.\n'
          f'Total: {len(test_pts)} Patients with {len(test_idx)} images.\n')

    train_pts = shuffled_zeros[3:] + shuffled_ones[3:]
    train_idx = [data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)
    image_counts = [0, 0]
    for idx in train_pts:
        label = data.get_patient_label(idx)
        image_counts[int(label)] += len(data.get_patient_subset(idx))
    print(f'Training set\n'
          f'____________\n'
          f'Non-metastatic: {len(shuffled_ones[3:])} with {image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[3:])} with {image_counts[0]} images.\n'
          f'Total: {len(train_pts)} Patients with {len(train_idx)} images.\n')

    print(f'Testing patients: {test_pts}.\nTraining patients: {train_pts}.')

    # Create dataloaders for fold
    batch_size = 64
    train_set = torch.utils.data.Subset(data, train_idx)
    test_set = torch.utils.data.Subset(data, test_idx)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_idx) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_idx) % batch_size == 1 else False))

    #####################
    # Prepare model zoo #
    #####################
    # ResNet18
    models = [ResNet18NPlaned(data.shape, start_width=64, n_classes=1)]

    # Basic CNNs
    models[len(models):] = [CNNet(data.shape),
                            RegularizedCNNet(data.shape),
                            ParallelCNNet(data.shape),
                            RegularizedParallelCNNet(data.shape)]

    # Put all models on GPU if available
    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

    ###################
    # Hyperparameters #
    ###################
    epochs = 1500
    learning_rate = 1e-5
    loss_function = nn.BCELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) for model in models]

    ###############
    # Output Prep #
    ###############
    # Prep results file
    for model in models:
        os.makedirs(f'outputs/{model.name}/plots', exist_ok=True)
        with open(f'outputs/{model.name}/results.txt', 'w') as f:
            f.write(f'{model.name} Results\n')

    # Model path
    try:
        os.mkdir('best_models')
    except FileExistsError:
        pass

    train_loss, eval_loss = 2 * [len(models) * [[]]]
    best_score = len(models) * [0]
    # For each epoch
    for ep in range(epochs):
        print(f'Epoch {ep}')
        epoch_loss = len(models) * [0]
        # Train
        for model in models:
            model.train()
        for x, target in train_loader:
            for i, model in enumerate(models):
                out = model(x)
                loss = loss_function(out, target.unsqueeze(1))
                loss.backward()
                epoch_loss[i] += loss.item()
                optimizers[i].step()
        for running, current in zip(train_loss, epoch_loss):
            running.append(current)

        # Evaluation
        for i, model in enumerate(models):
            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'Epoch {ep}\n'
                        f'{model.name}: Training loss: {train_loss[i]}. Evaluation scores:')
            print(f'>>> {model.name}: Training loss: {train_loss[i]}. Evaluation scores:')
            scores = score_model(model, test_loader, loss_fn=loss_function, print_results=True, make_plot=False,
                                 threshold_type='roc')
            eval_loss[i] = scores['Loss']
            if scores['ROC-AUC'] > best_score[i]:
                best_score[i] = scores['ROC-AUC']
                torch.save(model.state_dict(), f'best_models/Best {model.name}.pth')
                with open(f'outputs/{model.name}/results.txt', 'a') as f:
                    f.write(f'New best {model.name} saved at epoch {ep} with ROC-AUC of {scores["ROC-AUC"]}')

    # Plot epoch-wise outputs
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(models):
        plt.plot(range(1, epochs + 1), train_loss[i], label=f'{model.name} Training')
        plt.plot(range(1, epochs + 1), eval_loss[i], label=f'{model.name} Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/losses.png')

    # Testing
    for model in models:
        print(f'>>> {model.name}...')
        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'outputs/{model.name}/plots/test_results.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name}...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
            f.write('_____________________________________________________\n')

# Run
if __name__ == '__main__':
    main()
