# Import packages
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import os

from matplotlib import pyplot as plt

from my_modules.nsclc import patient_wise_train_test_splitter
import torchvision.transforms.v2 as tvt

from my_modules.custom_models import *
from my_modules.model_learning import train_epoch, valid_epoch, masked_loss
from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc.nsclc_dataset import NSCLCDataset


def main():
    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device=torch.device('cpu'),
                        label='Metastases', mask_on=True)
    data.normalize_channels('preset')
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.to(torch.device(device))
    data.augment()

    # Dataloader parameters
    batch_size = 32
    data_split = [0.8, 0.15, 0.05]
    set_lengths = [round(len(data) * fraction) for fraction in data_split]
    set_lengths[-1] = (set_lengths[-1] - 1 if np.sum(set_lengths) > len(data) else set_lengths[-1])

    # Set up hyperparameters
    epochs = [125, 250, 500, 1000]
    learning_rates = [1e-6]

    # Set up training functions
    optimizers = {'Adam': [optim.Adam, {}]}
    loss_function = masked_loss(nn.BCEWithLogitsLoss())

    # Model zoo for images
    models = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
              CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet]

    # region Augmented Images
    # Augment and recreated the dataloaders
    train_set, test_set = patient_wise_train_test_splitter(data, n=3)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nAugmented Images\n')

    # Model path
    try:
        os.mkdir('aug_img_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        # Iterate through sets of hyperparameters
        for lr in learning_rates:
            for name, (optim_fn, options) in optimizers.items():
                model = m(data.shape)
                if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
                    model.to(torch.device(device))
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                for ep in range(epochs[-1]):
                    # Train
                    model.train()
                    total_loss = 0
                    for x, target in train_loader:
                        with torch.autocast(device_type=device):
                            out = model(x)
                            loss = loss_function(out, target.unsqueeze(1), torch.all((~torch.isnan(x)), dim=1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        train_loss.append(total_loss)

                    # Test
                    if ep + 1 in epochs:
                        torch.save(model.state_dict(), f'aug_img_models/{data.name}__{model.name}__{lr}_{ep}.pth')
                        print(
                            f'>>> {model.name} for {ep + 1} epochs with learning rate of {lr} using {name} optimizer...')
                        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
                        fig.savefig(f'aug_img_models/{data.name}__{model.name}__{lr}_{ep}.png')
                        plt.close(fig)

                        with open(results_file_path, 'a') as f:
                            f.write(f'\n>>> {model.name} for {ep + 1} epochs with learning rate of {lr}\n')
                            for key, item in scores.items():
                                if 'Confusion' not in key:
                                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                            f.write('_____________________________________________________\n')

    # endregion

    # region Augmented Histograms
    # Transform and create the dataloaders
    data.dist_transform(nbins=25)
    train_set, test_set = patient_wise_train_test_splitter(data, n=3)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Prep results file
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nAugmented Histograms\n')

    # Update Model zoo for histograms
    models = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
              RNNet, RegularizedRNNet, ParallelRNNet, RegularizedParallelRNNet]

    # Update loss function (no need for masked loss)
    loss_function = nn.BCEWithLogitsLoss()

    try:
        os.mkdir('aug_hist_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        # Iterate through sets of hyperparameters
        for lr in learning_rates:
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                model = m(data.shape)
                if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
                    model.to(torch.device(device))
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                for ep in range(epochs[-1]):
                    # Train
                    model.train()
                    total_loss = 0
                    for x, target in train_loader:
                        with torch.autocast(device_type=device):
                            out = model(x)
                            loss = loss_function(out, target.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        train_loss.append(total_loss)

                    # Test
                    if ep + 1 in epochs:
                        torch.save(model.state_dict(), f'aug_hist_models/{data.name}__{model.name}__{lr}_{ep}.pth')
                        print(
                            f'>>> {model.name} for {ep + 1} epochs with learning rate of {lr} using {name} optimizer...')
                        with torch.autocast(device_type=device):
                            scores, fig = score_model(model, test_loader,
                                                      print_results=True, make_plot=True, threshold_type='roc')
                        fig.savefig(f'aug_img_models/{data.name}__{model.name}__{lr}_{ep}.png')
                        plt.close(fig)
                        plt.close('all')

                        with open(results_file_path, 'a') as f:
                            f.write(f'\n>>> {model.name} for {ep + 1} epochs with learning rate of {lr}\n')
                            for key, item in scores.items():
                                if 'Confusion' not in key:
                                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                            f.write('_____________________________________________________\n')
    # endregion

# Run
if __name__ == '__main__':
    main()
