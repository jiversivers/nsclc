# Import packages
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import os

from matplotlib import pyplot as plt

from my_modules.custom_models import *
from my_modules.model_learning import train_epoch, valid_epoch, masked_loss
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import NSCLCDataset, patient_wise_train_test_splitter


def main():
    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    # mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    data = NSCLCDataset('E:/NSCLC_Data_for_ML', ['orr'], device='cpu', use_cache=True, use_atlas=True,
                        use_patches=True, patch_size=(256, 256), label='Metastases')
    data.to(device)

    # Dataloader parameters
    batch_size = 64
    epochs = [1, 125, 250, 500, 1000, 2000]
    learning_rates = [1e-5, 5e-6, 1e-6]

    # Set up training functions
    optimizers = {'Adam': [optim.Adam, {}]}
    loss_function = nn.BCEWithLogitsLoss()

    # region Images
    # Make raw image dataloaders
    train_set, test_set = patient_wise_train_test_splitter(data, n=3)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Model zoo for images
    models = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
              CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet]
    try:
        os.mkdir('img_models')
    except FileExistsError:
        pass

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'w') as results_file:
        results_file.write('Results\nImages\n')

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for lr in learning_rates:
            # Iterate through optimizing functions
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                for ep in range(epochs[-1]):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer))

                    # Test
                    if ep + 1 in epochs:
                        torch.save(model.state_dict(), f'img_models/{data.name}__{model.name}__{lr}_{ep + 1}.pth')
                        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
                        fig.savefig(f'img_models/{data.name}__{model.name}__{lr}_{ep + 1}.png')
                        plt.close(fig)
                        with open(results_file_path, 'a') as f:
                            f.write(f'\n>>> {model.name} for {ep + 1} epochs with learning rate of {lr}\n')
                            for key, item in scores.items():
                                if 'Confusion' not in key:
                                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                            f.write('_____________________________________________________\n')

    # endregion

    # region Histograms
    epochs = [125, 250, 500, 1000, 2000]
    learning_rates = [1e-4, 1e-5, 5e-6, 1e-6]
    data.dist_transform(nbins=25)

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'a') as results_file:
        results_file.write('\n Histograms\n')

    # Update Model zoo for histograms
    models = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
              RNNet, RegularizedRNNet, ParallelRNNet, RegularizedParallelRNNet]

    # Update loss function (no need for masked loss)
    loss_function = nn.BCEWithLogitsLoss()

    try:
        os.mkdir('hist_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for lr in learning_rates:
            # Iterate through optimizing functions
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                for ep in range(epochs[-1]):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer, masked_loss_fn=False))

                    # Test
                    if ep + 1 in epochs:
                        torch.save(model.state_dict(), f'hist_models/{data.name}__{model.name}__{lr}_{ep + 1}.pth')
                        print(
                            f'>>> {model.name} for {ep + 1} epochs with learning rate of {lr} using {name} optimizer...')
                        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
                        fig.savefig(f'hist_models/{data.name}__{model.name}__{lr}_{ep + 1}.png')
                        plt.close(fig)

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
