# Import packages
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import numpy as np
import os

from my_modules.custom_models import *
from my_modules.model_learning import train_epoch, valid_epoch, test_model
from my_modules.nsclc.nsclc_dataset import NSCLCDataset

def main():
    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases')
    print('Normalizing data to channel max...')
    data.normalize_channels_to_max()

    # Dataloader parameters
    batch_size = 32
    data_split = [0.8, 0.15, 0.05]
    set_lengths = [round(len(data) * fraction) for fraction in data_split]
    set_lengths[-1] = (set_lengths[-1] - 1 if np.sum(set_lengths) > len(data) else set_lengths[-1])
    if torch.cuda.is_available():
        workers = [0, 0, 0]
    else:
        workers = [round(0.75 * mp.cpu_count() * fraction) for fraction in data_split]

    # Set up hyperparameters
    epochs = [125, 250, 500]
    learning_rates = [0.1, 0.01, 0.001]

    # Set up training functions
    optimizers = {'Adam': [optim.Adam, {}],
                  'SGD': [optim.SGD, {'momentum': 0.9}]}
    loss_function = nn.BCELoss()

    # region Raw Images
    # Make raw image dataloaders
    train_set, eval_set, test_set = torch.utils.data.random_split(dataset=data, lengths=set_lengths)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers[0],
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=workers[1],
                                              drop_last=(True if len(eval_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers[2],
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Model zoo for images
    models = [MLPNet, RegularizedMLPNet, RegularizedParallelMLPNet, CNNet, RegularizedCNNet, RegularizedParallelCNNet]
    try:
        os.mkdir('raw_img_models')
    except FileExistsError:
        pass

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'w') as results_file:
        results_file.write('Results\nRaw Images\n')

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for epoch, lr in zip(epochs, learning_rates):
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} on {next(model.parameters()).device.type}')
                print(f'{epoch} epochs with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                eval_loss = []
                eval_accu = []
                for ep in range(epoch):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer))

                    # Validate
                    model.eval()
                    loss, accu = valid_epoch(model, eval_loader, loss_function)
                    eval_loss.append(loss)
                    eval_accu.append(accu)

                    print(f'Epoch: {ep + 1} -- Training loss: {train_loss[-1]:.4f} -- '
                          f'Evaluation loss: {eval_loss[-1]:.4f} -- Evaluation accu: {eval_accu[-1]:.4f}')

                    # Save the model if it's better
                    if ep == 0 or eval_accu[-1] > best_accu:
                        best_accu = eval_accu[-1]
                        print(f' =====>>> Saving new best model! -- Evaluation accuracy: {best_accu * 100:.2f}%')
                        torch.save(model.state_dict(), f'raw_img_models/{data.name}__{model.name}.pth')

                # Test
                model.load_state_dict(torch.load(f'raw_img_models/{data.name}__{model.name}.pth'))
                correct = test_model(model, test_loader)
                print(f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                      f'learning rate of {lr} using {name} optimizer -- '
                      f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')

                with open(results_file_path, 'a') as f:
                    f.write(f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                            f'learning rate of {lr} using {name} optimizer -- '
                            f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%\n')

    # endregion

    # region Augmented Images
    # Augment and recreated the dataloaders
    train_set.dataset.augment()
    eval_set.dataset.augment()
    test_set.dataset.augment()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=workers[0],
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=workers[1],
                                              drop_last=(True if len(eval_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers[2],
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Prep results file
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nAugmented Images\n')

    # Model path
    try:
        os.mkdir('aug_img_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for epoch, lr in zip(epochs, learning_rates):
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} on {next(model.parameters()).device.type}')
                print(f'{epoch} epochs with learning rate of {lr} with {name} optimizer')
                print('_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                eval_loss = []
                eval_accu = []
                for ep in range(epoch):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer))

                    # Validate
                    model.eval()
                    loss, accu = valid_epoch(model, eval_loader, loss_function)
                    eval_loss.append(loss)
                    eval_accu.append(accu)

                    print(f'Epoch: {ep + 1} -- Training loss: {train_loss[-1]:.4f} -- '
                          f'Evaluation loss: {eval_loss[-1]:.4f} -- Evaluation accu: {eval_accu[-1]:.4f}')

                    # Save the model if it's better
                    if ep == 0 or eval_accu[-1] > best_accu:
                        best_accu = eval_accu[-1]
                        print(f' =====>>> Saving new best model! -- Evaluation accuracy: {best_accu * 100:.2f}%')
                        torch.save(model.state_dict(), f'aug_img_models/{data.name}__{model.name}.pth')

                # Test
                model.load_state_dict(torch.load(f'aug_img_models/{data.name}__{model.name}.pth'))
                correct = test_model(model, test_loader)
                print(f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                      f'learning rate of {lr} using {name} optimizer -- '
                      f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')

                with open(results_file_path, 'a') as f:
                    f.write(
                        f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                        f'learning rate of {lr} using {name} optimizer -- '
                        f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%\n')

    # endregion

    # region Augmented Histograms
    # Transform and create the dataloaders
    train_set.dataset.dist_transform()
    eval_set.dataset.dist_transform()
    test_set.dataset.dist_transform()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=workers[0],
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=workers[1],
                                              drop_last=(True if len(eval_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers[2],
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nAugmented Histograms\n')

    # Update Model zoo for histograms
    models = [MLPNet, RegularizedMLPNet, RegularizedParallelMLPNet, RNNet, RegularizedRNNet, RegularizedParallelRNNet]
    try:
        os.mkdir('aug_hist_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for epoch, lr in zip(epochs, learning_rates):
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} on {next(model.parameters()).device.type}')
                print(f'{epoch} epochs with learning rate of {lr} with {name} optimizer')
                print(
                    '_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                eval_loss = []
                eval_accu = []
                for ep in range(epoch):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer))

                    # Validate
                    model.eval()
                    loss, accu = valid_epoch(model, eval_loader, loss_function)
                    eval_loss.append(loss)
                    eval_accu.append(accu)

                    print(f'Epoch: {ep + 1} -- Training loss: {train_loss[-1]:.4f} -- '
                          f'Evaluation loss: {eval_loss[-1]:.4f} -- Evaluation accu: {eval_accu[-1]:.4f}')

                    # Save the model if it's better
                    if ep == 0 or eval_accu[-1] > best_accu:
                        best_accu = eval_accu[-1]
                        print(f' =====>>> Saving new best model! -- Evaluation accuracy: {best_accu * 100:.2f}%')
                        torch.save(model.state_dict(), f'aug_hist_models/{data.name}__{model.name}.pth')

                # Test
                model.load_state_dict(torch.load(f'aug_hist_models/{data.name}__{model.name}.pth'))
                correct = test_model(model, test_loader)
                print(f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                      f'learning rate of {lr} using {name} optimizer -- '
                      f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')

                with open(results_file_path, 'a') as f:
                    f.write(
                        f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                        f'learning rate of {lr} using {name} optimizer -- '
                        f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%\n')
    # endregion

    # region Raw Histograms
    # Transform and create the dataloaders
    train_set.dataset.augmented = False
    eval_set.dataset.augmented = False
    test_set.dataset.augmented = False
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=workers[0],
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=True, num_workers=workers[1],
                                              drop_last=(True if len(eval_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers[2],
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nRaw Histograms\n')

    # Model path
    try:
        os.mkdir('raw_hist_models')
    except FileExistsError:
        pass

    # Iterate through all models
    for m in models:
        model = m(data.shape)
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)
        # Iterate through sets of hyperparameters
        for epoch, lr in zip(epochs, learning_rates):
            # Iterate through optimizing functions
            for name, (optim_fn, options) in optimizers.items():
                optimizer = optim_fn(model.parameters(), lr=lr, **options)
                print(f'Training model {model.name} on {next(model.parameters()).device.type}')
                print(f'{epoch} epochs with learning rate of {lr} with {name} optimizer')
                print(
                    '_____________________________________________________________________________________________\n')
                # For each epoch
                train_loss = []
                eval_loss = []
                eval_accu = []
                for ep in range(epoch):
                    # Train
                    model.train()
                    train_loss.append(train_epoch(model, train_loader, loss_function, optimizer))

                    # Validate
                    model.eval()
                    loss, accu = valid_epoch(model, eval_loader, loss_function)
                    eval_loss.append(loss)
                    eval_accu.append(accu)

                    print(f'Epoch: {ep + 1} -- Training loss: {train_loss[-1]:.4f} -- '
                          f'Evaluation loss: {eval_loss[-1]:.4f} -- Evaluation accu: {eval_accu[-1]:.4f}')

                    # Save the model if it's better
                    if ep == 0 or eval_accu[-1] > best_accu:
                        best_accu = eval_accu[-1]
                        print(f' =====>>> Saving new best model! -- Evaluation accuracy: {best_accu * 100:.2f}%')
                        torch.save(model.state_dict(), f'raw_hist_models/{data.name}__{model.name}.pth')

                # Test
                model.load_state_dict(torch.load(f'raw_hist_models/{data.name}__{model.name}.pth'))
                correct = test_model(model, test_loader)
                print(f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                      f'learning rate of {lr} using {name} optimizer -- '
                      f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')

                with open(results_file_path, 'a') as f:
                    f.write(
                        f' =====>>> {model.name} on {next(model.parameters()).device.type} for {epoch} epochs with '
                        f'learning rate of {lr} using {name} optimizer -- '
                        f'Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%\n')
    # endregion


# Run
if __name__ == '__main__':
    main()
