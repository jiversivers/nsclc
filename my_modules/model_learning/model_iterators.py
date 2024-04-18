import numpy as np
import torch
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from my_modules.model_learning.model_phases import train_epoch, valid_epoch, test_model


def single_model_iterator(models, datasets, epochs, batch_size, criterion, optim_fun, num_workers=(0, 0, 0),
                          prefetch_factor=None, **kwargs):
    results = {}
    for key, data in datasets.items():
        results[key] = {}
        ######################
        # Create dataloaders #
        ######################
        train_size = int(0.75 * len(data))
        val_size = int(0.2 * train_size)
        test_size = len(data) - (train_size + val_size)
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset=data,
            lengths=[train_size, val_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers[0],
            prefetch_factor=prefetch_factor)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers[1],
            prefetch_factor=prefetch_factor)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers[2],
            prefetch_factor=prefetch_factor)

        data_shape = data[0][0].shape

        for net in models:
            #################
            # Prepare model #
            #################
            model = net(data_shape)
            model = model.to('cuda') if torch.cuda.is_available() else model
            model_path = f'./{key}_{model.name}.pth'

            optimizer = optim_fun(model.parameters(), **kwargs)

            ############
            # Training #
            ############
            tran_loss = []
            eval_accu = []
            eval_loss = []

            for epoch in range(epochs):
                print(f'>>>{model.name} Epoch {epoch+1}...')
                epoch_loss = train_epoch(model, train_loader, criterion, optimizer)
                tran_loss.append(epoch_loss)

                val_loss, val_accu = valid_epoch(model, val_loader, criterion)
                eval_loss.append(val_loss)
                eval_accu.append(val_accu)
                print(f'>>>Train Loss: {epoch_loss} >>> Eval Loss: {val_loss}. Accu: {val_accu}.')

                # Save best performing model
                if epoch == 0:
                    best_acc = eval_accu[-1]
                    torch.save(model.state_dict(), model_path)
                elif eval_accu[-1] > best_acc:
                    torch.save(model.state_dict(), model_path)
                    best_acc = eval_accu[-1]

            ###########
            # Testing #
            ###########
            model = model.to('cuda') if torch.cuda.is_available() else model
            model.load_state_dict(torch.load(model_path))

            correct = test_model(model, test_loader)

            # Testing results
            accu = correct / len(test_set)

            results[key][model.name] = accu
            show_results(model.name, accu, tran_loss, eval_loss, eval_accu)

    return results


def fold_model_iterator(models, datasets, folds, epochs, batch_size, criterion, optim_fun, workers=(0, 0),
                        prefetch_factor=None, **kwargs):
    kfold = KFold(n_splits=folds, shuffle=True)
    k_results = {}
    results = {}
    for key, data in datasets.items():
        data_shape = data[0][0].shape
        results[key] = {}
        k_results[key] = {}
        for net in models:
            model = net(data_shape)

            tran_loss = []
            eval_loss = []
            eval_accu = []
            k_results[key][model.name] = []
            for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(data)))):
                ###############################
                # Create Kth Fold Dataloaders #
                ###############################
                train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
                test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
                train_loader = torch.utils.data.DataLoader(data,
                                                           batch_size=batch_size,
                                                           sampler=train_sampler,
                                                           drop_last=True,
                                                           num_workers=workers[0],
                                                           prefetch_factor=prefetch_factor)
                test_loader = torch.utils.data.DataLoader(data,
                                                          batch_size=batch_size,
                                                          sampler=test_sampler,
                                                          drop_last=True,
                                                          num_workers=workers[2],
                                                          prefetch_factor=prefetch_factor)

                #######################
                # Prepare fresh model #
                #######################
                model = net(data_shape)
                model = model.to('cuda') if torch.cuda.is_available() else model
                model_path = f'./{model.name}_{key}_fold{fold + 1}.pth'
                optimizer = optim_fun(model.parameters(), **kwargs)

                for _ in range(epochs):
                    ############
                    # Training #
                    ############
                    epoch_loss = train_epoch(model, train_loader, criterion, optimizer)
                    tran_loss.append(epoch_loss)

                    ##############
                    # Evaluation #
                    ##############
                    val_loss, val_accu = valid_epoch(model, test_loader, criterion)
                    eval_loss.append(val_loss)
                    eval_accu.append(val_accu)

                ###########
                # Testing #
                ###########
                correct = test_model(model, test_loader)
                k_results[key][model.name].append(correct / len(test_loader.sampler))
                torch.save(model.state_dict(), model_path)

            accu = sum(k_results[key][model.name]) / folds

            results[key][model.name] = accu
            show_results(model.name, accu, tran_loss, eval_loss, eval_accu)

    return results, k_results


def show_results(name, accu, tran_loss, eval_loss, eval_accu):
    # Plot losses and eval accuracy from each epoch
    print(f'|-- {name} accuracy on {set}: {accu:.2%} --|')
    _, ax = plt.subplots(1, 2)
    ax[0].plot(tran_loss)
    ax[0].plot(eval_loss)
    ax[1].plot(eval_accu)
    plt.show()
