import numpy as np
import torch
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from .loader_maker import loader_maker
from .model_phases import train_epoch, valid_epoch, test_model


def single_model_iterator(models, dataset, epochs, batch_size, criterion, optim_fun, num_workers=(0, 0, 0),
                          prefetch_factor=None, pin_memory=None, **optim_kwargs):
    # Set default pin memory behavior
    if pin_memory is None:
        pin_memory = (False if dataset.device is torch.device('cuda') else True)

    results = {}

    # Create data loaders
    train_loader, val_loader, test_loader = loader_maker(dataset,
                                                         batch_size=batch_size,
                                                         split=(0.75, 0.2, 0.05),
                                                         num_workers=num_workers,
                                                         shuffle=(True, False, False),
                                                         prefetch_factor=prefetch_factor,
                                                         pin_memory=pin_memory)

    data_shape = dataset.shape

    for net in models:
        #################
        # Prepare model #
        #################
        model = net(data_shape)
        model = model.to('cuda') if torch.cuda.is_available() else model
        model_path = f'./{model.name}_Epochs-{epochs}_{dataset.name}.pth'

        optimizer = optim_fun(model.parameters(), **optim_kwargs)

        ############
        # Training #
        ############
        tran_loss = []
        eval_accu = []
        eval_loss = []

        for epoch in range(epochs):
            print(f'>>>{model.name} Epoch {epoch+1}/{epochs}...')
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
        accu = correct / len(test_loader.dataset)

        results[model.name] = accu
        print(f'|-- {model.name} accuracy on {dataset.name}: {accu:.2%} --|')

    return results


def fold_model_iterator(models, datasets, folds, epochs, batch_size, criterion, optim_fun, workers=(0, 0),
                        prefetch_factor=None, **optim_kwargs):
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

                # Create Kth Fold Dataloaders by subsampling set
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
                optimizer = optim_fun(model.parameters(), **optim_kwargs)

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
            show_results(model.name, data.name, accu, tran_loss, eval_loss, eval_accu)

    return results, k_results


def show_results(name, set_name, accu, tran_loss, eval_loss, eval_accu):
    # Plot losses and eval accuracy from each epoch
    print(f'|-- {name} accuracy on {set_name}: {accu:.2%} --|')
    _, ax = plt.subplots(1, 2)
    ax[0].plot(tran_loss)
    ax[0].plot(eval_loss)
    ax[1].plot(eval_accu)
    plt.show()
