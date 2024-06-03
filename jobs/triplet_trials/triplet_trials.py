import os

import torch
import torch.multiprocessing as mp
from torch import nn

from my_modules.custom_models import MLPNet, RegularizedParallelMLPNet, RegularizedMLPNet, \
    ParallelMLPNet, CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet
from my_modules.model_learning.loader_maker import split_augmented_data
from my_modules.nsclc import NSCLCDataset


def main():
    # Set up multiprocessing context
    # mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data with psuedo-RGB stack for each mode
    data = NSCLCDataset(
        # 'data/NSCLC_Data_for_ML',
        'E:/NSCLC_Data_for_ML',
        mode=['orr', 'taumean', 'boundfraction'], label='M', mask_on=False, device='cpu')
    data.augment()
    data.normalize_channels_to_max()
    data.to(device)

    # Set up hyperparameters
    batch_size = 64
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [5, 10, 50, 125, 250, 500, 1000, 2500]
    epoch = sorted(epochs)[-1]
    optim_fn = torch.optim.Adam
    loss_fn = nn.TripletMarginLoss

    # Split dataset
    training_set, validation_set, testing_set = split_augmented_data(data, augmentation_factor=5, split=(.75, .15, .1))
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  drop_last=(True if len(training_set) % batch_size == 1 else False))
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                    drop_last=(
                                                        True if len(validation_set) % batch_size == 1 else False))
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 drop_last=(True if len(testing_set) % batch_size == 1 else False))

    # Set up models to try
    model_fns = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
                 CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet]

    # Make model save dir
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    with open('outputs/results.txt', 'w') as results_file:
        results_file.write(f'Results\n')

    for model_fn in model_fns:
        os.makedirs(f'outputs/models/{model_fn.__name__}', exist_ok=True)
        for lr in learning_rates[-1:]:
            # Make fresh model and optimizer
            model = model_fn(data.shape)
            model.to(device)
            optimizer = optim_fn(model.parameters(), lr=lr)

            # Training
            model.train()
            for ep in range(epoch):
                # Model the entire loader
                outs = torch.tensor([]).to(device)
                targets = torch.tensor([]).to(device)
                for x, target in training_loader:
                    outs = torch.cat((outs, model(x)), dim=0)
                    targets = torch.cat((targets, target), dim=0)

                # Regroup outputs based on class (random by first instance
                anchor = outs[targets == target[0]]
                negative = outs[targets != targets[0]]

                # Truncate to the smallest class
                truncation_index = min(anchor.shape[0], negative.shape[0])
                # Make positives from any leftovers with reordered data to prevent self-overlap with anchor
                positive = torch.cat((anchor[truncation_index:], anchor[0:truncation_index]), dim=0)
                positive = positive[:truncation_index]
                anchor = anchor[:truncation_index]
                negative = negative[:truncation_index]

                # Calculate loader-wide loss, backprop, and learn
                loss = loss_fn(anchor, positive, negative)
                loss = loss.sum() / len(training_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                validation_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for x, target in validation_loader:
                        out = model(x)
                        validation_loss += nn.BCELoss()(out, target.unsqueeze(1))
                        prediction = torch.round(out)
                        correct += torch.sum(prediction == target.unsqueeze(1)).item()
                validation_accuracy = 100 * correct / len(validation_set)

                print(f'{model.name} || Epoch {ep + 1} || Loss {loss.item():.4f} || '
                      f'Validation Loss: {batch_size * validation_loss / len(validation_loader)} Accuracy {validation_accuracy:.2f}%')

                # Testing
                if ep + 1 in epochs:
                    model.eval()
                    correct = 0
                    for x, target in testing_loader:
                        out = model(x)
                        prediction = torch.round(out)
                        correct += torch.sum(prediction == target.unsqueeze(1)).item()
                    testing_accuracy = 100 * correct / len(validation_set)
                    print(f'>>> {model.name} || LR {lr} ||  Epoch {ep + 1} || Accuracy {testing_accuracy:.2f} <<<')

        # Hard clear cache to try to keep memory clear
        torch.cuda.empty_cache()

    data.dist_transform(nbins=25)
    # Set up models to try
    model_fns = [MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet,
                 RNNet, RegularizedRNNet, ParallelRNNet, RegularizedParallelRNNet]



if __name__ == '__main__':
    main()
